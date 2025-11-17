import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from pymongo import MongoClient
import requests
import os
import time
from datetime import datetime
import traceback
import json
from mimetypes import guess_type
import imghdr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_image")


def detect_image_content_type(filename: str, data: bytes) -> str:
    detected_type, _ = guess_type(filename)
    if detected_type and detected_type.startswith("image/"):
        return detected_type

    kind = imghdr.what(None, data)
    if kind:
        return f"image/{kind}"

    return "image/jpeg"


def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("[START] Upload_image function triggered")

    try:
        # Validate input
        file = req.files.get('file')
        if not file:
            logger.error("No file found in request.")
            return func.HttpResponse("No file uploaded", status_code=400)

        image_bytes = file.stream.read()
        image_name = file.filename
        logger.info(f"Received file: {image_name}")

        # Environment
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        mongo_uri = os.getenv("MONGO_URI")
        mongo_db_name = os.getenv("MONGO_DB", "yolov11db")
        mongo_collection = os.getenv("MONGO_COLLECTION", "yolov11-collection")
        yolo_endpoint = os.getenv("YOLO_ENDPOINT")

        if not blob_conn_str:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING")

        # Upload blob
        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)

        try:
            container_client.create_container()
        except Exception:
            pass

        content_type = detect_image_content_type(image_name, image_bytes)
        container_client.upload_blob(
            name=image_name,
            data=image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )

        blob_url = f"{container_client.url}/{image_name}"
        logger.info(f"Uploaded to Blob: {blob_url}")

        # MongoDB logging
        if mongo_uri:
            try:
                client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
                db = client[mongo_db_name]
                coll = db[mongo_collection]
                coll.insert_one({
                    "filename": image_name,
                    "blob_url": blob_url,
                    "upload_time": datetime.utcnow(),
                    "content_type": content_type,
                    "status": "uploaded"
                })
            except Exception as mongo_err:
                logger.error(f"MongoDB insert failed: {mongo_err}")

        # Routing logic
        lower_name = image_name.lower()
        if lower_name.startswith("dresscode_"):
            infer_url = f"{yolo_endpoint}/check_dresscode"
        elif lower_name.startswith("dustbin_"):
            infer_url = f"{yolo_endpoint}/dustbin_detect"
        else:
            infer_url = f"{yolo_endpoint}/infer"

        logger.info(f"Selected inference endpoint: {infer_url}")

        # Call YOLO API
        files = {"file": (image_name, image_bytes, content_type)}

        success = False
        for attempt in range(1, 3):
            try:
                response = requests.post(infer_url, files=files, timeout=40)
                logger.info(f"Inference status: {response.status_code}")

                if response.status_code == 200:
                    success = True
                    yolo_result = response.json()
                    break
            except Exception as e:
                logger.warning(f"Inference attempt {attempt} failed: {e}")
                time.sleep(1)

        if not success:
            yolo_result = {"error": "Inference failed"}

        total_time = round(time.time() - start_time, 2)
        logger.info(f"Completed in {total_time}s")

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "blob_url": blob_url,
                "prediction": yolo_result
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as ex:
        logger.error("Exception in Upload_image")
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json"
        )
