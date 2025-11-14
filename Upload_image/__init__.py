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

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_image")


def detect_image_content_type(filename: str, data: bytes) -> str:
    """
    Robust MIME detection:
    1️⃣ Try Python mimetypes (by file extension)
    2️⃣ Fallback to imghdr (actual bytes)
    3️⃣ Default to image/jpeg if still unknown
    """
    detected_type, _ = guess_type(filename)
    if detected_type and detected_type.startswith("image/"):
        return detected_type

    # Fallback: detect from bytes
    kind = imghdr.what(None, data)
    if kind:
        return f"image/{kind}"

    return "image/jpeg"


def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("🔵 [START] Upload_image function triggered")

    try:
        # -----------------------
        # Validate multipart form upload
        # -----------------------
        file = req.files.get('file')
        if not file:
            logger.error("❌ No file found in request (expecting form field 'file').")
            return func.HttpResponse("No file uploaded", status_code=400)

        image_bytes = file.stream.read()
        image_name = file.filename
        size_kb = round(len(image_bytes) / 1024, 1)
        logger.info(f"📁 Received file: {image_name} ({size_kb} KB)")

        # -----------------------
        # Load environment variables
        # -----------------------
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        mongo_uri = os.getenv("MONGO_URI")
        mongo_db_name = os.getenv("MONGO_DB", "yolov11db")
        mongo_collection = os.getenv("MONGO_COLLECTION", "yolov11-collection")
        yolo_endpoint = os.getenv("YOLO_ENDPOINT")

        if not blob_conn_str:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING")

        logger.info(f"🌍 ENV: blob_container={blob_container}, YOLO_ENDPOINT={yolo_endpoint or 'MISSING'}")

        # -----------------------
        # Upload to Azure Blob Storage
        # -----------------------
        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)

        try:
            container_client.create_container()
        except Exception:
            pass  # Already exists

        content_type = detect_image_content_type(image_name, image_bytes)
        logger.info(f"🖼️ Detected content type for blob: {content_type}")

        container_client.upload_blob(
            name=image_name,
            data=image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )

        blob_url = f"{container_client.url}/{image_name}"
        logger.info(f"✅ Uploaded to Blob: {blob_url}")

        # -----------------------
        # Save metadata to MongoDB
        # -----------------------
        if mongo_uri:
            try:
                client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
                db = client[mongo_db_name]
                coll = db[mongo_collection]
                doc = {
                    "filename": image_name,
                    "blob_url": blob_url,
                    "upload_time": datetime.utcnow(),
                    "content_type": content_type,
                    "status": "uploaded"
                }
                coll.insert_one(doc)
                logger.info("✅ Inserted metadata into MongoDB.")
            except Exception as mongo_err:
                logger.error(f"⚠️ MongoDB insert failed: {mongo_err}")
                logger.debug(traceback.format_exc())
        else:
            logger.warning("⚠️ MONGO_URI not set — skipping Mongo logging.")

        # -----------------------
        # Trigger YOLO inference (multipart upload)
        # -----------------------
        if yolo_endpoint:

            files = {
                "file": (image_name, image_bytes, content_type)
            }

            success = False

            for attempt in range(1, 3):
                try:
                    logger.info(f"🚀 Sending inference request (attempt {attempt}) → {yolo_endpoint}/infer")

                    response = requests.post(
                        f"{yolo_endpoint}/infer",
                        files=files,     # <-- IMPORTANT: multipart form data
                        timeout=40
                    )

                    logger.info(f"📨 Status: {response.status_code}")
                    logger.debug(f"Response: {response.text[:400]}")

                    if response.status_code == 200:
                        success = True
                        logger.info("✅ Inference succeeded.")
                        break
                    else:
                        logger.warning(f"⚠️ Inference failed (status {response.status_code}): {response.text}")

                except Exception as e:
                    logger.warning(f"⚠️ Inference attempt {attempt} failed: {e}")
                    logger.debug(traceback.format_exc())
                    time.sleep(2)

            if not success:
                logger.error("❌ All inference attempts failed.")

        else:
            logger.warning("⚠️ YOLO_ENDPOINT not configured — skipping inference.")

        total_time = round(time.time() - start_time, 2)
        logger.info(f"🏁 Upload_image completed in {total_time}s")

        return func.HttpResponse(
            json.dumps({"status": "success", "blob_url": blob_url}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as ex:
        logger.error("🔥 Exception in Upload_image function")
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json"
        )
