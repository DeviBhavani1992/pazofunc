import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
import requests
import os
import time
from datetime import datetime
import traceback

# -------------------------------------------------
# ✅ Configure logging for Azure Functions
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_image")

def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("🔵 [START] Azure Function 'Upload_image' triggered")

    try:
        # -------------------------------------------------
        # 🔹 Step 1: Validate input
        # -------------------------------------------------
        file = req.files.get('file')
        if not file:
            logger.error("❌ No file found in request.")
            return func.HttpResponse("No file uploaded", status_code=400)

        image_bytes = file.stream.read()
        image_name = file.filename
        logger.info(f"📁 Received file: {image_name} ({len(image_bytes)} bytes)")

        # -------------------------------------------------
        # 🔹 Step 2: Load environment variables
        # -------------------------------------------------
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        mongo_uri = os.getenv("MONGO_URI")
        mongo_db_name = os.getenv("MONGO_DB", "yolov11")
        mongo_collection = os.getenv("MONGO_COLLECTION", "detections")
        yolo_endpoint = os.getenv("YOLO_ENDPOINT")

        logger.info(
            f"🧩 ENV CHECK → "
            f"AZURE_STORAGE_CONNECTION_STRING={'SET' if blob_conn_str else '❌ MISSING'}, "
            f"BLOB_CONTAINER_NAME={blob_container}, "
            f"MONGO_URI={'SET' if mongo_uri else '❌ MISSING'}, "
            f"YOLO_ENDPOINT={'SET' if yolo_endpoint else '❌ MISSING'}"
        )

        # -------------------------------------------------
        # 🔹 Step 3: Upload image to Azure Blob
        # -------------------------------------------------
        upload_start = time.time()
        if not blob_conn_str:
            raise ValueError("Missing environment variable: AZURE_STORAGE_CONNECTION_STRING")

        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)

        container_client.upload_blob(name=image_name, data=image_bytes, overwrite=True)
        blob_url = f"{container_client.url}/{image_name}"

        upload_time = round(time.time() - upload_start, 2)
        logger.info(f"✅ Uploaded to Blob: {blob_url} (⏱️ {upload_time}s)")

        # -------------------------------------------------
        # 🔹 Step 4: Log metadata to Cosmos MongoDB
        # -------------------------------------------------
        if mongo_uri:
            try:
                mongo_start = time.time()
                client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
                db = client[mongo_db_name]
                collection = db[mongo_collection]
                doc = {
                    "filename": image_name,
                    "upload_time": datetime.utcnow(),
                    "blob_url": blob_url,
                    "status": "uploaded"
                }
                collection.insert_one(doc)
                mongo_time = round(time.time() - mongo_start, 2)
                logger.info(f"✅ Metadata inserted into Cosmos MongoDB (⏱️ {mongo_time}s)")
            except Exception as mongo_err:
                logger.error(f"⚠️ Cosmos MongoDB insertion failed: {mongo_err}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ MONGO_URI not configured — skipping MongoDB logging.")

        # -------------------------------------------------
        # 🔹 Step 5: Trigger YOLOv11 inference via ACA
        # -------------------------------------------------
        if yolo_endpoint:
            payload = {"blob_url": blob_url}
            inference_success = False

            for attempt in range(1, 6):
                try:
                    logger.info(f"🚀 Sending inference request to YOLOv11 (Attempt {attempt}) → {yolo_endpoint}/infer")
                    response = requests.post(f"{yolo_endpoint}/infer", json=payload, timeout=25)
                    response.raise_for_status()

                    logger.info(f"✅ YOLOv11 inference successful (HTTP {response.status_code})")
                    logger.info(f"🔍 Response: {response.text[:500]}")  # limit long responses
                    inference_success = True
                    break

                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt} failed: {str(e)}")
                    time.sleep(3)

            if not inference_success:
                logger.error("❌ YOLOv11 did not respond after 5 retries.")
        else:
            logger.warning("⚠️ YOLO_ENDPOINT not set — skipping inference trigger.")

        # -------------------------------------------------
        # 🔹 Step 6: Final summary + return
        # -------------------------------------------------
        total_time = round(time.time() - start_time, 2)
        logger.info(f"🏁 Upload_image completed successfully in {total_time}s")

        return func.HttpResponse(
            f"✅ Upload & inference completed successfully!\nFile URL: {blob_url}\nTotal Time: {total_time}s",
            status_code=200
        )

    except Exception as e:
        logger.error("🔥 Exception in Upload_image function")
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return func.HttpResponse(
            f"Internal server error: {str(e)}\n\n{traceback.format_exc()}",
            status_code=500
        )
