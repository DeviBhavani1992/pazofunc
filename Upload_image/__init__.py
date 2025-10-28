import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
import requests
import os
import time
from datetime import datetime
import traceback


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🔵 Azure Function 'Upload_image' triggered")

    try:
        # -------------------------
        # 🔹 Step 1: Validate input
        # -------------------------
        file = req.files.get('file')
        if not file:
            logging.error("❌ No file found in request.")
            return func.HttpResponse("No file uploaded", status_code=400)

        image_bytes = file.stream.read()
        image_name = file.filename
        logging.info(f"📁 Received file: {image_name} ({len(image_bytes)} bytes)")

        # -------------------------
        # 🔹 Step 2: Validate environment variables
        # -------------------------
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        mongo_uri = os.getenv("MONGO_URI")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "image_db")
        mongo_collection = os.getenv("MONGO_COLLECTION", "uploads")
        yolo_endpoint = os.getenv("YOLO_ENDPOINT")  # e.g. https://yolov11-app.<env>.azurecontainerapps.io

        logging.info(
            f"🧩 ENV CHECK → "
            f"AZURE_STORAGE_CONNECTION_STRING={'SET' if blob_conn_str else 'MISSING'}, "
            f"BLOB_CONTAINER_NAME={blob_container}, "
            f"MONGO_URI={'SET' if mongo_uri else 'MISSING'}, "
            f"YOLO_ENDPOINT={'SET' if yolo_endpoint else 'MISSING'}"
        )

        # -------------------------
        # 🔹 Step 3: Upload to Azure Blob Storage
        # -------------------------
        if not blob_conn_str:
            raise ValueError("Missing environment variable: AZURE_STORAGE_CONNECTION_STRING")

        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)
        container_client.upload_blob(name=image_name, data=image_bytes, overwrite=True)

        blob_url = f"{container_client.url}/{image_name}"
        logging.info(f"✅ Uploaded to Blob: {blob_url}")

        # -------------------------
        # 🔹 Step 4: Optional — Log metadata to MongoDB
        # -------------------------
        if mongo_uri:
            try:
                client = MongoClient(mongo_uri)
                db = client[mongo_db_name]
                collection = db[mongo_collection]
                doc = {
                    "filename": image_name,
                    "upload_time": datetime.utcnow(),
                    "blob_url": blob_url,
                }
                collection.insert_one(doc)
                logging.info(f"✅ Metadata inserted into MongoDB")
            except Exception as mongo_err:
                logging.error(f"⚠️ MongoDB insertion failed: {mongo_err}")
                logging.error(traceback.format_exc())
        else:
            logging.warning("⚠️ Skipping MongoDB logging (MONGO_URI not configured).")

        # -------------------------
        # 🔹 Step 5: Trigger YOLOv11 inference (auto scale-up)
        # -------------------------
        if yolo_endpoint:
            payload = {"blob_url": blob_url}
            for attempt in range(5):
                try:
                    r = requests.post(f"{yolo_endpoint}/infer", json=payload, timeout=10)
                    r.raise_for_status()
                    logging.info(f"✅ YOLOv11 inference sent: {r.status_code}")
                    logging.info(f"Response: {r.text}")
                    break
                except Exception as e:
                    logging.warning(f"⚠️ Attempt {attempt+1} failed: {e}")
                    time.sleep(5)
            else:
                logging.error("❌ YOLOv11 did not respond after multiple retries.")
        else:
            logging.warning("⚠️ YOLO_ENDPOINT not configured; skipping inference trigger.")

        # -------------------------
        # 🔹 Step 6: Return success
        # -------------------------
        return func.HttpResponse(
            f"✅ Upload successful! File URL: {blob_url}",
            status_code=200
        )

    except Exception as e:
        logging.error("🔥 Exception in Upload_image function")
        logging.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return func.HttpResponse(
            f"Internal server error: {str(e)}\n\n{traceback.format_exc()}",
            status_code=500
        )
