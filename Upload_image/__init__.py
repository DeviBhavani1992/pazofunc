import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
import os
from datetime import datetime
import traceback
from io import BytesIO
from PIL import Image


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🔵 Azure Function 'Upload_image' triggered")

    try:
        # -------------------------
        # 🔹 Step 1: Validate input
        # -------------------------
        logging.info("📥 Checking for uploaded file in request...")
        file = req.files.get('file')
        if not file:
            logging.error("❌ No file found in request.")
            return func.HttpResponse("No file uploaded", status_code=400)

        # Read file bytes
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

        logging.info(
            f"🧩 ENV CHECK → "
            f"AZURE_STORAGE_CONNECTION_STRING={'SET' if blob_conn_str else 'MISSING'}, "
            f"BLOB_CONTAINER_NAME={blob_container}, "
            f"MONGO_URI={'SET' if mongo_uri else 'MISSING'}, "
            f"MONGO_DB_NAME={mongo_db_name}, "
            f"MONGO_COLLECTION={mongo_collection}"
        )

        # -------------------------
        # 🔹 Step 3: Upload to Azure Blob Storage
        # -------------------------
        logging.info("📤 Connecting to Azure Blob Storage...")
        if not blob_conn_str:
            raise ValueError("Missing environment variable: AZURE_STORAGE_CONNECTION_STRING")

        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)

        logging.info(f"📦 Uploading file '{image_name}' to container '{blob_container}'...")
        container_client.upload_blob(name=image_name, data=image_bytes, overwrite=True)

        blob_url = f"{container_client.url}/{image_name}"
        logging.info(f"✅ Successfully uploaded to Blob: {blob_url}")

        # -------------------------
        # 🔹 Step 4: Optional — Log metadata to MongoDB
        # -------------------------
        if mongo_uri:
            try:
                logging.info("📡 Connecting to MongoDB...")
                client = MongoClient(mongo_uri)
                db = client[mongo_db_name]
                collection = db[mongo_collection]

                doc = {
                    "filename": image_name,
                    "upload_time": datetime.utcnow(),
                    "blob_url": blob_url,
                }

                collection.insert_one(doc)
                logging.info(f"✅ Inserted metadata into MongoDB: {mongo_db_name}.{mongo_collection}")

            except Exception as mongo_err:
                logging.error(f"⚠️ MongoDB insertion failed: {str(mongo_err)}")
                logging.error(traceback.format_exc())

        else:
            logging.warning("⚠️ Skipping MongoDB logging (MONGO_URI not configured).")

        # -------------------------
        # 🔹 Step 5: Return successful response
        # -------------------------
        logging.info("🎉 Upload operation completed successfully.")
        return func.HttpResponse(
            f"Upload successful! File URL: {blob_url}",
            status_code=200
        )

    except Exception as e:
        # -------------------------
        # 🔥 Step 6: Handle any exception
        # -------------------------
        error_trace = traceback.format_exc()
        logging.error("🔥 Exception in Upload_image function")
        logging.error(f"Error message: {str(e)}")
        logging.error(f"Traceback:\n{error_trace}")

        return func.HttpResponse(
            f"Internal server error: {str(e)}\n\nTraceback:\n{error_trace}",
            status_code=500
        )
