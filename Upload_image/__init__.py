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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "pazo")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "inference_logs")

AZURE_STORAGE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "uploads")

YOLO_ENDPOINT = os.getenv("YOLO_ENDPOINT")  # e.g. https://yolo-app.azurecontainerapps.io
if not YOLO_ENDPOINT:
    raise Exception("Missing YOLO_ENDPOINT environment variable!")

YOLO_ENDPOINT = YOLO_ENDPOINT.rstrip("/")

# ---------------------------------------------------------------------
# INFERENCE URLS (AUTO BUILT FROM YOLO_ENDPOINT)
# ---------------------------------------------------------------------
INFERENCE_ENDPOINTS = {
    "dresscode": f"{YOLO_ENDPOINT}/check_dresscode",
    "dustbin": f"{YOLO_ENDPOINT}/dustbin_detect",
    "default": f"{YOLO_ENDPOINT}/infer",
    "general": f"{YOLO_ENDPOINT}/infer",
}

# ---------------------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------------------
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
log_collection = mongo_db[MONGO_COLLECTION]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def upload_to_blob(container, blob_name, data, content_type):
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

        content_settings = ContentSettings(content_type=content_type)

        blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)

        return blob_client.url
    except Exception as e:
        logging.error(f"Blob upload error: {str(e)}")
        raise


def route_inference(category):
    """Return correct YOLO inference endpoint based on category."""
    return INFERENCE_ENDPOINTS.get(category.lower(), INFERENCE_ENDPOINTS["default"])


# ---------------------------------------------------------------------
# MAIN HTTP TRIGGER ENTRY
# ---------------------------------------------------------------------

async def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()

    try:
        category = req.params.get("category", "default")

        file = req.files.get("file")
        if not file:
            return func.HttpResponse("Image file missing", status_code=400)

        filename = file.filename
        content = file.stream.read()

        # Guess MIME type
        mime_type, _ = guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"

        # Validate image
        if not imghdr.what(None, h=content):
            return func.HttpResponse("Invalid image", status_code=400)

        # Upload to Blob
        blob_path = f"{category}/{filename}"
        blob_url = upload_to_blob(
            AZURE_BLOB_CONTAINER,
            blob_path,
            content,
            mime_type
        )

        # Select inference endpoint
        inference_url = route_inference(category)

        logging.info(f"Calling YOLO endpoint: {inference_url}")

        response = requests.post(
            inference_url,
            files={"file": (filename, content, mime_type)},
            timeout=120
        )

        inference_result = response.json() if response.ok else {"error": response.text}

        # Log to MongoDB
        log_entry = {
            "timestamp": datetime.utcnow(),
            "filename": filename,
            "category": category,
            "blob_url": blob_url,
            "inference_url": inference_url,
            "inference_result": inference_result,
            "execution_time_sec": round(time.time() - start_time, 3)
        }

        log_collection.insert_one(log_entry)

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "file_url": blob_url,
                "inference": inference_result
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
