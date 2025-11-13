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

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://localhost:8000")  # 👈 change if needed
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "uploads")

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def detect_image_content_type(filename: str, data: bytes) -> str:
    detected_type, _ = guess_type(filename)
    if detected_type and detected_type.startswith("image/"):
        return detected_type
    kind = imghdr.what(None, data)
    return f"image/{kind or 'jpeg'}"

def call_inference(blob_url, mode):
    """Call YOLO inference API (dresscode/dustbin/infer)"""
    endpoint_map = {
        "dresscode": f"{INFERENCE_BASE_URL}/check_dresscode",
        "dustbin": f"{INFERENCE_BASE_URL}/dustbin_detect",
        "general": f"{INFERENCE_BASE_URL}/infer",
    }
    endpoint = endpoint_map.get(mode, endpoint_map["general"])
    logger.info(f"📡 Sending to inference endpoint: {endpoint}")

    try:
        resp = requests.post(endpoint, json={"blob_url": blob_url}, timeout=180)
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.error(f"❌ Inference API returned {resp.status_code}: {resp.text}")
            return {"error": f"Inference failed ({resp.status_code})", "blob_url": blob_url}
    except Exception as e:
        logger.error(f"⚠️ Inference request error: {e}")
        return {"error": str(e), "blob_url": blob_url}

# -------------------------------------------------------------------
# MAIN FUNCTION ENTRY
# -------------------------------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("🚀 [Azure Function Triggered] YOLOv11 unified pipeline")

    try:
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not blob_conn_str:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING in environment variables")

        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(BLOB_CONTAINER)

        try:
            container_client.create_container()
        except Exception:
            pass  # ignore if already exists

        # Handle files
        files = req.files.getlist("file") or list(req.files.values())
        if not files:
            logger.warning("⚠️ No files received in request")
            return func.HttpResponse(
                json.dumps({"error": "No image files uploaded"}),
                status_code=400,
                mimetype="application/json",
            )

        results = []
        for idx, file in enumerate(files, start=1):
            image_bytes = file.stream.read()
            filename = file.filename
            logger.info(f"📤 Uploading {filename} to blob storage...")

            content_type = detect_image_content_type(filename, image_bytes)
            container_client.upload_blob(
                name=filename,
                data=image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )

            blob_url = f"{container_client.url}/{filename}"
            logger.info(f"✅ Uploaded blob URL: {blob_url}")

            # Detect inference type based on file prefix
            if filename.lower().startswith("dresscode_"):
                analysis_type = "dresscode"
            elif filename.lower().startswith("dustbin_"):
                analysis_type = "dustbin"
            else:
                analysis_type = "general"

            # Call inference service
            inference_result = call_inference(blob_url, analysis_type)
            inference_result["blob_url"] = blob_url
            inference_result["filename"] = filename
            inference_result["type"] = analysis_type
            results.append(inference_result)

        total_time = round(time.time() - start_time, 2)
        logger.info(f"🏁 Pipeline completed in {total_time}s for {len(results)} image(s)")

        return func.HttpResponse(
            json.dumps({"results": results, "processing_time": total_time}),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logger.error(f"🔥 Exception in pipeline: {e}")
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
