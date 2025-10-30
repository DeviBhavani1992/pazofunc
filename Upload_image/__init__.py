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

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_image")

def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("🔵 [START] Upload_image function triggered")

    try:
        # Validate multipart/form-data file upload
        file = req.files.get('file')
        if not file:
            logger.error("❌ No file found in request (expecting form field 'file').")
            return func.HttpResponse("No file uploaded", status_code=400)

        image_bytes = file.stream.read()
        image_name = file.filename
        size_kb = round(len(image_bytes) / 1024, 1)
        logger.info(f"📁 Received file: {image_name} ({size_kb} KB)")

        # Environment variables
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        mongo_uri = os.getenv("MONGO_URI")
        mongo_db_name = os.getenv("MONGO_DB", "yolov11db")
        mongo_collection = os.getenv("MONGO_COLLECTION", "yolov11-collection")
        yolo_endpoint = os.getenv("YOLO_ENDPOINT")  # e.g. https://<app>.azurecontainerapps.io

        logger.info(
            f"ENV: BLOB_CONTAINER={blob_container}, MONGO_URI={'SET' if mongo_uri else 'MISSING'}, YOLO_ENDPOINT={'SET' if yolo_endpoint else 'MISSING'}"
        )

        if not blob_conn_str:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING")

        # -----------------------
        # Upload blob with correct content type (best practice)
        # -----------------------
        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)

        # Ensure container exists (idempotent)
        try:
            container_client.create_container()
            logger.info(f"Created blob container '{blob_container}'")
        except Exception:
            # ignore if exists or permission issues - container may already exist
            pass

        # Detect MIME type from filename; fallback to octet-stream
        detected_type, _ = guess_type(image_name)
        content_type = detected_type or "application/octet-stream"
        logger.info(f"Detected content-type for upload: {content_type}")

        # Upload with content settings (this ensures blob has proper Content-Type)
        try:
            container_client.upload_blob(
                name=image_name,
                data=image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )
        except Exception as e:
            logger.warning(f"Upload attempt failed: {e}; retrying once.")
            time.sleep(1)
            container_client.upload_blob(
                name=image_name,
                data=image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )

        blob_url = f"{container_client.url}/{image_name}"
        logger.info(f"✅ Uploaded to Blob: {blob_url}")

        # -----------------------
        # Insert metadata to Cosmos (Mongo API) if configured
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
                logger.info("✅ Inserted metadata into MongoDB (Cosmos).")
            except Exception as mongo_err:
                logger.error(f"⚠️ Cosmos insertion failed: {mongo_err}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ MONGO_URI not set — skipping metadata insert.")

        # -----------------------
        # Trigger YOLO inference (retries + proper headers)
        # -----------------------
        if yolo_endpoint:
            payload = {"blob_url": blob_url}
            headers = {"Content-Type": "application/json"}
            inference_success = False

            for attempt in range(1, 6):
                try:
                    logger.info(f"→ Inference attempt {attempt} to {yolo_endpoint}/infer")
                    # Use json= to let requests set the body correctly, and send explicit header too
                    response = requests.post(
                        f"{yolo_endpoint}/infer",
                        json=payload,
                        headers=headers,
                        timeout=30
                    )

                    logger.info(f"📨 Inference response status: {response.status_code}")
                    logger.info(f"📨 Inference response body (truncated): {response.text[:400]}")

                    response.raise_for_status()
                    inference_success = True
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Inference attempt {attempt} failed: {e}")
                    logger.debug(traceback.format_exc())
                    time.sleep(2)

            if not inference_success:
                logger.error("❌ YOLO inference failed after retries.")
        else:
            logger.warning("⚠️ YOLO_ENDPOINT not configured — skipping inference trigger.")

        total_time = round(time.time() - start_time, 2)
        logger.info(f"🏁 Upload_image finished in {total_time}s")

        return func.HttpResponse(
            f"OK: {blob_url}",
            status_code=200
        )

    except Exception as ex:
        logger.error("🔥 Exception in Upload_image function")
        logger.error(str(ex))
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            f"Internal server error: {str(ex)}",
            status_code=500
        )
