import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from io import BytesIO
import json
import traceback

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("🚀 Azure Function 'Upload_image' triggered.")

    try:
        # --- Step 1: Read environment variables ---
        blob_conn_str = os.getenv("BLOB_CONNECTION_STRING")
        container_name = os.getenv("BLOB_CONTAINER_NAME")

        if not blob_conn_str:
            logger.error("❌ Missing environment variable: BLOB_CONNECTION_STRING")
            return func.HttpResponse("Missing BLOB_CONNECTION_STRING", status_code=500)
        if not container_name:
            logger.error("❌ Missing environment variable: BLOB_CONTAINER_NAME")
            return func.HttpResponse("Missing BLOB_CONTAINER_NAME", status_code=500)

        logger.info(f"✅ Environment variables loaded. Container: {container_name}")

        # --- Step 2: Initialize blob client ---
        try:
            blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
            container_client = blob_service_client.get_container_client(container_name)
        except Exception as e:
            logger.error(f"❌ Failed to initialize BlobServiceClient: {e}")
            return func.HttpResponse(f"Failed to connect to blob: {e}", status_code=500)

        # --- Step 3: Read request body ---
        file_bytes = req.get_body()
        if not file_bytes:
            logger.warning("⚠️ No file received in request body.")
            return func.HttpResponse("No file received", status_code=400)

        filename = req.headers.get("x-filename", f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{filename}"

        logger.info(f"📝 Uploading file: {blob_name} (size: {len(file_bytes)} bytes)")

        # --- Step 4: Upload file ---
        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(BytesIO(file_bytes), overwrite=True)
            logger.info(f"✅ File uploaded successfully as {blob_name}")
        except Exception as e:
            logger.error(f"❌ Blob upload failed: {e}")
            logger.debug(traceback.format_exc())
            return func.HttpResponse(f"Upload failed: {e}", status_code=500)

        # --- Step 5: Generate SAS URL ---
        try:
            account_name = blob_service_client.account_name
            account_key = blob_service_client.credential.account_key
            if not account_key:
                logger.warning("⚠️ Account key not found in credential — SAS URL might not work if using managed identity.")
                sas_token = None
            else:
                sas_token = generate_blob_sas(
                    account_name=account_name,
                    container_name=container_name,
                    blob_name=blob_name,
                    account_key=account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1),
                )

            if sas_token:
                blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
            else:
                blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"

            logger.info(f"🔗 Blob accessible at: {blob_url}")

        except Exception as e:
            logger.error(f"❌ SAS URL generation failed: {e}")
            logger.debug(traceback.format_exc())
            blob_url = None

        # --- Step 6: Return JSON response ---
        response_body = {
            "blob_name": blob_name,
            "blob_url": blob_url,
            "timestamp": timestamp,
            "status": "success",
        }

        return func.HttpResponse(
            body=json.dumps(response_body, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"🔥 Unexpected failure: {e}\n{tb}")
        return func.HttpResponse(f"Internal Server Error:\n{tb}", status_code=500)
