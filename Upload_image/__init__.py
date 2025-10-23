import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from io import BytesIO
import json
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("🚀 Starting 'Upload_image' Azure Function with full diagnostics...")

    debug_info = {}  # collect runtime diagnostics

    try:
        # --- ENVIRONMENT VARIABLES ---
        blob_conn_str = os.getenv("BLOB_CONNECTION_STRING")
        container_name = os.getenv("BLOB_CONTAINER_NAME")
        debug_info["BLOB_CONNECTION_STRING_present"] = bool(blob_conn_str)
        debug_info["BLOB_CONTAINER_NAME"] = container_name

        if not blob_conn_str or not container_name:
            msg = f"❌ Missing required env vars: BLOB_CONNECTION_STRING or BLOB_CONTAINER_NAME\n\nDEBUG={json.dumps(debug_info, indent=2)}"
            logger.error(msg)
            return func.HttpResponse(msg, status_code=500)

        # --- INIT CLIENTS ---
        try:
            blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
            container_client = blob_service_client.get_container_client(container_name)
            debug_info["account_name"] = blob_service_client.account_name
            logger.info(f"✅ Connected to blob account: {blob_service_client.account_name}")
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"❌ BlobServiceClient init failed: {e}\n\n{tb}\n\nDEBUG={json.dumps(debug_info, indent=2)}"
            logger.error(msg)
            return func.HttpResponse(msg, status_code=500)

        # --- READ FILE ---
        try:
            file_bytes = req.get_body()
            debug_info["file_size"] = len(file_bytes)
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"❌ Failed to read request body: {e}\n{tb}\n\nDEBUG={json.dumps(debug_info, indent=2)}"
            logger.error(msg)
            return func.HttpResponse(msg, status_code=400)

        if not file_bytes:
            msg = f"⚠️ No file received.\nDEBUG={json.dumps(debug_info, indent=2)}"
            logger.warning(msg)
            return func.HttpResponse(msg, status_code=400)

        # --- PREPARE UPLOAD ---
        filename = req.headers.get("x-filename", f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{filename}"
        debug_info["blob_name"] = blob_name

        # --- UPLOAD FILE ---
        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(BytesIO(file_bytes), overwrite=True)
            logger.info(f"✅ Uploaded: {blob_name}")
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"❌ Upload failed: {e}\n\n{tb}\n\nDEBUG={json.dumps(debug_info, indent=2)}"
            logger.error(msg)
            return func.HttpResponse(msg, status_code=500)

        # --- GENERATE SAS URL ---
        try:
            account_key = None
            if hasattr(blob_service_client.credential, "account_key"):
                account_key = blob_service_client.credential.account_key
            debug_info["account_key_present"] = bool(account_key)

            if account_key:
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=container_name,
                    blob_name=blob_name,
                    account_key=account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1)
                )
                blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
            else:
                blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"

            logger.info(f"🔗 Blob URL: {blob_url}")
            debug_info["blob_url"] = blob_url
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"❌ SAS generation failed: {e}\n\n{tb}\n\nDEBUG={json.dumps(debug_info, indent=2)}"
            logger.error(msg)
            return func.HttpResponse(msg, status_code=500)

        # --- RETURN RESPONSE ---
        result = {
            "status": "success",
            "blob_name": blob_name,
            "blob_url": blob_url,
            "debug_info": debug_info
        }

        return func.HttpResponse(json.dumps(result, indent=2), mimetype="application/json", status_code=200)

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"🔥 Fatal error in function: {e}\n\n{tb}\n\nDEBUG={json.dumps(debug_info, indent=2)}"
        logger.error(msg)
        return func.HttpResponse(msg, status_code=500)
