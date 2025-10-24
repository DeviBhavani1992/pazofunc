import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from io import BytesIO
import json
import traceback

# -----------------------------
# Environment Variables
# -----------------------------
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

if not BLOB_CONNECTION_STRING or not BLOB_CONTAINER_NAME:
    raise ValueError("❌ Missing BLOB_CONNECTION_STRING or BLOB_CONTAINER_NAME environment variables.")

# -----------------------------
# Initialize Clients
# -----------------------------
try:
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    logging.info(f"✅ Connected to Blob Storage: {blob_service_client.account_name}")
except Exception as e:
    logging.error(f"❌ Failed to connect to Azure Blob Storage: {e}")
    raise

# -----------------------------
# Azure Function Entry Point
# -----------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 Upload_image function triggered.")

    try:
        # -----------------------------
        # Read File Bytes
        # -----------------------------
        file_bytes = req.get_body()
        if not file_bytes:
            logging.warning("⚠️ No file data found in request body.")
            return func.HttpResponse(
                json.dumps({"error": "No file received in request body."}),
                status_code=400,
                mimetype="application/json"
            )

        file_stream = BytesIO(file_bytes)

        # -----------------------------
        # Prepare Blob Info
        # -----------------------------
        filename = req.headers.get("x-filename", f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{filename}"

        blob_client = container_client.get_blob_client(blob_name)

        # -----------------------------
        # Upload to Blob Storage
        # -----------------------------
        blob_client.upload_blob(file_stream, overwrite=True)
        logging.info(f"✅ File uploaded successfully: {blob_name}")

        # -----------------------------
        # Generate Accessible URL
        # -----------------------------
        account_name = blob_service_client.account_name
        blob_url = f"https://{account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"

        # Attempt to generate SAS token (only works if account key available)
        try:
            account_key = getattr(blob_service_client.credential, "account_key", None)
            if account_key:
                sas_token = generate_blob_sas(
                    account_name=account_name,
                    container_name=BLOB_CONTAINER_NAME,
                    blob_name=blob_name,
                    account_key=account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1)
                )
                blob_url = f"{blob_url}?{sas_token}"
                logging.info("🔑 SAS token generated successfully.")
            else:
                logging.warning("⚠️ No account key found — SAS token not generated. Blob may be private.")
        except Exception as sas_err:
            logging.error(f"❌ Failed to generate SAS token: {sas_err}")

        # -----------------------------
        # Return JSON Response
        # -----------------------------
        return func.HttpResponse(
            body=json.dumps({"blob_name": blob_name, "blob_url": blob_url}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"❌ Exception occurred:\n{tb}")
        return func.HttpResponse(
            body=json.dumps({"error": "Upload failed", "details": tb}),
            status_code=500,
            mimetype="application/json"
        )
