import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from io import BytesIO
import json
import traceback

# --- Read environment variables ---
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# --- Initialize Blob client ---
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 Upload_image function triggered.")

    try:
        # --- Get file bytes from request ---
        file_bytes = req.get_body()
        if not file_bytes:
            return func.HttpResponse("❌ No file received in request body.", status_code=400)

        # --- Extract filename from headers ---
        filename = req.headers.get("x-filename", f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{filename}"

        # --- Upload to Azure Blob Storage (using BytesIO to avoid corruption) ---
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(BytesIO(file_bytes), overwrite=True)
        logging.info(f"✅ File uploaded successfully as {blob_name}")

        # --- Generate temporary SAS URL (valid for 1 hour) ---
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=BLOB_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1),
        )

        blob_url = (
            f"https://{blob_service_client.account_name}.blob.core.windows.net/"
            f"{BLOB_CONTAINER_NAME}/{blob_name}?{sas_token}"
        )

        # --- Return success response ---
        response_body = {
            "blob_name": blob_name,
            "blob_url": blob_url
        }

        logging.info(f"✅ Upload complete. SAS URL generated for {blob_name}")
        return func.HttpResponse(
            body=json.dumps(response_body),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"❌ Exception occurred:\n{tb}")
        return func.HttpResponse(f"❌ Upload failed:\n{tb}", status_code=500)
