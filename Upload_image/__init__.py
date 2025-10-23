import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import json
import traceback
from io import BytesIO

# Read connection string and container name from environment variables
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Upload_image function triggered.")

    try:
        # Read raw bytes from request body
        file_bytes = req.get_body()
        if not file_bytes:
            return func.HttpResponse("❌ No file received in request body.", status_code=400)

        # Get filename from header or generate default
        filename = req.headers.get(
            "x-filename",
            f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        )

        # Create unique blob name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{filename}"

        # Upload the blob
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
        logging.info(f"✅ File uploaded successfully as {blob_name}")

        # Generate a temporary SAS URL valid for 1 hour
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=BLOB_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        blob_url = f"{blob_client.url}?{sas_token}"

        # Return JSON response
        return func.HttpResponse(
            body=json.dumps({
                "blob_name": blob_name,
                "blob_url": blob_url
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"❌ Exception occurred:\n{tb}")
        return func.HttpResponse(f"❌ Upload failed:\n{tb}", status_code=500)
