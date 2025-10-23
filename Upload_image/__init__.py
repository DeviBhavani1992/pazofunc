import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import json
import traceback

BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Upload_image function triggered.")

    try:
        file_bytes = req.get_body()
        if not file_bytes:
            return func.HttpResponse("❌ No file received in request body.", status_code=400)

        filename = req.headers.get("x-filename", f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{filename}"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
        logging.info(f"✅ File uploaded successfully as {blob_name}")

        # Direct URL (no SAS, works for public container)
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"

        return func.HttpResponse(
            body=json.dumps({"blob_name": blob_name, "blob_url": blob_url}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"❌ Exception occurred:\n{tb}")
        return func.HttpResponse(f"❌ Upload failed:\n{tb}", status_code=500)
