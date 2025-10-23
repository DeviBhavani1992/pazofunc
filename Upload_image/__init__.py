import logging
import json
import os
import base64 # NEW: For Base64 decoding
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
# from io import BytesIO # No longer needed if we correctly handle raw bytes vs. Base64
import traceback
import magic 

# --- Read environment variables ---
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# --- Initialize Blob client (Error handling refined) ---
if not BLOB_CONNECTION_STRING:
    raise ValueError("BLOB_CONNECTION_STRING environment variable is not set.")
try:
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
except Exception as e:
    logging.error(f"Error initializing BlobServiceClient: {e}")
    raise

# Define accepted image types
ACCEPTED_IMAGE_TYPES = {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png']
}

def get_file_type_info(file_bytes):
    """Uses 'magic' to determine file type."""
    try:
        mime = magic.from_buffer(file_bytes, mime=True)
        if mime in ACCEPTED_IMAGE_TYPES:
            extension = ACCEPTED_IMAGE_TYPES[mime][0] 
            return mime, extension
        return None, None
    except Exception as e:
        logging.warning(f"Error determining file type with magic: {e}")
        return None, None

# This remains the entry point for your Azure Function
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="Upload_image")
def Upload_image(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 Upload_image function triggered.")

    try:
        # --- 1. DETERMINE AND DECODE FILE BYTES ---
        content_type = req.headers.get('Content-Type', '').lower()
        file_bytes = None
        filename = req.headers.get("x-filename", "image")

        if 'application/json' in content_type:
            # Assume client sent a JSON payload with a Base64 string
            try:
                req_json = req.get_json()
                # Common practice is to name the field 'file_data' or 'base64_data'
                base64_data = req_json.get('file_data') 
                if not base64_data:
                    return func.HttpResponse("❌ JSON body missing 'file_data' field.", status_code=400)
                
                # Decode the Base64 string into raw bytes
                file_bytes = base64.b64decode(base64_data)
                logging.info("Detected JSON/Base64 payload and decoded successfully.")
            except ValueError:
                return func.HttpResponse("❌ Invalid JSON or Base64 data.", status_code=400)
            except Exception as e:
                return func.HttpResponse(f"❌ Failed to process JSON payload: {e}", status_code=400)
                
        else:
            # Assume client sent raw binary data (e.g., via multipart/form-data or raw body)
            file_bytes = req.get_body()
            logging.info("Detected raw binary payload.")

        if not file_bytes:
            return func.HttpResponse("❌ No file data received in request body.", status_code=400)

        # --- 2. VALIDATE FILE FORMAT ---
        mime_type, file_ext = get_file_type_info(file_bytes)
        
        if not mime_type:
            # The file bytes do not correspond to a valid PNG or JPEG format
            return func.HttpResponse(
                "❌ Invalid file type. Only JPEG and PNG are accepted or file data is corrupted.", 
                status_code=415 
            )

        # --- 3. PREPARE BLOB METADATA ---
        base_name = os.path.splitext(filename)[0]
        final_filename = f"{base_name}{file_ext}" # Use validated extension

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{final_filename}"

        # --- 4. UPLOAD TO AZURE BLOB STORAGE ---
        blob_client = container_client.get_blob_client(blob_name)
        
        blob_client.upload_blob(
            file_bytes,                                   
            overwrite=True, 
            # CRITICAL: Set the correct MIME type for the downloaded file to work
            content_settings={'content_type': mime_type} 
        )
        logging.info(f"✅ File uploaded as {blob_name} with MIME type {mime_type}")

        # --- 5. GENERATE SAS URL AND RETURN RESPONSE ---
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
        return func.HttpResponse(f"❌ Internal Server Error. Details: {e}", status_code=500)