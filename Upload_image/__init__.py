import logging
import json
import os
import base64 
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import traceback
import magic 
import azure.functions as func

# --- Configuration and Initialization (Unchanged) ---
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

if not BLOB_CONNECTION_STRING:
    raise ValueError("BLOB_CONNECTION_STRING environment variable is not set.")
try:
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
except Exception as e:
    logging.error(f"Error initializing BlobServiceClient: {e}")
    raise

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
        # Log a warning but don't fail, as this might be expected if the data is corrupted/Base64
        logging.warning(f"File type determination failed: {e}")
        return None, None

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="Upload_image")
def Upload_image(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 Upload_image function triggered.")

    try:
        # --- 1. GET RAW BODY ---
        original_body = req.get_body()
        file_bytes = original_body # Start with raw body assumption
        filename = req.headers.get("x-filename", "image")

        if not original_body:
            return func.HttpResponse("❌ No file data received in request body.", status_code=400)

        # --- 2. INITIAL VALIDATION (Assumes Raw/Binary Data) ---
        mime_type, file_ext = get_file_type_info(file_bytes)

        if not mime_type:
            # File validation failed. Check if it's Base64 string that needs decoding.
            logging.warning("Initial validation failed. Attempting Base64 decode...")
            
            try:
                # Attempt to get the body content as a string (it might be raw binary data, 
                # but we try to decode it as a Base64 string)
                
                # Check 1: If it was sent as JSON, pull the data field
                if 'application/json' in req.headers.get('Content-Type', '').lower():
                     req_json = req.get_json()
                     base64_string = req_json.get('file_data') # Assuming key 'file_data'
                else:
                    # Check 2: If it was sent as raw body, try to decode the bytes as a string
                    base64_string = original_body.decode('utf-8')
                
                if base64_string:
                    file_bytes = base64.b64decode(base64_string)
                    mime_type, file_ext = get_file_type_info(file_bytes)
                    
                    if mime_type:
                        logging.info("Successfully decoded Base64 string and validated image.")
                    else:
                        # Failed after Base64 decode
                        logging.error("Base64 decode successful, but resulting bytes are still invalid image format.")
                        return func.HttpResponse(
                            "❌ File data is corrupted or not a valid Base64 encoded image.", 
                            status_code=415 
                        )
                else:
                    # No string data to decode
                    raise ValueError("No string data found to attempt Base64 decoding.")

            except Exception as decode_e:
                logging.error(f"Failed all decoding attempts: {decode_e}")
                # If all attempts fail, return the final error
                return func.HttpResponse(
                    "❌ Unrecognized image file format. Ensure you send valid JPEG/PNG raw bytes OR Base64 string.", 
                    status_code=415 
                )


        # --- 3. PREPARE BLOB METADATA (Flow continues here only if mime_type is successfully set) ---
        base_name = os.path.splitext(filename)[0]
        final_filename = f"{base_name}{file_ext}" 

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"{timestamp}_{final_filename}"

        # --- 4. UPLOAD TO AZURE BLOB STORAGE ---
        blob_client = container_client.get_blob_client(blob_name)
        
        blob_client.upload_blob(
            file_bytes, # <-- This now holds the validated, correct binary data
            overwrite=True, 
            content_settings={'content_type': mime_type} 
        )
        logging.info(f"✅ File uploaded as {blob_name} with MIME type {mime_type}")

        # --- 5. GENERATE SAS URL AND RETURN RESPONSE (Unchanged) ---
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