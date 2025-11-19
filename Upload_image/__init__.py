import logging
import azure.functions as func
import os
import json
import requests
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.filedatalake import DataLakeServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# ENVIRONMENT VARIABLES
# -----------------------------
BLOB_CONN_STR = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME")

ADLS_CONN_STR = os.getenv("ADLS_CONNECTION_STRING")
ADLS_ACCOUNT_URL = os.getenv("ADLS_ACCOUNT_URL")
ADLS_CONTAINER = os.getenv("ADLS_CONTAINER_NAME")

GEMMA_ENDPOINT = os.getenv("GEMMA_ENDPOINT")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

# -----------------------------
# DEFAULT PROMPTS
# -----------------------------
DEFAULT_PROMPTS = {
    "dresscode": """
From this image validate the dress code:
- Shirt: black or white
- Pant: black
- Shoe: must be present (color optional)
- Beards: No

If any rule fails, say dress code inappropriate.
Give rating score based on rules.
""",
    "dustbin": """
Check image for dustbin:
- Is dustbin present?
- Is it clean or not clean?
- Is poly cover available inside?
- Is it not overfilled?

Give final rating score.
""",
    "lightscheck": """
Check if all indoor lights are ON.
Return:
- How many lights detected
- How many ON
- Final score
"""
}

# ------------------------------------------------------------------------------------
# WRITE RESULTS JSON TO ADLS GEN2
# ------------------------------------------------------------------------------------
def save_json_to_adls(filename, result_json):
    try:
        adls_client = DataLakeServiceClient.from_connection_string(ADLS_CONN_STR)
        filesystem = adls_client.get_file_system_client(ADLS_CONTAINER)

        path = f"results/{filename.replace('.jpg', '.json').replace('.png', '.json')}"
        file_client = filesystem.get_file_client(path)

        json_bytes = json.dumps(result_json, indent=2).encode("utf-8")
        file_client.upload_data(json_bytes, overwrite=True)

        return True
    except Exception as e:
        logger.exception("Failed to save JSON to ADLS")
        return False


# ------------------------------------------------------------------------------------
# GEMMA / OLLAMA CALL
# ------------------------------------------------------------------------------------
def run_gemma_inference(image_url, prompt):
    try:
        payload = {
            "model": "gemma2:27b",
            "prompt": f"{prompt}\n\nImage URL: {image_url}",
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        resp = requests.post(f"{GEMMA_ENDPOINT}/api/generate",
                             headers=headers,
                             data=json.dumps(payload),
                             timeout=240)

        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": resp.text}

    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------------------------
# MAIN FUNCTION APP ENTRYPOINT
# ------------------------------------------------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("⚡ Upload_image function triggered")

    category = req.params.get("category")
    if not category:
        return func.HttpResponse("Missing category", status_code=400)

    if category not in DEFAULT_PROMPTS:
        return func.HttpResponse("Invalid category", status_code=400)

    # -----------------------------
    # Read image file
    # -----------------------------
    file = req.files.get("file")
    if not file:
        return func.HttpResponse("File not found", status_code=400)

    filename = file.filename
    content = file.stream.read()

    # -----------------------------
    # Upload original image → Blob
    # -----------------------------
    try:
        blob_client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container_client = blob_client.get_container_client(BLOB_CONTAINER)

        blob_path = f"{category}/{filename}"
        container_client.upload_blob(
            blob_path,
            content,
            overwrite=True,
            content_settings=ContentSettings(content_type=file.mimetype)
        )

        blob_url = f"https://{blob_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER}/{blob_path}"

    except Exception as e:
        logger.exception("Blob upload error")
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500)

    # -----------------------------
    # Run GEMMA inference
    # -----------------------------
    result = run_gemma_inference(blob_url, DEFAULT_PROMPTS[category])

    # -----------------------------
    # Save results to ADLS
    # -----------------------------
    json_filename = filename.replace(".jpg", ".json").replace(".png", ".json")

    save_json_to_adls(json_filename, result)

    # -----------------------------
    # Response
    # -----------------------------
    return func.HttpResponse(
        json.dumps({
            "filename": filename,
            "category": category,
            "blob_url": blob_url,
            "result": result
        }),
        mimetype="application/json"
    )
