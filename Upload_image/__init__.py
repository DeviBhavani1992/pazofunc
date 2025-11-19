import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
import os
import json
import traceback
import requests
from datetime import datetime

# -------------------------------------------------------------
# ENVIRONMENT SETTINGS
# -------------------------------------------------------------
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "uploads")

GEMMA_ENDPOINT = os.getenv("GEMMA_ENDPOINT")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

# -------------------------------------------------------------
# DEFAULT PROMPTS
# -------------------------------------------------------------
DEFAULT_PROMPTS = {
    "dresscode": """
From this image validate the dress code:
- Shirt : Black or White
- Pant : Black
- Shoe : Must be present (color optional)
- Beards : Not allowed
If criteria fail → say "dress code inappropriate".
Finally give a score (0–100).
""",

    "dustbin": """
Check the image for:
- Is dustbin present?
- Is it clean or not?
- Does it have poly cover inside?
- Is garbage not overflowing?
Give final score (0–100).
""",

    "lightscheck": """
Check whether all the lights in the image are ON.
If ON → give high score.
If OFF → give low score.
Provide score (0–100).
"""
}

# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info("Processing request...")

        # -------------------------------------------------------------
        # READ PARAMETERS
        # -------------------------------------------------------------
        category = req.params.get("category")
        if not category:
            return func.HttpResponse("Missing category", status_code=400)

        if category not in DEFAULT_PROMPTS:
            return func.HttpResponse("Invalid category provided", status_code=400)

        # -------------------------------------------------------------
        # READ IMAGE FILE
        # -------------------------------------------------------------
        file = req.files.get("file")
        if not file:
            return func.HttpResponse("Image file missing", status_code=400)

        filename = file.filename
        blob_path = f"{category}/{filename}"

        # -------------------------------------------------------------
        # UPLOAD IMAGE TO BLOB
        # -------------------------------------------------------------
        blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
        container_client = blob_service.get_container_client(BLOB_CONTAINER_NAME)

        container_client.upload_blob(
            name=blob_path,
            data=file.stream.read(),
            overwrite=True,
            content_settings=ContentSettings(content_type=file.content_type)
        )

        blob_url = container_client.get_blob_client(blob_path).url

        # -------------------------------------------------------------
        # LLM – SEND IMAGE + PROMPT
        # -------------------------------------------------------------
        prompt = DEFAULT_PROMPTS[category]

        payload = {
            "prompt": prompt,
            "image_url": blob_url
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMMA_API_KEY}"
        }

        response = requests.post(GEMMA_ENDPOINT, json=payload, headers=headers)
        llm_output = response.json()

        # -------------------------------------------------------------
        # SAVE RESULT JSON BACK TO ADLS
        # -------------------------------------------------------------
        result_filename = f"{category}/{filename.split('.')[0]}_result.json"

        container_client.upload_blob(
            name=result_filename,
            data=json.dumps(llm_output, indent=4),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
        )

        # -------------------------------------------------------------
        # RETURN RESPONSE
        # -------------------------------------------------------------
        return func.HttpResponse(
            json.dumps({
                "message": "Success",
                "category": category,
                "image_blob_url": blob_url,
                "result_blob": result_filename,
                "llm_output": llm_output
            }, indent=4),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
