# Complete __init__.py for Azure Function App
# Includes:
# 1. Parametric category routing (folder -> endpoint)
# 2. Upload to Blob
# 3. Call Inference Container App
# 4. Log to ADLS/Synapse (Parquet)
# 5. Return JSON response to Streamlit

import logging
import json
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.filedatalake import DataLakeServiceClient
import os
import requests
from datetime import datetime
from io import BytesIO
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Default Store ID
DEFAULT_STORE_ID = "CHN01"

# Blob Storage Config
BLOB_CONN_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER = "uploads"
BLOB_BASE_URL = os.getenv("BLOB_BASE_URL")  # example: https://xxxx.blob.core.windows.net/uploads

blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)
blob_client_container = blob_service.get_container_client(BLOB_CONTAINER)

# ADLS / Synapse Config
ADLS_CONN_STRING = os.getenv("ADLS_CONNECTION_STRING")
ADLS_FS_NAME = "raw"  # e.g., raw zone
adls_service = DataLakeServiceClient.from_connection_string(ADLS_CONN_STRING)
adls_fs = adls_service.get_file_system_client(ADLS_FS_NAME)

# Inference Service URLs (Container App)
INFERENCE_ENDPOINTS = {
    "dresscode": os.getenv("INFERENCE_DRESSCODE_URL"),
    "dustbin": os.getenv("INFERENCE_DUSTBIN_URL"),
    "default": os.getenv("INFERENCE_DEFAULT_URL")
}

# ---------------------------------------------------------------------
# LOG WRITER (ADLS PARQUET)
# ---------------------------------------------------------------------

def write_log_to_synapse(store_id, category, filename, input_blob_url, inference_blob_url, inference_result):
    """
    Writes upload + inference metadata into ADLS in Parquet format
    Partitioned by year/month/day for Synapse compatibility.
    """
    try:
        now = datetime.utcnow()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")

        path = f"logs/{store_id}/{category}/year={year}/month={month}/day={day}/log.parquet"

        df = pd.DataFrame([
            {
                "timestamp": now.isoformat(),
                "store_id": store_id,
                "category": category,
                "filename": filename,
                "input_blob_url": input_blob_url,
                "inference_blob_url": inference_blob_url,
                "inference_result": inference_result
            }
        ])

        file_client = adls_fs.get_file_client(path)

        # Try reading existing file
        try:
            existing = file_client.download_file().readall()
            old_df = pd.read_parquet(BytesIO(existing))
            final_df = pd.concat([old_df, df], ignore_index=True)
        except Exception:
            final_df = df

        output = BytesIO()
        final_df.to_parquet(output, index=False)
        output.seek(0)

        file_client.upload_data(output.read(), overwrite=True)
        logging.info(f"Log written to ADLS: {path}")

    except Exception as e:
        logging.error(f"ADLS Log Write Failed: {str(e)}")


# ---------------------------------------------------------------------
# MAIN RUN METHOD
# ---------------------------------------------------------------------

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Upload_image Function Triggered")

    try:
        file = req.files.get("file")
        if not file:
            return func.HttpResponse("No file received", status_code=400)

        category = req.params.get("category", "default").lower()
        store_id = req.params.get("store_id", DEFAULT_STORE_ID)

        if category not in INFERENCE_ENDPOINTS:
            category = "default"

        filename = file.filename
        prefix = category
        blob_path = f"{prefix}/{filename}"

        # Upload original to Blob
        blob_client = blob_client_container.get_blob_client(blob_path)
        blob_client.upload_blob(file.stream.read(), overwrite=True)

        input_blob_url = f"{BLOB_BASE_URL}/{blob_path}"

        # ------------------------------------------------------------------
        # CALL INFERENCE SERVICE
        # ------------------------------------------------------------------

        inference_url = INFERENCE_ENDPOINTS[category]
        files = {"file": (filename, file.stream, "application/octet-stream")}

        infer_response = requests.post(inference_url, files=files)

        if infer_response.status_code != 200:
            return func.HttpResponse(json.dumps({
                "status": "inference_failed", "error": infer_response.text
            }), status_code=500)

        infer_json = infer_response.json()
        inference_result = infer_json.get("result")
        inference_image_bytes = infer_json.get("image_bytes")

        # Upload inference image
        inference_blob_path = f"{prefix}/inferenced_{filename}"
        inference_blob_client = blob_client_container.get_blob_client(inference_blob_path)
        inference_blob_client.upload_blob(BytesIO(bytes(inference_image_bytes)), overwrite=True)

        inference_blob_url = f"{BLOB_BASE_URL}/{inference_blob_path}"

        # ------------------------------------------------------------------
        # WRITE LOG TO ADLS
        # ------------------------------------------------------------------

        write_log_to_synapse(
            store_id,
            prefix,
            filename,
            input_blob_url,
            inference_blob_url,
            inference_result
        )

        # ------------------------------------------------------------------
        # PREPARE RESPONSE
        # ------------------------------------------------------------------

        response = {
            "status": "success",
            "uploaded_to": input_blob_url,
            "inference_result": inference_result,
            "input_image": input_blob_url,
            "inferenced_image": inference_blob_url,
            "category": prefix,
            "filename": filename
        }

        return func.HttpResponse(json.dumps(response), status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500)
