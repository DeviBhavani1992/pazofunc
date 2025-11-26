import os
import json
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

# ===============================================
# ENV VARIABLES REQUIRED
# ===============================================
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
FILESYSTEM_NAME = os.getenv("FILESYSTEM_NAME", "pazo")   # container name

if not STORAGE_ACCOUNT_NAME:
    raise Exception("Missing env: STORAGE_ACCOUNT_NAME")

# Authenticate using Managed Identity
credential = DefaultAzureCredential()

# ADLS client
service_client = DataLakeServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net",
    credential=credential
)

filesystem_client = service_client.get_file_system_client(FILESYSTEM_NAME)


# ===============================================
# UPLOAD JSON TO ADLS
# ===============================================
def upload_json_to_adls(store_id: str, category: str, payload: dict) -> str:

    now = datetime.utcnow()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    filename = f"{category}_{timestamp}.json"

    # FINAL PATH (as required)
    adls_path = f"pazo/{store_id}/{year}/{month}/{day}/{filename}"

    # Create directory client
    directory = filesystem_client.get_directory_client(
        f"pazo/{store_id}/{year}/{month}/{day}"
    )
    directory.create_directory(exist_ok=True)

    # Create or get file inside directory
    file_client = directory.get_file_client(filename)

    # Convert payload → JSON string
    data = json.dumps(payload, indent=2)

    file_client.upload_data(data, overwrite=True)

    return adls_path
