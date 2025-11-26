import os
import json
from azure.storage.filedatalake import DataLakeServiceClient

ADLS_ACCOUNT_NAME = os.getenv("ADLS_ACCOUNT_NAME")
ADLS_ACCOUNT_KEY = os.getenv("ADLS_ACCOUNT_KEY")
ADLS_FILE_SYSTEM = os.getenv("ADLS_FILE_SYSTEM", "pazo")

def get_service_client():
    return DataLakeServiceClient(
        account_url=f"https://{ADLS_ACCOUNT_NAME}.dfs.core.windows.net",
        credential=ADLS_ACCOUNT_KEY
    )

def upload_json_to_adls(data, category, timestamp, store_id):
    service_client = get_service_client()
    file_system_client = service_client.get_file_system_client(file_system=ADLS_FILE_SYSTEM)

    now = timestamp[:8]  # YYYYMMDD
    year = now[:4]
    month = now[4:6]
    day = now[6:8]

    file_path = f"pazo/{store_id}/{year}/{month}/{day}/{category}_{timestamp}.json"

    directory_path = f"pazo/{store_id}/{year}/{month}/{day}"
    try:
        dir_client = file_system_client.get_directory_client(directory_path)
        dir_client.create_directory()
    except:
        pass  # Directory may already exist

    file_client = file_system_client.get_file_client(file_path)
    file_client.create_file()
    file_client.append_data(json.dumps(data), offset=0, length=len(json.dumps(data)))
    file_client.flush_data(len(json.dumps(data)))

    return file_path

