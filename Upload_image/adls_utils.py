import os
import json
import logging
from azure.storage.filedatalake import DataLakeServiceClient

ADLS_CONNECTION_STRING = os.getenv("ADLS_CONNECTION_STRING")
ADLS_CONTAINER_NAME = os.getenv("ADLS_CONTAINER_NAME", "pazo")

def get_service_client():
    if not ADLS_CONNECTION_STRING:
        raise ValueError("ADLS_CONNECTION_STRING must be set")
    
    return DataLakeServiceClient.from_connection_string(ADLS_CONNECTION_STRING)

def upload_json_to_adls(data, category, timestamp, store_id):
    try:
        logging.info(f"Starting ADLS upload for {category}_{timestamp}")
        
        service_client = get_service_client()
        file_system_client = service_client.get_file_system_client(file_system=ADLS_CONTAINER_NAME)

        now = timestamp[:8]  # YYYYMMDD
        year = now[:4]
        month = now[4:6]
        day = now[6:8]

        file_path = f"pazo/{store_id}/{year}/{month}/{day}/{category}_{timestamp}.json"
        directory_path = f"pazo/{store_id}/{year}/{month}/{day}"
        
        logging.info(f"Creating directory: {directory_path}")
        
        # Create directory if it doesn't exist
        try:
            dir_client = file_system_client.get_directory_client(directory_path)
            dir_client.create_directory()
            logging.info(f"Directory created: {directory_path}")
        except Exception as dir_error:
            logging.info(f"Directory creation skipped (may already exist): {str(dir_error)}")

        # Upload file
        json_data = json.dumps(data, indent=2)
        json_bytes = json_data.encode('utf-8')
        
        logging.info(f"Uploading file: {file_path}")
        
        file_client = file_system_client.get_file_client(file_path)
        file_client.create_file()
        file_client.append_data(json_bytes, offset=0, length=len(json_bytes))
        file_client.flush_data(len(json_bytes))
        
        logging.info(f"Successfully uploaded to ADLS: {file_path}")
        return file_path
        
    except Exception as e:
        logging.error(f"ADLS upload failed: {str(e)}")
        raise Exception(f"Failed to upload to ADLS: {str(e)}")

