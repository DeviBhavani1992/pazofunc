import streamlit as st
import requests
import os
import logging
import pdb
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read Azure Function URL from environment variable
AZURE_FUNCTION_URL = os.getenv("AZURE_FUNCTION_URL")

if not AZURE_FUNCTION_URL:
    st.error("❌ AZURE_FUNCTION_URL environment variable is not set.")
    st.stop()

st.title("📸 Capture or Upload Image to Azure Blob Storage")

# Let user capture or upload an image
uploaded_image = st.camera_input("Take a photo") or st.file_uploader("Or choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Selected image", width='stretch')

    if st.button("Upload to Azure Blob Storage 🚀"):
        # Prepare the file
        files = {'file': (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        headers = {"x-filename": uploaded_image.name}

        with st.spinner("Uploading to Azure..."):
            try:
                pdb.set_trace() 
                logger.info(f"Uploading {uploaded_image.name} to Azure Function...")
                response = requests.post(AZURE_FUNCTION_URL, files=files, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Uploaded successfully as {result['blob_name']}")
                    st.write(f"🌐 Access URL (1-hour SAS link): {result['blob_url']}")
                    logger.info(f"Upload successful: {result}")
                else:
                    st.error(f"❌ Upload failed: {response.text}")
                    logger.error(f"Upload failed with status {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                logger.exception("Exception occurred during upload")

