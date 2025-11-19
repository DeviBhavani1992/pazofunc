import streamlit as st
import requests
from datetime import datetime
import logging

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
AZURE_FUNCTION_URL = "https://cavin-pazzo-20251015-ci.azurewebsites.net/api/Upload_image"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Image Evaluation Portal", page_icon="🖼️")
st.title("📸 Image Evaluation Portal")
st.markdown("Upload images for **Dress Code**, **Dustbin**, and **Lights Check** evaluation.")

# ---------------------------------------------------------------
# SECTION: DRESSCODE
# ---------------------------------------------------------------
st.header("👔 Dress Code Validation")
dresscode_files = st.file_uploader(
    "Upload images for Dress Code",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dresscode_uploader",
)

# ---------------------------------------------------------------
# SECTION: DUSTBIN
# ---------------------------------------------------------------
st.header("🗑️ Dustbin Evaluation")
dustbin_files = st.file_uploader(
    "Upload images for Dustbin Check",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dustbin_uploader",
)

# ---------------------------------------------------------------
# SECTION: LIGHTS CHECK
# ---------------------------------------------------------------
st.header("💡 Lights Check")
lights_files = st.file_uploader(
    "Upload images for Lights ON/OFF Check",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="lights_uploader",
)

# ---------------------------------------------------------------
# SUBMIT BUTTON
# ---------------------------------------------------------------
if st.button("🚀 Submit"):
    if not dresscode_files and not dustbin_files and not lights_files:
        st.warning("Please upload at least one image.")
    else:
        st.info("Processing images... Please wait.")
        results = []

        def upload_and_infer(file, category):
            try:
                filename = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}"

                files_payload = {
                    "file": (filename, file.getvalue(), file.type or "image/jpeg")
                }

                resp = requests.post(
                    f"{AZURE_FUNCTION_URL}?category={category}",
                    files=files_payload,
                    timeout=180
                )

                return resp.json() if resp.status_code == 200 else {
                    "filename": filename,
                    "category": category,
                    "status": "error",
                    "message": resp.text
                }

            except Exception as e:
                return {
                    "filename": file.name,
                    "category": category,
                    "status": "error",
                    "message": str(e),
                }

        # PROCESS IMAGES FOR EACH CATEGORY
        for f in dresscode_files or []:
            results.append(upload_and_infer(f, "dresscode"))

        for f in dustbin_files or []:
            results.append(upload_and_infer(f, "dustbin"))

        for f in lights_files or []:
            results.append(upload_and_infer(f, "lightscheck"))

        st.success("✅ Completed!")

        for r in results:
            st.json(r)

st.markdown("---")
st.caption("Powered by Azure Functions + Gemma + ADLS")
