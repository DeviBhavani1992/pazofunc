import streamlit as st
import requests
from datetime import datetime
import logging

AZURE_FUNCTION_URL = "https://cavin-pazzo-20251015-ci.azurewebsites.net/api/Upload_image"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Pazo AI Portal", page_icon="✨")

st.title("✨ AI Image Analysis Portal")
st.markdown("Upload images for Dress Code / Dustbin / Lights Check AI evaluation.")

# -----------------------------
# SECTION: DRESSCODE
# -----------------------------
st.header("👔 Dress Code")
dress_files = st.file_uploader(
    "Upload Dress Code Images",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key="dress"
)

# -----------------------------
# SECTION: DUSTBIN
# -----------------------------
st.header("🗑️ Dustbin Check")
dustbin_files = st.file_uploader(
    "Upload Dustbin Images",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key="dustbin"
)

# -----------------------------
# SECTION: LIGHT CHECK
# -----------------------------
st.header("💡 Light Check")
light_files = st.file_uploader(
    "Upload Lights Check Images",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key="lights"
)

# -----------------------------
# SUBMIT ALL
# -----------------------------
if st.button("🚀 Submit for Analysis"):

    all_files = [
        ("dresscode", dress_files),
        ("dustbin", dustbin_files),
        ("lightscheck", light_files)
    ]

    results = []

    for category, files in all_files:
        if not files:
            continue

        for file in files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{category}_{timestamp}_{file.name}"

            files_payload = {
                "file": (fname, file.getvalue(), file.type)
            }

            endpoint = f"{AZURE_FUNCTION_URL}?category={category}"

            response = requests.post(endpoint, files=files_payload)

            if response.status_code == 200:
                results.append(response.json())
            else:
                results.append({
                    "filename": fname,
                    "category": category,
                    "status": "error",
                    "message": response.text
                })

    st.success("Completed!")

    st.subheader("📊 Results")
    for r in results:
        st.json(r)
