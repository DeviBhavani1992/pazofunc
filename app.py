import streamlit as st
import requests
from datetime import datetime
import logging

# Azure Function URL
AZURE_FUNCTION_URL = (
    "https://cavin-pazzo-20251015-ci.azurewebsites.net/api/Upload_image"
    "?code=F5MbFDI6XcXgRrbm7wX3JcyZdPzsOjswD2KCQROj9haWAzFuiNw41g=="
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Pazo AI Portal", page_icon="✨")

st.title("✨  Image Analysis Dashboard")
st.markdown("Upload images for all AI Evaluation Categories below.")

# Categories including new restroomcheck
categories = {
    "dresscode": "👔 Dress Code",
    "lightscheck": "💡 Lights Check",
    "floorcheck": "🧹 Floor Check",
    "nailpolishtray": "💅 Nail Polish Tray Check",
    "shampoobottles": "🧴 Shampoo Bottles Check",
    "restroomcheck": "🚽 Rest Room  Check",
    "dustbin": "🗑️ Dustbin Check"
}

uploaded_files = {}

st.header("📸 Upload Images")

# Auto-generate uploaders
for key, label in categories.items():
    with st.expander(label):
        uploaded_files[key] = st.file_uploader(
            f"Upload {label} Images",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
            key=key
        )

if st.button("🚀 Submit All for AI Analysis"):
    results = []
    total_files = sum(len(files) for files in uploaded_files.values() if files)

    if total_files == 0:
        st.error("Please upload at least one image before submitting.")
        st.stop()

    st.info(f"Processing {total_files} images... Please wait ⏳")

    for category, files in uploaded_files.items():
        if not files:
            continue

        for file in files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{category}_{timestamp}_{file.name}"

            files_payload = {
                "file": (fname, file.getvalue(), file.type)
            }

            endpoint = f"{AZURE_FUNCTION_URL}&category={category}"

            try:
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

            except Exception as e:
                results.append({
                    "filename": fname,
                    "category": category,
                    "status": "error",
                    "message": str(e)
                })

    st.success("🎉 Analysis Completed!")

    st.header("📊 AI Results")
    for r in results:
        st.json(r)
