import streamlit as st
import requests
import os
from datetime import datetime
import logging

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

AZURE_FUNCTION_URL = "https://cavin-pazzo-20251015-ci.azurewebsites.net/api/Upload_image"
AZURE_FUNCTION_KEY = "0j57J9uuzUDQ8VOJK9ElE3oPff4_NPBmjEkxwIBDjdRFAzFub_o5sQ=="  # REQUIRED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Smart Detection Portal",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Smart Detection Portal")
st.markdown("Upload images for analysis and receive structured AI evaluation.")

# ---------------------------------------------------------------
# SECTION 1: DRESS CODE UPLOAD
# ---------------------------------------------------------------
st.header("👔 Dress Code Check")
dresscode_files = st.file_uploader(
    "Upload one or more images for Dress Code validation",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dresscode_uploader",
)

if dresscode_files:
    st.write("Uploaded Files:")
    for file in dresscode_files:
        st.markdown(f"- {file.name}")


# ---------------------------------------------------------------
# SECTION 2: DUSTBIN UPLOAD
# ---------------------------------------------------------------
st.header("🗑 Dustbin Condition Check")
dustbin_files = st.file_uploader(
    "Upload one or more images for Dustbin validation",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dustbin_uploader",
)

if dustbin_files:
    st.write("Uploaded Files:")
    for file in dustbin_files:
        st.markdown(f"- {file.name}")


# ---------------------------------------------------------------
# SECTION 3: LIGHTS CHECK
# ---------------------------------------------------------------
st.header("💡 Lights ON/OFF Check")
lightscheck_files = st.file_uploader(
    "Upload one or more images for Lights ON/OFF verification",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="lightscheck_uploader",
)

if lightscheck_files:
    st.write("Uploaded Files:")
    for file in lightscheck_files:
        st.markdown(f"- {file
