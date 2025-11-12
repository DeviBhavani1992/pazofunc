import streamlit as st
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_FUNCTION_URL = os.getenv("AZURE_FUNCTION_URL")

if not AZURE_FUNCTION_URL:
    st.error("❌ AZURE_FUNCTION_URL environment variable is not set.")
    st.stop()

st.title("📸 Upload Images to Azure Blob Storage with AI Analysis")

option = st.radio("Choose input method:", ["Upload Images", "Capture from Camera"])

uploaded_images = []
if option == "Upload Images":
    uploaded_images = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
else:
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        uploaded_images = [camera_image]

if uploaded_images:
    cols = st.columns(len(uploaded_images))
    for i, img in enumerate(uploaded_images):
        with cols[i]:
            st.image(img, caption=f"Image {i+1}", width=300)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Upload Only 🚀"):
            with st.spinner("Uploading..."):
                try:
                    files = [('files', (img.name, img.getvalue(), img.type)) for img in uploaded_images]
                    response = requests.post(AZURE_FUNCTION_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {len(uploaded_images)} images uploaded successfully")
                        for url in result['uploaded']:
                            st.write(f"🌐 {url}")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

    with col2:
        if st.button("Upload + Dress Code 👔"):
            with st.spinner("Uploading and analyzing dress code..."):
                try:
                    files = [('files', (img.name, img.getvalue(), img.type)) for img in uploaded_images]
                    response = requests.post(f"{AZURE_FUNCTION_URL}?action=dresscode", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Analysis completed")
                        for res in result['results']:
                            status_icon = "✅" if res['status'] == "compliant" else "❌"
                            st.write(f"{status_icon} Image {res['image']}: {res['message']}")
                            st.write(f"   Shirt: {res['detections']['shirt']}, Pant: {res['detections']['pant']}, Shoe: {res['detections']['shoe']}")
                            st.write(f"   🌐 {res['blob_url']}")
                    else:
                        st.error(f"❌ Analysis failed: {response.text}")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

    with col3:
        if st.button("Upload + Dustbin 🗑️"):
            with st.spinner("Uploading and detecting dustbins..."):
                try:
                    files = [('files', (img.name, img.getvalue(), img.type)) for img in uploaded_images]
                    response = requests.post(f"{AZURE_FUNCTION_URL}?action=dustbin", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Detection completed")
                        for res in result['results']:
                            status_icon = "✅" if res['status'] == "dustbin_found" else "❌"
                            st.write(f"{status_icon} Image {res['image']}: {res['message']}")
                            if res['detections']:
                                for det in res['detections']:
                                    st.write(f"   - {det['label']}: {det['confidence']:.2f}")
                            st.write(f"   🌐 {res['blob_url']}")
                    else:
                        st.error(f"❌ Detection failed: {response.text}")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

elif not uploaded_images:
    st.info("Please upload or capture images to proceed.")