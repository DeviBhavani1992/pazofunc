import streamlit as st
import requests
import os
import logging
import json

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

    if st.button("Upload Images 🚀"):
        with st.spinner("Uploading to Azure..."):
            try:
                uploaded_urls = []

                for i, img in enumerate(uploaded_images):
                    files = {'file': (img.name, img.getvalue(), img.type)}
                    response = requests.post(AZURE_FUNCTION_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()

                        # Your Azure Function returns this format:
                        # { "status": "success", "uploaded": ["https://..."], "count": N }
                        if "uploaded" in result:
                            uploaded_urls.extend(result["uploaded"])
                        elif "blob_url" in result:
                            uploaded_urls.append(result["blob_url"])
                        else:
                            st.warning(f"⚠️ Unexpected response for image {i+1}: {result}")
                    else:
                        st.error(f"❌ Upload failed for image {i+1}: {response.text}")
                        st.stop()

                if uploaded_urls:
                    st.success(f"✅ {len(uploaded_urls)} images uploaded successfully!")

                    for i, url in enumerate(uploaded_urls, 1):
                        st.write(f"🌐 **Image {i} Blob URL:** {url}")

                    # Save for analysis phase
                    st.session_state.uploaded_urls = uploaded_urls
                else:
                    st.error("❌ No uploaded URLs received from Function App.")

            except Exception as e:
                st.error(f"⚠️ Upload error: {str(e)}")

    # Show analysis section only if upload worked
    if "uploaded_urls" in st.session_state and st.session_state.uploaded_urls:
        st.subheader("🔍 Analyze Uploaded Images")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Dress Code 👔"):
                with st.spinner("Analyzing dress code..."):
                    try:
                        for i, url in enumerate(st.session_state.uploaded_urls):
                            dresscode_url = AZURE_FUNCTION_URL.replace('/upload_image', '/dresscode_analysis')
                            response = requests.post(dresscode_url, json={'blob_url': url})
                            
                            if response.status_code == 200:
                                result = response.json()
                                status_icon = "✅" if result['status'] == "compliant" else "❌"
                                st.write(f"{status_icon} Image {i+1}: {result['message']}")
                                st.write(f"   Shirt: {result['detections']['shirt']}, Pant: {result['detections']['pant']}, Shoe: {result['detections']['shoe']}")
                            else:
                                st.error(f"❌ Analysis failed for image {i+1}: {response.text}")
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")

        with col2:
            if st.button("Check Dustbin Detection 🗑️"):
                with st.spinner("Detecting dustbins..."):
                    try:
                        for i, url in enumerate(st.session_state.uploaded_urls):
                            dustbin_url = AZURE_FUNCTION_URL.replace('/upload_image', '/dustbin_detection')
                            response = requests.post(dustbin_url, json={'blob_url': url})
                            
                            if response.status_code == 200:
                                result = response.json()
                                status_icon = "✅" if result['status'] == "dustbin_found" else "❌"
                                st.write(f"{status_icon} Image {i+1}: {result['message']}")
                                if result['detections']:
                                    for det in result['detections']:
                                        st.write(f"   - {det['label']}: {det['confidence']:.2f}")
                            else:
                                st.error(f"❌ Detection failed for image {i+1}: {response.text}")
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")

else:
    st.info("📥 Please upload or capture images to proceed.")
