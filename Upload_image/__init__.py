import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from pymongo import MongoClient
import requests
import os
import time
from datetime import datetime
import traceback
import json
from mimetypes import guess_type
import imghdr
import tempfile
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_image")

# Load Models
try:
    dress_model = YOLO(os.path.join(os.path.dirname(__file__), "..", "models", "yolov11_fashipnpedia.pt"))
    dustbin_model = YOLO(os.path.join(os.path.dirname(__file__), "..", "models", "dustbin_yolo11_best.pt"))
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    dress_model = None
    dustbin_model = None

def detect_image_content_type(filename: str, data: bytes) -> str:
    detected_type, _ = guess_type(filename)
    if detected_type and detected_type.startswith("image/"):
        return detected_type
    kind = imghdr.what(None, data)
    if kind:
        return f"image/{kind}"
    return "image/jpeg"

def get_dominant_color_with_percentage(image, box, k=3):
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return (0, 0, 0), 0.0
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pixels = cropped.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
    counts = Counter(kmeans.labels_)
    total = sum(counts.values())
    dominant_idx, dominant_count = counts.most_common(1)[0]
    dominant_color = kmeans.cluster_centers_[dominant_idx]
    percentage = (dominant_count / total) * 100
    return dominant_color, percentage

def get_color_name(rgb):
    r, g, b = rgb
    brightness = np.mean([r, g, b])
    if brightness > 170 and abs(r - g) < 40 and abs(r - b) < 40:
        return "white"
    elif brightness < 90:
        return "black"
    else:
        return "other"

def analyze_from_blob_url(blob_url, analysis_type):
    try:
        # Download image from blob
        response = requests.get(blob_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image from {blob_url}"}
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(response.content)
            image_path = tmp_file.name
        logger.info(f"[DEBUG] Downloaded image size: {len(response.content)} bytes")
        logger.info(f"[DEBUG] Image path exists: {os.path.exists(image_path)}")


        img = cv2.imread(image_path)
        logger.info(f"[DEBUG] Starting analysis type: {analysis_type}")
        logger.info(f"[DEBUG] Dress model loaded: {dress_model is not None}")
        logger.info(f"[DEBUG] Dustbin model loaded: {dustbin_model is not None}")

        if analysis_type == "dresscode":
            if not dress_model:
                return {"error": "Dress code model not available"}
            
            model_results = dress_model.predict(image_path, conf=0.25)
            r = model_results[0]
            names = r.names

            shirt_color = pant_color = shoe_color = None

            if len(r.boxes):
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    label = names[int(cls)].lower()
                    color_rgb, _ = get_dominant_color_with_percentage(img, box)
                    color_name = get_color_name(color_rgb)

                    if any(k in label for k in ["shirt", "blouse", "top", "t-shirt", "tee"]):
                        shirt_color = color_name
                    elif any(k in label for k in ["pant", "trouser", "jean", "slacks"]):
                        pant_color = color_name
                    elif any(k in label for k in ["shoe", "footwear", "sneaker", "boot"]):
                        shoe_color = color_name

            if (shirt_color in ["white", "black"]) and (pant_color == "black") and (shoe_color == "black"):
                status = "compliant"
                message = "Dress code is appropriate."
            else:
                violations = []
                if shirt_color not in ["white", "black"]:
                    violations.append("shirt must be white or black")
                if pant_color != "black":
                    violations.append("pants must be black")
                if shoe_color != "black":
                    violations.append("shoes must be black")
                status = "non_compliant"
                message = f"Dress code violation: {', '.join(violations)}"
            
            result = {
                "status": status,
                "message": message,
                "detections": {"shirt": shirt_color, "pant": pant_color, "shoe": shoe_color}
            }

        elif analysis_type == "dustbin":
            if not dustbin_model:
                return {"error": "Dustbin model not available"}
            
            model_results = dustbin_model.predict(image_path, conf=0.25)
            r = model_results[0]
            names = r.names

            detections = []
            if len(r.boxes):
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    detections.append({
                        "label": names[int(cls)],
                        "confidence": float(conf),
                        "bbox": [float(x) for x in box.tolist()]
                    })

            result = {
                "status": "dustbin_found" if detections else "no_dustbin",
                "message": f"Found {len(detections)} dustbin(s)" if detections else "No dustbin detected",
                "detections": detections
            }

        os.unlink(image_path)
        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e)}

def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("🔵 [START] Multi-function triggered")

    try:
        # Get all parameters and log them
        all_params = dict(req.params)
        logger.info(f"All request params: {all_params}")
        
        action = req.params.get('action', 'upload')
        blob_url = req.params.get('blob_url')
        
        # Also check form data for parameters
        if not blob_url:
            try:
                form_data = req.get_json() or {}
                blob_url = form_data.get('blob_url')
            except:
                pass
        
        logger.info(f"Detected - Action: {action}, Blob URL: {blob_url}")
        
        # Force action detection for testing
        if 'action' in all_params:
            action = all_params['action']
            logger.info(f"Forced action from params: {action}")

        # If blob_url provided, analyze directly
        if blob_url and action in ['dresscode', 'dustbin']:
            logger.info(f"🔍 Analyzing blob URL: {blob_url}")
            try:
                analysis_result = analyze_from_blob_url(blob_url, action)
                analysis_result["image"] = 1
                analysis_result["blob_url"] = blob_url
                
                return func.HttpResponse(
                    json.dumps({"results": [analysis_result]}),
                    status_code=200,
                    mimetype="application/json"
                )
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return func.HttpResponse(
                    json.dumps({"error": f"Analysis failed: {str(e)}"}),
                    status_code=500,
                    mimetype="application/json"
                )

        # Get files from request only if no blob_url
        if not blob_url:
            files = req.files.getlist('files') or list(req.files.values())
            if not files:
                logger.error("❌ No files found and no blob_url provided")
                return func.HttpResponse(json.dumps({"error": "No files uploaded"}), status_code=400, mimetype="application/json")
        else:
            files = []

        # Get environment variables
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        
        if not blob_conn_str:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING")

        # Initialize blob service
        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(blob_container)

        # Create container if it doesn't exist
        try:
            container_client.create_container()
        except Exception:
            pass

        # Upload files to blob storage
        uploaded_urls = []
        for i, file in enumerate(files):
            image_bytes = file.stream.read()
            image_name = file.filename or f"image_{i+1}.jpg"
            size_kb = round(len(image_bytes) / 1024, 1)
            logger.info(f"📁 Uploading file: {image_name} ({size_kb} KB)")
            
            content_type = detect_image_content_type(image_name, image_bytes)
            container_client.upload_blob(
                name=image_name,
                data=image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )
            
            blob_url = f"{container_client.url}/{image_name}"
            uploaded_urls.append(blob_url)
            logger.info(f"✅ Uploaded: {blob_url}")

        # Perform analysis if requested
        if action in ['dresscode', 'dustbin']:
            results = []
            for i, blob_url in enumerate(uploaded_urls, 1):
                logger.info(f"🔍 Analyzing image {i}: {blob_url}")
                analysis_result = analyze_from_blob_url(blob_url, action)
                analysis_result["image"] = i
                analysis_result["blob_url"] = blob_url
                results.append(analysis_result)
            
            total_time = round(time.time() - start_time, 2)
            logger.info(f"🏁 Analysis completed in {total_time}s")
            
            return func.HttpResponse(
                json.dumps({"results": results}),
                status_code=200,
                mimetype="application/json"
            )

        # Default upload response
        total_time = round(time.time() - start_time, 2)
        logger.info(f"🏁 Upload completed in {total_time}s")

        return func.HttpResponse(
            json.dumps({
                "status": "success", 
                "uploaded": uploaded_urls,
                "count": len(uploaded_urls)
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as ex:
        logger.error("🔥 Exception in function")
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json"
        )