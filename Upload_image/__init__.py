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

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://localhost:8000")  # 👈 change if needed
INFERENCE_BASE_URL = os.getenv("YOLO_ENDPOINT")
if not INFERENCE_BASE_URL:
    raise ValueError("YOLO_ENDPOINT is not set in Function App configuration")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "uploads")

# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------
try:
    dress_model = YOLO(os.path.join(os.path.dirname(__file__), "..", "models", "yolov11_fashipnpedia.pt"))
    logger.info("✅ Dress code model loaded")
except Exception as e:
    logger.error(f"❌ Dress code model loading failed: {e}")
    dress_model = None

try:
    dustbin_model = YOLO(os.path.join(os.path.dirname(__file__), "..", "models", "dustbin_yolo11_best.pt"))
    logger.info("✅ Dustbin model loaded")
except Exception as e:
    logger.error(f"❌ Dustbin model loading failed: {e}")
    dustbin_model = None

# -------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# -------------------------------------------------------------------
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

def analyze_dresscode(blob_url):
    if not dress_model:
        return {"error": "Dress code model not available"}
    
    try:
        response = requests.get(blob_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image from {blob_url}"}
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(response.content)
            image_path = tmp_file.name

        img = cv2.imread(image_path)
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

        os.unlink(image_path)
        return {
            "status": status,
            "message": message,
            "detections": {"shirt": shirt_color, "pant": pant_color, "shoe": shoe_color}
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_dustbin(blob_url):
    if not dustbin_model:
        return {"error": "Dustbin model not available"}
    
    try:
        response = requests.get(blob_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image from {blob_url}"}
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(response.content)
            image_path = tmp_file.name

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

        os.unlink(image_path)
        return {
            "status": "dustbin_found" if detections else "no_dustbin",
            "message": f"Found {len(detections)} dustbin(s)" if detections else "No dustbin detected",
            "detections": detections
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def detect_image_content_type(filename: str, data: bytes) -> str:
    detected_type, _ = guess_type(filename)
    if detected_type and detected_type.startswith("image/"):
        return detected_type
    kind = imghdr.what(None, data)
    return f"image/{kind or 'jpeg'}"

def call_inference(blob_url, mode):
    """Call local analysis functions"""
    logger.info(f"🧠 Running {mode} analysis on {blob_url}")
    
    if mode == "dresscode":
        return analyze_dresscode(blob_url)
    elif mode == "dustbin":
        return analyze_dustbin(blob_url)
    else:
        # For general inference, try external API if available
        if INFERENCE_BASE_URL:
            try:
                endpoint = f"{INFERENCE_BASE_URL}/infer"
                resp = requests.post(endpoint, json={"blob_url": blob_url}, timeout=180)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.error(f"⚠️ External inference failed: {e}")
        return {"error": "General inference not available", "blob_url": blob_url}

# -------------------------------------------------------------------
# MAIN FUNCTION ENTRY
# -------------------------------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()
    logger.info("🚀 [Azure Function Triggered] YOLOv11 unified pipeline")

    try:
        blob_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not blob_conn_str:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING in environment variables")

        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service.get_container_client(BLOB_CONTAINER)

        try:
            container_client.create_container()
        except Exception:
            pass  # ignore if already exists

        # Handle files
        files = req.files.getlist("file") or list(req.files.values())
        if not files:
            logger.warning("⚠️ No files received in request")
            return func.HttpResponse(
                json.dumps({"error": "No image files uploaded"}),
                status_code=400,
                mimetype="application/json",
            )

        results = []
        for idx, file in enumerate(files, start=1):
            image_bytes = file.stream.read()
            filename = file.filename
            logger.info(f"📤 Uploading {filename} to blob storage...")

            content_type = detect_image_content_type(filename, image_bytes)
            container_client.upload_blob(
                name=filename,
                data=image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )

            blob_url = f"{container_client.url}/{filename}"
            logger.info(f"✅ Uploaded blob URL: {blob_url}")

            # Detect inference type based on file prefix
            if filename.lower().startswith("dresscode_"):
                analysis_type = "dresscode"
            elif filename.lower().startswith("dustbin_"):
                analysis_type = "dustbin"
            else:
                analysis_type = "general"

            # Call inference service
            inference_result = call_inference(blob_url, analysis_type)
            inference_result["blob_url"] = blob_url
            inference_result["filename"] = filename
            inference_result["type"] = analysis_type
            results.append(inference_result)

        total_time = round(time.time() - start_time, 2)
        logger.info(f"🏁 Pipeline completed in {total_time}s for {len(results)} image(s)")

        return func.HttpResponse(
            json.dumps({"results": results, "processing_time": total_time}),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logger.error(f"🔥 Exception in pipeline: {e}")
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
