"""
YOLOv11 Inference Service (FastAPI + Uvicorn)
Tolerant to Content-Type mismatches (e.g. application/octet-stream)
Optional MongoDB logging (skipped if Cosmos not configured)
"""

from ultralytics import YOLO
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from PIL import Image, UnidentifiedImageError
from bson import ObjectId
import requests
import io
import os
import datetime
import traceback
import json
import logging
import imghdr
import time
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# -------------------------------------------------------------------
# App + Logging Setup
# -------------------------------------------------------------------
app = FastAPI(title="YOLOv11 Inference Service", version="1.6")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("yolov11")

def log(msg: str):
    logger.info(msg)

# -------------------------------------------------------------------
# Configurations from Environment
# -------------------------------------------------------------------
MODEL_PATH = "/app/models/yolo11n.pt"
CLOTHING_MODEL_PATH = "/app/models/deepfashion2_yolov8s-seg.pt"
SHOE_MODEL_PATH = "/app/models/yolov11_fashipnpedia.pt"
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "yolov11db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "yolov11-collection")

# -------------------------------------------------------------------
# Load YOLOv11 Models
# -------------------------------------------------------------------
try:
    log("🔄 Loading YOLOv11 model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    log("✅ YOLOv11 model loaded successfully.")
except Exception as e:
    log(f"❌ Model load failed: {e}")
    traceback.print_exc()
    raise

# Load models for dress code detection
clothing_model = None
shoe_model = None

try:
    if os.path.exists(CLOTHING_MODEL_PATH):
        log("🔄 Loading Clothing model (DeepFashion2)...")
        clothing_model = YOLO(CLOTHING_MODEL_PATH)
        log("✅ Clothing model loaded successfully.")
except Exception as e:
    log(f"⚠️ Clothing model load failed: {e}")

try:
    if os.path.exists(SHOE_MODEL_PATH):
        log("🔄 Loading Shoe model (Fashionpedia)...")
        shoe_model = YOLO(SHOE_MODEL_PATH)
        log("✅ Shoe model loaded successfully.")
except Exception as e:
    log(f"⚠️ Shoe model load failed: {e}")

# -------------------------------------------------------------------
# Connect to MongoDB (Optional)
# -------------------------------------------------------------------
collection = None
if MONGO_URI.strip():
    try:
        mc = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
        db = mc[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        log(f"✅ MongoDB connected: DB={MONGO_DB}, Collection={MONGO_COLLECTION}")
    except Exception as e:
        log(f"⚠️ MongoDB connection failed: {e}")
        collection = None
else:
    log("⚠️ MONGO_URI not provided — skipping MongoDB setup.")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def to_json_safe(obj):
    """Recursively convert ObjectId and other non-JSON types to strings."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, list):
        return [to_json_safe(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    return obj

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

# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------
@app.get("/")
async def health():
    log("🩺 Health check triggered.")
    return {"status": "ok"}

# -------------------------------------------------------------------
# Inference Endpoint
# -------------------------------------------------------------------
@app.post("/infer")
async def infer(request: Request):
    log("📥 Received /infer request")

    # Parse JSON body robustly
    try:
        data = await request.json()
    except Exception:
        raw = await request.body()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid or missing JSON body")

    blob_url = data.get("blob_url")
    if not blob_url:
        raise HTTPException(status_code=400, detail="Missing blob_url")

    # Download image
    log(f"🌐 Downloading blob: {blob_url}")
    try:
        resp = requests.get(blob_url, timeout=20)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to fetch or decode image: {ex}")

    # Run YOLO inference
    try:
        log("🧠 Running YOLOv11 inference...")
        results = model.predict(source=image, conf=0.25)
        log("✅ Inference complete.")
    except Exception as ex:
        log(f"❌ Inference failed: {ex}")
        raise HTTPException(status_code=500, detail="Inference failed")

    detections = []
    for r in results:
        for box in getattr(r, "boxes", []):
            detections.append({
                "class": int(box.cls),
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()]
            })

    record = {
        "blob_url": blob_url,
        "detections": detections,
        "total_objects": len(detections),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

    # Save to MongoDB (skip if not connected)
    if collection:
        try:
            insert_result = collection.insert_one(record)
            record["_id"] = insert_result.inserted_id
            log("✅ Saved inference result to MongoDB.")
        except Exception as e:
            log(f"⚠️ MongoDB insertion failed: {e}")
            log(traceback.format_exc())
    else:
        log("⚠️ Skipping MongoDB insertion (no connection).")

    return JSONResponse(content=to_json_safe(record), status_code=200)

# -------------------------------------------------------------------
# Dress Code Check Endpoint (UNTOUCHED)
# -------------------------------------------------------------------
@app.post("/check_dresscode")
async def check_dresscode(request: Request):
    log("📥 Received /check_dresscode request")

    if not clothing_model or not shoe_model:
        raise HTTPException(status_code=503, detail="Required models not available")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    blob_url = data.get("blob_url")
    if not blob_url:
        raise HTTPException(status_code=400, detail="Missing blob_url")

    try:
        image_response = requests.get(blob_url, timeout=20)
        image_response.raise_for_status()
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to download/process image: {ex}")

    try:
        shirt_color = pant_color = shoe_color = None

        clothing_results = clothing_model.predict(source=image, conf=0.25)
        clothing_r = clothing_results[0]
        clothing_names = clothing_r.names

        if len(clothing_r.boxes):
            for box, cls, conf in zip(clothing_r.boxes.xyxy, clothing_r.boxes.cls, clothing_r.boxes.conf):
                label = clothing_names[int(cls)].lower()
                color_rgb, _ = get_dominant_color_with_percentage(img_cv, box)
                color_name = get_color_name(color_rgb)

                if any(k in label for k in ["shirt", "blouse", "top", "t-shirt", "tee"]):
                    shirt_color = color_name
                elif any(k in label for k in ["pant", "trouser", "jean", "slacks"]):
                    pant_color = color_name

        shoe_results = shoe_model.predict(source=image, conf=0.25)
        shoe_r = shoe_results[0]
        shoe_names = shoe_r.names

        if len(shoe_r.boxes):
            for box, cls, conf in zip(shoe_r.boxes.xyxy, shoe_r.boxes.cls, shoe_r.boxes.conf):
                label = shoe_names[int(cls)].lower()
                color_rgb, _ = get_dominant_color_with_percentage(img_cv, box)
                color_name = get_color_name(color_rgb)

                if any(k in label for k in ["shoe", "footwear", "sneaker", "boot"]):
                    shoe_color = color_name

        violations = []
        if not shirt_color:
            violations.append("shirt not detected")
        elif shirt_color not in ["white", "black"]:
            violations.append("shirt must be white or black")

        if not pant_color:
            violations.append("pants not detected")
        elif pant_color != "black":
            violations.append("pants must be black")

        if not shoe_color:
            violations.append("shoes not detected")
        elif shoe_color != "black":
            violations.append("shoes must be black")

        status = "compliant" if not violations else "non_compliant"
        message = "Dress code is appropriate." if not violations else f"Dress code violation: {', '.join(violations)}"

        result = {
            "blob_url": blob_url,
            "status": status,
            "message": message,
            "detections": {
                "shirt": shirt_color,
                "pant": pant_color,
                "shoe": shoe_color,
            },
            "models_used": {
                "clothing": "deepfashion2_yolov8s-seg.pt",
                "shoes": "yolov11_fashipnpedia.pt",
            },
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

        log("✅ Dress code check complete")
        return JSONResponse(content=result, status_code=200)

    except Exception as ex:
        log(f"❌ Dress code check failed: {ex}")
        raise HTTPException(status_code=500, detail="Dress code check failed")

# -------------------------------------------------------------------
# Run locally
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    log("🚀 Starting YOLOv11 FastAPI app on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
