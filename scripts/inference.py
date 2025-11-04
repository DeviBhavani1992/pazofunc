 
"""
YOLOv11 Inference Service (FastAPI + Uvicorn)
Tolerant to Content-Type mismatches (e.g. application/octet-stream)
Saves inference results to Cosmos DB (Mongo API)
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
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "yolov11db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "yolov11-collection")

# -------------------------------------------------------------------
# Load YOLOv11 Model
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

# -------------------------------------------------------------------
# Connect to MongoDB (Optional)
# -------------------------------------------------------------------
collection = None
try:
    if MONGO_URI:
        mc = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
        db = mc[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        log(f"✅ MongoDB connected: DB={MONGO_DB}, Collection={MONGO_COLLECTION}")
    else:
        log("⚠️ MONGO_URI not set — skipping MongoDB logging.")
except Exception as e:
    log(f"⚠️ MongoDB connection failed: {e}")
    traceback.print_exc()
    collection = None

# -------------------------------------------------------------------
# Helper: Convert ObjectId safely
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

    # Log request Content-Type
    req_content_type = (request.headers.get("content-type") or "").lower()
    log(f"📦 Request Content-Type: {req_content_type}")

    # Parse JSON body robustly
    try:
        if "application/json" in req_content_type:
            data = await request.json()
        else:
            raw = await request.body()
            try:
                data = json.loads(raw.decode("utf-8"))
                log("⚠️ Parsed JSON manually from non-JSON content-type.")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid or missing JSON body")
    except Exception as ex:
        log(f"⚠️ JSON parse error: {ex}")
        raise HTTPException(status_code=400, detail="Invalid or missing JSON body")

    # Validate blob_url
    blob_url = data.get("blob_url")
    if not blob_url:
        log("⚠️ Missing blob_url in request")
        raise HTTPException(status_code=400, detail="Missing blob_url")

    log(f"🌐 Downloading blob: {blob_url}")
    image_response = None

    # Download image with retries
    for attempt in range(1, 4):
        try:
            image_response = requests.get(blob_url, timeout=20)
            image_response.raise_for_status()
            break
        except Exception as ex:
            log(f"⚠️ Attempt {attempt} to download blob failed: {ex}")
            if attempt == 3:
                raise HTTPException(status_code=500, detail=f"Failed to download blob: {ex}")
            time.sleep(1)

    # Validate blob content type
    blob_ct = (image_response.headers.get("Content-Type") or "").lower()
    log(f"📦 Blob Content-Type: {blob_ct}")

    if not (("image" in blob_ct) or ("octet-stream" in blob_ct) or (blob_ct == "")):
        log(f"⚠️ Unacceptable blob Content-Type: {blob_ct}")
        raise HTTPException(status_code=400, detail=f"Invalid blob content type: {blob_ct}")

    # Decode image
    try:
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        log(f"✅ Image decoded successfully (size={image.size})")
    except UnidentifiedImageError:
        log("⚠️ PIL could not identify image; trying imghdr fallback.")
        kind = imghdr.what(None, h=image_response.content)
        if not kind:
            raise HTTPException(status_code=400, detail="Invalid image data")
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        log(f"✅ Image decoded via imghdr fallback: type={kind}")
    except Exception as ex:
        log(f"❌ Image decode failed: {ex}")
        raise HTTPException(status_code=400, detail="Failed to decode image")

    # Run YOLO inference
    try:
        log("🧠 Running YOLOv11 inference...")
        results = model.predict(source=image, conf=0.25)
        log("✅ Inference complete.")
    except Exception as ex:
        log(f"❌ Inference failed: {ex}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Inference failed")

    # Extract detections
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

    # Save to MongoDB (best-effort)
    try:
        if collection is not None:
            insert_result = collection.insert_one(record)
            record["_id"] = insert_result.inserted_id
            log("✅ Saved inference result to MongoDB.")
        else:
            log("⚠️ MongoDB collection not available; skipping save.")
    except Exception as e:
        log(f"⚠️ MongoDB insertion failed: {e}")
        log(traceback.format_exc())

    # Convert any ObjectIds before returning
    safe_record = to_json_safe(record)

    log("✅ Returning inference result (200 OK).")
    return JSONResponse(content=safe_record, status_code=200)

# -------------------------------------------------------------------
# Run locally
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    log("🚀 Starting YOLOv11 FastAPI app on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

