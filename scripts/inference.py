import os
import time
import io
import requests
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from typing import List, Optional
import uvicorn
from ultralytics import YOLO

app = FastAPI(title="YOLOv11 Unified Inference API", version="2.0")

# ---------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------
MODEL_PATHS = {
    "clothing": "/app/models/deepfashion2_yolov8s-seg.pt",
    "shoes": "/app/models/yolov11_fashipnpedia.pt",
    "dustbin": "/app/models/dustbin_yolo11_best.pt",
    "general": "/app/models/yolo11n.pt",
}

MODEL_CACHE = {}

def get_model(model_key: str):
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Invalid model key '{model_key}'")
    if model_key not in MODEL_CACHE:
        model_path = MODEL_PATHS[model_key]
        print(f"🔄 Loading model: {model_path}")
        MODEL_CACHE[model_key] = YOLO(model_path)
    return MODEL_CACHE[model_key]

# ---------------------------------------------------------------
# COLOR ANALYSIS FUNCTIONS
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------
class BlobRequest(BaseModel):
    blob_url: str

# ---------------------------------------------------------------
# INFERENCE FUNCTIONS
# ---------------------------------------------------------------
def run_general_inference(image: Image.Image, source_name: str):
    start_time = time.time()
    model = get_model("general")
    results = model(image)
    elapsed = round(time.time() - start_time, 2)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names.get(cls_id, str(cls_id))
            detections.append({
                "class": label,
                "confidence": round(conf, 2)
            })

    class_counts = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    return {
        "status": "success",
        "model": os.path.basename(MODEL_PATHS["general"]),
        "inference_type": "general",
        "filename": source_name,
        "processing_time_sec": elapsed,
        "total_detections": len(detections),
        "detections_by_class": class_counts,
        "detections": detections,
    }

def run_dresscode_analysis(image: Image.Image, source_name: str):
    start_time = time.time()
    
    # Load both models
    clothing_model = get_model("clothing")
    shoe_model = get_model("shoes")
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    shirt_color = pant_color = shoe_color = None
    
    # Detect clothing (shirts, pants) using DeepFashion2 model
    clothing_results = clothing_model(image)
    if len(clothing_results[0].boxes):
        for box, cls, conf in zip(clothing_results[0].boxes.xyxy, clothing_results[0].boxes.cls, clothing_results[0].boxes.conf):
            label = clothing_results[0].names[int(cls)].lower()
            color_rgb, _ = get_dominant_color_with_percentage(img, box)
            color_name = get_color_name(color_rgb)
            
            if any(k in label for k in ["shirt", "blouse", "top", "t-shirt", "tee", "upper"]):
                shirt_color = color_name
            elif any(k in label for k in ["pant", "trouser", "jean", "slacks", "lower"]):
                pant_color = color_name
    
    # Detect shoes using Fashionpedia model
    shoe_results = shoe_model(image)
    if len(shoe_results[0].boxes):
        for box, cls, conf in zip(shoe_results[0].boxes.xyxy, shoe_results[0].boxes.cls, shoe_results[0].boxes.conf):
            label = shoe_results[0].names[int(cls)].lower()
            if any(k in label for k in ["shoe", "footwear", "sneaker", "boot"]):
                color_rgb, _ = get_dominant_color_with_percentage(img, box)
                shoe_color = get_color_name(color_rgb)
    
    elapsed = round(time.time() - start_time, 2)
    
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
    
    return {
        "status": status,
        "message": message,
        "detections": {"shirt": shirt_color, "pant": pant_color, "shoe": shoe_color},
        "processing_time_sec": elapsed,
        "filename": source_name
    }

def run_dustbin_detection(image: Image.Image, source_name: str):
    start_time = time.time()
    model = get_model("dustbin")
    results = model(image)
    elapsed = round(time.time() - start_time, 2)
    
    r = results[0]
    names = r.names
    
    detections = []
    if len(r.boxes):
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            detections.append({
                "label": names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x) for x in box.tolist()]
            })
    
    status = "dustbin_found" if detections else "no_dustbin"
    message = f"Found {len(detections)} dustbin(s)" if detections else "No dustbin detected"
    
    return {
        "status": status,
        "message": message,
        "detections": detections,
        "total_dustbins": len(detections),
        "processing_time_sec": elapsed,
        "filename": source_name
    }

# ---------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------
@app.post("/infer")
async def general_infer(file: UploadFile = File(...)):
    """General YOLOv11 inference"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = run_general_inference(image, file.filename)
    return result

@app.post("/check_dresscode")
async def check_dresscode(file: Optional[UploadFile] = None, req: Optional[BlobRequest] = None):
    """Dress code analysis"""
    image = None
    source_name = "unknown"
    
    if file:
        source_name = file.filename
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    elif req and req.blob_url:
        resp = requests.get(req.blob_url)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        source_name = os.path.basename(req.blob_url)
    else:
        return {"error": "No image provided"}
    
    return run_dresscode_analysis(image, source_name)

@app.post("/dustbin_detect")
async def dustbin_detect(file: Optional[UploadFile] = None, req: Optional[BlobRequest] = None):
    """Dustbin detection"""
    image = None
    source_name = "unknown"
    
    if file:
        source_name = file.filename
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    elif req and req.blob_url:
        resp = requests.get(req.blob_url)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        source_name = os.path.basename(req.blob_url)
    else:
        return {"error": "No image provided"}
    
    return run_dustbin_detection(image, source_name)

@app.get("/health")
async def health():
    return {"status": "healthy", "loaded_models": list(MODEL_CACHE.keys())}

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)