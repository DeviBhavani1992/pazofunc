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

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient, ContentSettings

app = FastAPI(title="YOLOv11 Unified Inference API", version="3.0")

# ---------------------------------------------------------------
# AZURE BLOB CONFIG
# ---------------------------------------------------------------
BLOB_CONNECTION_STR = "YOUR_STORAGE_CONNECTION_STRING_HERE"
CONTAINER_NAME = "images"

blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STR)
container_client = blob_service.get_container_client(CONTAINER_NAME)

def upload_to_blob(folder: str, filename: str, image_bytes: bytes):
    """
    Upload final inferenced image to Azure Blob Storage.
    folder = dresscode / dustbin / general
    """

    blob_path = f"uploads/{folder}/{filename}"
    
    content_settings = ContentSettings(content_type="image/jpeg")

    container_client.upload_blob(
        name=blob_path,
        data=image_bytes,
        overwrite=True,
        content_settings=content_settings
    )

    return f"uploads/{folder}/{filename}"


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
        MODEL_CACHE[model_key] = YOLO(model_path)
    return MODEL_CACHE[model_key]

# ---------------------------------------------------------------
# COLOR ANALYSIS
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
# INFERENCE HELPERS
# ---------------------------------------------------------------
def save_inferenced_image(pil_img):
    """Convert PIL to bytes for blob upload"""
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()

# ---------------------------------------------------------------
# GENERAL INFERENCE
# ---------------------------------------------------------------
def run_general_inference(image: Image.Image, source_name: str):
    model = get_model("general")
    results = model(image)

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

    # Convert result image to bytes
    infer_img = results[0].plot()  
    infer_img_pil = Image.fromarray(infer_img)
    img_bytes = save_inferenced_image(infer_img_pil)

    # Upload into folder: uploads/general/
    blob_path = upload_to_blob("general", source_name, img_bytes)

    return {
        "status": "success",
        "filename": source_name,
        "blob_saved_to": blob_path,
        "detections": detections
    }

# ---------------------------------------------------------------
# DRESSCODE INFERENCE
# ---------------------------------------------------------------
def run_dresscode_analysis(image: Image.Image, source_name: str):
    clothing_model = get_model("clothing")
    shoe_model = get_model("shoes")
    
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    shirt_color = pant_color = shoe_color = None
    
    # Clothing detection
    clothing_results = clothing_model(image)
    if len(clothing_results[0].boxes):
        for box, cls, conf in zip(clothing_results[0].boxes.xyxy,
                                  clothing_results[0].boxes.cls,
                                  clothing_results[0].boxes.conf):
            label = clothing_results[0].names[int(cls)].lower()
            color_rgb, _ = get_dominant_color_with_percentage(img, box)
            color_name = get_color_name(color_rgb)
            
            if any(k in label for k in ["shirt", "top", "t-shirt"]):
                shirt_color = color_name
            elif any(k in label for k in ["pant", "jean", "trouser"]):
                pant_color = color_name
    
    # Shoe detection
    shoe_results = shoe_model(image)
    if len(shoe_results[0].boxes):
        for box, cls, conf in zip(shoe_results[0].boxes.xyxy,
                                  shoe_results[0].boxes.cls,
                                  shoe_results[0].boxes.conf):
            label = shoe_results[0].names[int(cls)].lower()
            if any(k in label for k in ["shoe", "boot", "sneaker"]):
                color_rgb, _ = get_dominant_color_with_percentage(img, box)
                shoe_color = get_color_name(color_rgb)
    
    # Final Result
    if (shirt_color in ["white", "black"]) and (pant_color == "black") and (shoe_color == "black"):
        status = "compliant"
        message = "Dress code is appropriate."
    else:
        status = "non_compliant"
        message = "Dress code violation"

    # Save inferenced image
    infer_img = clothing_results[0].plot()
    infer_pil = Image.fromarray(infer_img)
    img_bytes = save_inferenced_image(infer_pil)

    blob_path = upload_to_blob("dresscode", source_name, img_bytes)

    return {
        "status": status,
        "message": message,
        "detections": {"shirt": shirt_color, "pant": pant_color, "shoe": shoe_color},
        "blob_saved_to": blob_path,
        "filename": source_name,
    }

# ---------------------------------------------------------------
# DUSTBIN INFERENCE
# ---------------------------------------------------------------
def run_dustbin_detection(image: Image.Image, source_name: str):
    model = get_model("dustbin")
    results = model(image)

    r = results[0]
    detections = []

    if len(r.boxes):
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            detections.append({
                "label": r.names[int(cls)],
                "confidence": float(conf)
            })

    # Save annotated image
    infer_img = r.plot()
    infer_pil = Image.fromarray(infer_img)
    img_bytes = save_inferenced_image(infer_pil)

    blob_path = upload_to_blob("dustbin", source_name, img_bytes)

    return {
        "status": "success",
        "detections": detections,
        "blob_saved_to": blob_path,
        "filename": source_name
    }

# ---------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------
@app.post("/infer")
async def general_infer(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return run_general_inference(image, file.filename)

@app.post("/check_dresscode")
async def check_dresscode(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return run_dresscode_analysis(image, file.filename)

@app.post("/dustbin_detect")
async def dustbin_detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return run_dustbin_detection(image, file.filename)

@app.get("/health")
async def health():
    return {"status": "healthy", "loaded_models": list(MODEL_CACHE.keys())}


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
