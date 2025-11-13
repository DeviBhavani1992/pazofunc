import os
import time
import io
import torch
import requests
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
from typing import List, Optional
import uvicorn
from ultralytics import YOLO

app = FastAPI(title="YOLOv11 Inference API", version="1.0")

# ---------------------------------------------------------------
# MODEL PATHS (Update as per your folder)
# ---------------------------------------------------------------
MODEL_PATHS = {
    "dresscode": "/home/devi-1202324/Azure/pazofunc/models/yolov11_fashipnpedia.pt",
    "dustbin": "/home/devi-1202324/Azure/pazofunc/models/dustbin_yolo11_best.pt",
    "general": "/home/devi-1202324/Azure/pazofunc/models/yolo11m-seg.pt",
}

# Cache models to avoid reloading each request
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
# Request models
# ---------------------------------------------------------------
class BlobRequest(BaseModel):
    blob_url: str


# ---------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------
def run_inference(model_key: str, image: Image.Image, source_name: str):
    start_time = time.time()
    model = get_model(model_key)
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

    # Summary
    class_counts = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    return {
        "status": "success",
        "model": os.path.basename(MODEL_PATHS[model_key]),
        "inference_type": model_key,
        "filename": source_name,
        "processing_time_sec": elapsed,
        "total_detections": len(detections),
        "detections_by_class": class_counts,
        "detections": detections,
    }


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------

@app.post("/infer")
async def general_infer(file: UploadFile = File(...)):
    """General YOLOv11 model"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = run_inference("general", image, file.filename)
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
    return run_inference("dresscode", image, source_name)


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
    return run_inference("dustbin", image, source_name)


# ---------------------------------------------------------------
# HEALTHCHECK
# ---------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "loaded_models": list(MODEL_CACHE.keys())}


# ---------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
