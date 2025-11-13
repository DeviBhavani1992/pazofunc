import os
import io
import time
import torch
import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from typing import Optional
import uvicorn
from ultralytics import YOLO

app = FastAPI(title="YOLOv11 Unified Inference API", version="1.1")

# -------------------------------------------------------------------
# MODEL PATHS
# -------------------------------------------------------------------
MODEL_PATHS = {
    "fashionpedia": "/home/devi-1202324/Azure/pazofunc/models/yolov11_fashipnpedia.pt",  # shoes
    "deepfashion2": "/home/devi-1202324/Azure/pazofunc/models/deepfashion2_yolov8s-seg.pt",  # shirts/pants
    "dustbin": "/home/devi-1202324/Azure/pazofunc/models/dustbin_yolo11_best.pt",
    "general": "/home/devi-1202324/Azure/pazofunc/models/yolo11m-seg.pt",
}

MODEL_CACHE = {}


# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------
def get_model(model_key: str):
    if model_key not in MODEL_PATHS:
        raise ValueError(f"❌ Invalid model key: {model_key}")
    if model_key not in MODEL_CACHE:
        model_path = MODEL_PATHS[model_key]
        print(f"🔄 Loading model: {model_path}")
        MODEL_CACHE[model_key] = YOLO(model_path)
    return MODEL_CACHE[model_key]


# -------------------------------------------------------------------
# REQUEST MODELS
# -------------------------------------------------------------------
class BlobRequest(BaseModel):
    blob_url: str


# -------------------------------------------------------------------
# INFERENCE HELPER
# -------------------------------------------------------------------
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

    # Summaries
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


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def fetch_image_from_blob(blob_url: str) -> Image.Image:
    resp = requests.get(blob_url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.post("/infer")
async def general_infer(file: UploadFile = File(...)):
    """Generic YOLO model inference"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return run_inference("general", image, file.filename)


@app.post("/check_dresscode")
async def check_dresscode(req: Optional[BlobRequest] = None, file: Optional[UploadFile] = None):
    """
    Dresscode detection:
      - Uses DeepFashion2 for shirts/pants
      - Uses Fashionpedia for shoes
    """
    image = None
    source_name = "unknown"

    # Handle input
    if file:
        source_name = file.filename
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    elif req and req.blob_url:
        image = fetch_image_from_blob(req.blob_url)
        source_name = os.path.basename(req.blob_url)
    else:
        return {"error": "No image provided"}

    # Run two models
    fashionpedia_result = run_inference("fashionpedia", image, source_name)
    deepfashion_result = run_inference("deepfashion2", image, source_name)

    combined = {
        "status": "success",
        "inference_type": "dresscode",
        "results": {
            "fashionpedia": fashionpedia_result,
            "deepfashion2": deepfashion_result
        }
    }
    return combined


@app.post("/dustbin_detect")
async def dustbin_detect(req: Optional[BlobRequest] = None, file: Optional[UploadFile] = None):
    """Dustbin detection"""
    image = None
    source_name = "unknown"

    if file:
        source_name = file.filename
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    elif req and req.blob_url:
        image = fetch_image_from_blob(req.blob_url)
        source_name = os.path.basename(req.blob_url)
    else:
        return {"error": "No image provided"}

    return run_inference("dustbin", image, source_name)


# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "loaded_models": list(MODEL_CACHE.keys())}


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
