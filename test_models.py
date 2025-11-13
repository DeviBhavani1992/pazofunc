#!/usr/bin/env python3
import os
from ultralytics import YOLO

# Test model paths
MODEL_PATH = "/app/models/yolo11n.pt"
CLOTHING_MODEL_PATH = "/app/models/deepfashion2_yolov8s-seg.pt"
SHOE_MODEL_PATH = "/app/models/yolov11_fashipnpedia.pt"

print("🔍 Testing model loading...")

# Check if files exist
print(f"General model exists: {os.path.exists(MODEL_PATH)}")
print(f"Clothing model exists: {os.path.exists(CLOTHING_MODEL_PATH)}")
print(f"Shoe model exists: {os.path.exists(SHOE_MODEL_PATH)}")

# Try loading models
try:
    print("Loading general model...")
    model = YOLO(MODEL_PATH)
    print("✅ General model loaded")
except Exception as e:
    print(f"❌ General model failed: {e}")

try:
    print("Loading clothing model...")
    clothing_model = YOLO(CLOTHING_MODEL_PATH)
    print("✅ Clothing model loaded")
except Exception as e:
    print(f"❌ Clothing model failed: {e}")

try:
    print("Loading shoe model...")
    shoe_model = YOLO(SHOE_MODEL_PATH)
    print("✅ Shoe model loaded")
except Exception as e:
    print(f"❌ Shoe model failed: {e}")

print("✅ Model test complete")