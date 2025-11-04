from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load YOLOE open-vocabulary model
model_path = "/app/models/yoloe_s.pt"
model = YOLO(model_path)

# Define your dress code
MANDATORY_ITEMS = ["white shirt", "black pant", "shoe"]
OPTIONAL_ITEMS = ["jacket"]

@app.route("/check_dresscode", methods=["POST"])
def check_dresscode():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    image_path = "/tmp/input.jpg"
    file.save(image_path)

    # Run detection for each prompt
    missing_items = []
    detections = {}

    for item in MANDATORY_ITEMS + OPTIONAL_ITEMS:
        results = model.predict(image_path, prompt=item)
        if len(results[0].boxes) > 0:
            detections[item] = True
        else:
            detections[item] = False

    # Check compliance
    for item in MANDATORY_ITEMS:
        if not detections[item]:
            missing_items.append(item)

    if missing_items:
        message = f"Dress code violation: missing {', '.join(missing_items)}"
        status = "non_compliant"
    else:
        message = "Dress code is appropriate."
        status = "compliant"

    return jsonify({
        "status": status,
        "message": message,
        "detections": detections
    })
