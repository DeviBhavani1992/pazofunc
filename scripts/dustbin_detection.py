from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load dustbin detection YOLO model
model_path = "/home/devi-1202324/Azure/pazofunc/models/dustbin_yolo11_best.pt"
model = YOLO(model_path)

@app.route("/detect_dustbin", methods=["POST"])
def detect_dustbin():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    for i, file in enumerate(files, 1):
        image_path = f"/tmp/dustbin_input_{i}.jpg"
        file.save(image_path)

        model_results = model.predict(image_path, conf=0.25)
        r = model_results[0]
        names = r.names

        dustbins_detected = []
        if len(r.boxes):
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                label = names[int(cls)]
                confidence = float(conf)
                bbox = [float(x) for x in box.tolist()]
                
                dustbins_detected.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                })

        status = "dustbin_found" if dustbins_detected else "no_dustbin"
        message = f"Found {len(dustbins_detected)} dustbin(s)" if dustbins_detected else "No dustbin detected"
        
        results.append({
            "image": i,
            "status": status,
            "message": message,
            "detections": dustbins_detected,
            "total_dustbins": len(dustbins_detected)
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)