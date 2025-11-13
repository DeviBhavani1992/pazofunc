import logging
import azure.functions as func
import requests
import os
import json
import tempfile
import traceback
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dresscode_analysis")

# Load dress code model
try:
    dress_model = YOLO(os.path.join(os.path.dirname(__file__), "..", "models", "yolov11_fashipnpedia.pt"))
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    dress_model = None

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

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("Dress code analysis triggered")

    try:
        data = req.get_json()
        blob_url = data.get('blob_url')
        
        if not blob_url:
            return func.HttpResponse(
                json.dumps({"error": "Missing blob_url"}),
                status_code=400,
                mimetype="application/json"
            )

        if not dress_model:
            return func.HttpResponse(
                json.dumps({"error": "Dress code model not available"}),
                status_code=500,
                mimetype="application/json"
            )

        # Download image
        response = requests.get(blob_url)
        if response.status_code != 200:
            return func.HttpResponse(
                json.dumps({"error": f"Failed to download image from {blob_url}"}),
                status_code=400,
                mimetype="application/json"
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(response.content)
            image_path = tmp_file.name

        img = cv2.imread(image_path)
        
        # Run inference
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

        result = {
            "status": status,
            "message": message,
            "detections": {"shirt": shirt_color, "pant": pant_color, "shoe": shoe_color},
            "blob_url": blob_url
        }

        os.unlink(image_path)
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as ex:
        logger.error(f"Error: {ex}")
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json"
        )