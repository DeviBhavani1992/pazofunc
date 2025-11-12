from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

app = Flask(__name__)

# Load Fashionpedia YOLO model
model_path = "/home/devi-1202324/Azure/pazofunc/models/yolov11_fashipnpedia.pt"
model = YOLO(model_path)

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

@app.route("/check_dresscode", methods=["POST"])
def check_dresscode():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    for i, file in enumerate(files, 1):
        image_path = f"/tmp/input_{i}.jpg"
        file.save(image_path)
        img = cv2.imread(image_path)

        model_results = model.predict(image_path, conf=0.25)
        r = model_results[0]
        names = r.names

        shirt_color = None
        pant_color = None
        shoe_color = None

        if len(r.boxes):
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                label = names[int(cls)].lower()
                color_rgb, color_percent = get_dominant_color_with_percentage(img, box)
                color_name = get_color_name(color_rgb)

                if any(k in label for k in ["shirt", "blouse", "top", "t-shirt", "tee"]):
                    shirt_color = color_name
                elif any(k in label for k in ["pant", "trouser", "jean", "slacks"]):
                    pant_color = color_name
                elif any(k in label for k in ["shoe", "footwear", "sneaker", "boot"]):
                    shoe_color = color_name

        # Dress code logic
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
        
        results.append({
            "image": i,
            "status": status,
            "message": message,
            "detections": {
                "shirt": shirt_color,
                "pant": pant_color,
                "shoe": shoe_color
            }
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)