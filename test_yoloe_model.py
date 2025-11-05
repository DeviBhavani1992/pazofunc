from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import csv
from collections import Counter
from sklearn.cluster import KMeans

# ------------------------------------
# Paths
# ------------------------------------
MODEL_PATH = "/home/devi-1202324/Azure/pazofunc/models/deepfashion2_yolov8s-seg.pt"
IMAGE_FOLDER = "/home/devi-1202324/Downloads/OneDrive_2_10-31-2025"
OUTPUT_CSV = "/home/devi-1202324/Azure/pazofunc/deepfashion2_dresscode.csv"

# ------------------------------------
# Helper: Dominant color and percentage
# ------------------------------------
def get_dominant_color_with_percentage(image, box, k=3):
    """Return dominant color RGB and its percentage inside the detected region"""
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
    """Classify RGB color into white / black / other"""
    r, g, b = rgb
    brightness = np.mean([r, g, b])
    # improved black range detection
    if brightness > 170 and abs(r - g) < 40 and abs(r - b) < 40:
        return "white"
    elif brightness < 90:  # widened threshold for black (was 60)
        return "black"
    else:
        return "other"


# ------------------------------------
# Load Model
# ------------------------------------
print("🔄 Loading DeepFashion2 YOLO model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ------------------------------------
# Get All Images in Folder
# ------------------------------------
image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
              glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))

if not image_files:
    raise FileNotFoundError(f"❌ No image files found in {IMAGE_FOLDER}")

print(f"🖼️ Found {len(image_files)} image(s) for detection.\n")

# ------------------------------------
# Run Inference + Color Check
# ------------------------------------
csv_data = [["Image", "Label", "Detected_Color", "Color_%", "Confidence", "Dress_Code_Status"]]

for img_path in image_files:
    print(f"📸 Processing: {os.path.basename(img_path)}")
    img = cv2.imread(img_path)

    results = model.predict(img_path, conf=0.25, save=True, show=False)
    r = results[0]
    names = r.names

    shirt_color = None
    pant_color = None

    if not len(r.boxes):
        print("   ⚠️ No detections found.\n")
        csv_data.append([os.path.basename(img_path), "-", "-", "-", "-", "Dress Code Violation (No detections)"])
        continue

    print("   🧩 Detected Objects:")
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        label = names[int(cls)]
        confidence = float(conf)

        # --- Get dominant color + percentage
        color_rgb, color_percent = get_dominant_color_with_percentage(img, box)
        color_name = get_color_name(color_rgb)

        print(f"     - {label} ({confidence:.2f}) → Color: {color_name} ({color_percent:.1f}%)")

        # --- Assign shirt and pant colors
        if any(k in label.lower() for k in ["shirt", "top", "t-shirt", "tee"]):
            shirt_color = color_name
        elif any(k in label.lower() for k in ["pant", "trouser", "jean"]):
            pant_color = color_name

        csv_data.append([
            os.path.basename(img_path),
            label,
            color_name,
            f"{color_percent:.1f}%",
            confidence,
            ""
        ])

    # ✅ Final Dress Code Logic (revised)
    if (shirt_color in ["white", "black"]) and (pant_color == "black"):
        status = "Dress Code Passed ✅"
    else:
        status = "Dress Code Violation ❌"

    print(f"🎯 Result: {status}\n")
    csv_data.append(["-", "-", "-", "-", "-", status])

print("✅ All images processed.\n")

# ------------------------------------
# Save Detection Summary to CSV
# ------------------------------------
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"📁 Detection summary saved to: {OUTPUT_CSV}")
print(f"📂 Visual results saved under: {results[0].save_dir}")
