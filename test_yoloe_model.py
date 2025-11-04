from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import csv

# ------------------------------------
# Paths
# ------------------------------------
MODEL_PATH = "/home/devi-1202324/Azure/pazofunc/models/deepfashion2_yolov8s-seg.pt"
IMAGE_FOLDER = "/home/devi-1202324/Downloads/OneDrive_2_10-31-2025"
OUTPUT_CSV = "/home/devi-1202324/Azure/pazofunc/deepfashion2_dresscode.csv"

# ------------------------------------
# Helper: Get Dominant Color
# ------------------------------------
def get_dominant_color(image, box):
    """Extract dominant color (BGR) inside bounding box."""
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return (0, 0, 0)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(cropped.reshape(-1, 3), axis=0)
    return avg_color  # [R, G, B]

def get_color_name(rgb):
    """Classify a color as white, black, or other."""
    r, g, b = rgb
    brightness = np.mean([r, g, b])
    if brightness > 180 and abs(r-g) < 30 and abs(r-b) < 30:
        return "white"
    elif brightness < 60:
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
csv_data = [["Image", "Label", "Detected_Color", "Confidence", "Dress_Code_Status"]]

for img_path in image_files:
    print(f"📸 Processing: {os.path.basename(img_path)}")
    img = cv2.imread(img_path)

    results = model.predict(img_path, conf=0.25, save=True, show=False)
    r = results[0]
    names = r.names

    # Track shirt/pant status
    shirt_color = None
    pant_color = None

    if not len(r.boxes):
        print("   ⚠️ No detections found.\n")
        csv_data.append([os.path.basename(img_path), "-", "-", "-", "Dress Code Violation (No detections)"])
        continue

    print("   🧩 Detected Objects:")
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        label = names[int(cls)]
        confidence = float(conf)

        # Get dominant color inside bounding box
        color_rgb = get_dominant_color(img, box)
        color_name = get_color_name(color_rgb)

        print(f"     - {label} ({confidence:.2f}) → Color: {color_name}")

        # Store shirt/pant colors for validation
        if "shirt" in label.lower() or "top" in label.lower():
            shirt_color = color_name
        elif "pant" in label.lower() or "trouser" in label.lower() or "jean" in label.lower():
            pant_color = color_name

        csv_data.append([os.path.basename(img_path), label, color_name, confidence, ""])

    # -------------------------------
    # Dress Code Validation
    # -------------------------------
    status = "Dress Code Violation"
    if shirt_color in ["white", "black"] and pant_color == "black":
        status = "Dress Code Passed"

    print(f"🎯 Result: {status}\n")
    csv_data.append(["-", "-", "-", "-", status])

print("✅ All images processed.\n")

# ------------------------------------
# Save Detection Summary to CSV
# ------------------------------------
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"📁 Detection summary saved to: {OUTPUT_CSV}")
print(f"📂 Visual results saved under: {results[0].save_dir}")
