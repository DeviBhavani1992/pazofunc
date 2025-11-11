# -*- coding: utf-8 -*-
"""
Fashionpedia YOLOv11 Training Script (Local GPU Version)
Cleaned and adapted for /opt/ working directory
"""

import os
from glob import glob
import yaml
import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import kagglehub

# ==========================================================
# 🧱 CONFIG
# ==========================================================
HOME = "/opt"
DATA_DIR = f"{HOME}/datasets/data/fashionpedia"
YAML_PATH = f"{DATA_DIR}/data.yaml"

# ==========================================================
# 📦 STEP 1: Install & Download Data
# ==========================================================
os.system("pip install -q ultralytics kagglehub torchvision pyyaml matplotlib pillow opencv-python")

print("✅ Libraries installed successfully.")

# Download dataset from KaggleHub
path = kagglehub.dataset_download("pranaysanam/fashionpedia")
print("📂 Dataset downloaded to:", path)

# ==========================================================
# 🗂️ STEP 2: Copy Dataset to /opt/datasets/data/fashionpedia
# ==========================================================
def make_ds(root, path_to_copy):
    os.makedirs(path_to_copy, exist_ok=True)
    files = glob(f"{root}/*")
    for file in files:
        if os.path.isdir(file):
            os.system(f"cp -r '{file}' '{path_to_copy}'")
        elif os.path.isfile(file):
            os.system(f"cp '{file}' '{path_to_copy}'")

root = path  # from kagglehub
make_ds(root=root, path_to_copy=DATA_DIR)

print("✅ Dataset copied to:", DATA_DIR)
os.system(f"ls -ltr {DATA_DIR}")

# ==========================================================
# 🧾 STEP 3: Fix data.yaml paths (absolute paths)
# ==========================================================
fixed_yaml = f"""
train: {DATA_DIR}/images/train
val: {DATA_DIR}/images/val
test: {DATA_DIR}/images/test

nc: 46
names:
  0: "shirt, blouse"
  1: "top, t-shirt, sweatshirt"
  2: "sweater"
  3: "cardigan"
  4: "jacket"
  5: "vest"
  6: "pants"
  7: "shorts"
  8: "skirt"
  9: "coat"
  10: "dress"
  11: "jumpsuit"
  12: "cape"
  13: "glasses"
  14: "hat"
  15: "headband, head covering, hair accessory"
  16: "tie"
  17: "glove"
  18: "watch"
  19: "belt"
  20: "leg warmer"
  21: "tights, stockings"
  22: "sock"
  23: "shoe"
  24: "bag, wallet"
  25: "scarf"
  26: "umbrella"
  27: "hood"
  28: "collar"
  29: "lapel"
  30: "epaulette"
  31: "sleeve"
  32: "pocket"
  33: "neckline"
  34: "buckle"
  35: "zipper"
  36: "applique"
  37: "bead"
  38: "bow"
  39: "flower"
  40: "fringe"
  41: "ribbon"
  42: "rivet"
  43: "ruffle"
  44: "sequin"
  45: "tassel"
"""

with open(YAML_PATH, "w") as f:
    f.write(fixed_yaml)
print(f"✅ YAML file updated: {YAML_PATH}")

# ==========================================================
# 🎨 STEP 4: Visualization Helper
# ==========================================================
class Visualization:
    def __init__(self, data_types, n_ims, rows, cmap="rgb"):
        self.data_types = data_types
        self.n_ims = n_ims
        self.rows = rows
        self.cmap = cmap
        self.colors = ["firebrick", "darkorange", "blueviolet"]
        self.get_cls_names()
        self.get_bboxes()

    def get_cls_names(self):
        with open(YAML_PATH, "r") as file:
            data = yaml.safe_load(file)
        self.class_dict = {i: n for i, n in enumerate(data["names"].values())}

    def get_bboxes(self):
        self.vis_datas, self.analysis_datas, self.im_paths = {}, {}, {}
        for data_type in self.data_types:
            all_bboxes, all_analysis_datas = [], {}
            im_paths = glob(f"{DATA_DIR}/images/{data_type}/*")
            for im_path in im_paths:
                lbl_path = im_path.replace("images", "labels").replace(".jpg", ".txt")
                if not os.path.isfile(lbl_path):
                    continue
                with open(lbl_path) as f:
                    lines = f.readlines()
                bboxes = []
                for data in lines:
                    parts = data.strip().split()[:5]
                    cls_name = self.class_dict[int(parts[0])]
                    bboxes.append([cls_name] + [float(x) for x in parts[1:]])
                    all_analysis_datas[cls_name] = all_analysis_datas.get(cls_name, 0) + 1
                all_bboxes.append(bboxes)
            self.vis_datas[data_type] = all_bboxes
            self.analysis_datas[data_type] = all_analysis_datas
            self.im_paths[data_type] = im_paths

    def plot(self, rows, cols, count, im_path, bboxes):
        plt.subplot(rows, cols, count)
        or_im = np.array(Image.open(im_path).convert("RGB"))
        height, width, _ = or_im.shape
        for bbox in bboxes:
            class_id, x_center, y_center, w, h = bbox
            x_min = int((x_center - w / 2) * width)
            y_min = int((y_center - h / 2) * height)
            x_max = int((x_center + w / 2) * width)
            y_max = int((y_center + h / 2) * height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(or_im, (x_min, y_min), (x_max, y_max), color, 2)
        plt.imshow(or_im)
        plt.axis("off")
        plt.title(f"{len(bboxes)} object(s)")
        return count + 1

    def vis(self, save_name):
        plt.figure(figsize=(25, 20))
        cols = self.n_ims // self.rows
        indices = random.sample(range(len(self.vis_datas[save_name])), min(self.n_ims, len(self.vis_datas[save_name])))
        count = 1
        for i in indices:
            im_path, bboxes = self.im_paths[save_name][i], self.vis_datas[save_name][i]
            count = self.plot(self.rows, cols, count, im_path, bboxes)
        plt.show()

    def data_analysis(self, save_name, color):
        cls_names = list(self.analysis_datas[save_name].keys())
        counts = list(self.analysis_datas[save_name].values())
        _, ax = plt.subplots(figsize=(30, 10))
        ax.bar(cls_names, counts, color=color)
        ax.set_title(f"{save_name.upper()} Class Distribution")
        plt.xticks(rotation=90)
        plt.show()

    def run(self):
        for i, name in enumerate(self.data_types):
            self.data_analysis(name, self.colors[i])
            self.vis(name)

# ==========================================================
# 📊 STEP 5: Run Visualization
# ==========================================================
vis = Visualization(data_types=["train", "val", "test"], n_ims=12, rows=4, cmap="rgb")
vis.run()

# ==========================================================
# 🚀 STEP 6: Train YOLOv11
# ==========================================================
model = YOLO("yolo11n.pt")

train_results = model.train(
    data=YAML_PATH,
    epochs=50,
    imgsz=480,
    device=0,  # GPU
    project="/opt/runs/fashionpedia",  # Custom output dir
    name="yolo11_fashion",             # Experiment name
    save=True,
    save_period=1,                     # Optional: save every epoch
)

print("✅ Training complete. Results saved in /opt/runs/fashionpedia/yolo11_fashion")

# ==========================================================
# 💾 STEP 7: Save Best Model to a Known Location
# ==========================================================
best_model_path = os.path.join("/opt/runs/fashionpedia/yolo11_fashion", "weights", "best.pt")
final_save_path = "/opt/models/fashionpedia_yolo11_best.pt"

os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

if os.path.exists(best_model_path):
    os.system(f"cp '{best_model_path}' '{final_save_path}'")
    print(f"✅ Best model saved at: {final_save_path}")
else:
    print("⚠️ Could not find best.pt — training may not have completed successfully.")
