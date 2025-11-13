import logging
import azure.functions as func
import requests
import os
import json
import tempfile
import traceback
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dustbin_detection")

# Load Dustbin model from models directory (same layout as dresscode function)
try:
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "dustbin_yolo11_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Dustbin model not found at {model_path}")
    dustbin_model = YOLO(model_path)
    logger.info(f"Dustbin model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load dustbin model: {e}")
    dustbin_model = None


def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("Dustbin detection function triggered")

    try:
        # Parse incoming JSON body (expecting {"blob_url": "..."} )
        try:
            data = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON body"}),
                status_code=400,
                mimetype="application/json"
            )

        blob_url = data.get("blob_url")
        if not blob_url:
            return func.HttpResponse(
                json.dumps({"error": "Missing blob_url"}),
                status_code=400,
                mimetype="application/json"
            )

        if not dustbin_model:
            return func.HttpResponse(
                json.dumps({"error": "Dustbin model not available on server"}),
                status_code=500,
                mimetype="application/json"
            )

        # Download the image (with simple error handling)
        resp = requests.get(blob_url, timeout=20)
        if resp.status_code != 200:
            return func.HttpResponse(
                json.dumps({"error": f"Failed to download image from {blob_url} (status {resp.status_code})"}),
                status_code=400,
                mimetype="application/json"
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(resp.content)
            image_path = tmp.name

        logger.info(f"Image downloaded ({len(resp.content)} bytes) -> {image_path}")

        # Run YOLO inference
        try:
            model_results = dustbin_model.predict(image_path, conf=0.25)
            r = model_results[0]
            names = r.names
        except Exception as ex:
            logger.error(f"Inference error: {ex}")
            # cleanup
            try:
                os.unlink(image_path)
            except Exception:
                pass
            return func.HttpResponse(
                json.dumps({"error": f"Inference failed: {str(ex)}"}),
                status_code=500,
                mimetype="application/json"
            )

        detections = []
        if len(r.boxes):
            # r.boxes.xyxy, r.boxes.cls, r.boxes.conf iterate together
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                detections.append({
                    "label": names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [float(x) for x in box.tolist()]
                })

        # Build response
        if detections:
            status = "dustbin_found"
            message = f"Detected {len(detections)} dustbin(s)."
        else:
            status = "no_dustbin"
            message = "No dustbin detected in the image."

        result = {
            "status": status,
            "message": message,
            "detections": detections,
            "blob_url": blob_url
        }

        # Cleanup temp file
        try:
            os.unlink(image_path)
        except Exception:
            pass

        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as ex:
        logger.error(f"Unhandled error in dustbin_detection: {ex}")
        logger.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(ex)}),
            status_code=500,
            mimetype="application/json"
        )
