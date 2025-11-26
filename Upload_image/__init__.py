import logging
import azure.functions as func
import json
import requests
import os
from datetime import datetime
from .adls_utils import upload_json_to_adls

# ===============================================
# OLLAMA URL
# ===============================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# ===============================================
# STRICT JSON RULE
# ===============================================
JSON_RULE = """
You MUST respond ONLY in valid JSON.
Do NOT include explanations or extra text.
Format:
{
  "status": "pass or fail",
  "summary": "short summary",
  "score": "number or N/A",
  "details": {
    "issues_found": [],
    "comments": ""
  }
}
"""

# ===============================================
# MAIN HTTP TRIGGER
# ===============================================
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function received request.")

    try:
        # 1. Read category
        category = req.params.get("category")
        if not category:
            return func.HttpResponse(
                json.dumps({"error": "Missing category"}),
                mimetype="application/json",
                status_code=400
            )

        # 2. Read store_id
        store_id = req.params.get("store_id")
        if not store_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing store_id"}),
                mimetype="application/json",
                status_code=400
            )

        # 3. Read file
        file = req.files.get("file")
        if not file:
            return func.HttpResponse(
                json.dumps({"error": "Missing file upload"}),
                mimetype="application/json",
                status_code=400
            )

        # Save file locally
        local_path = f"/tmp/{file.filename}"
        with open(local_path, "wb") as f:
            f.write(file.read())

        logging.info(f"Saved uploaded file: {local_path}")

        # Prompt prep
        prompt = build_prompt(category)
        ai_result = run_moondream_inference(prompt, local_path)

        # Build final payload
        final_payload = {
            "filename": file.filename,
            "store_id": store_id,
            "category": category,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "result": ai_result
        }

        # 4. Upload JSON results to ADLS
        adls_path = upload_json_to_adls(store_id, category, final_payload)

        final_payload["adls_path"] = adls_path

        return func.HttpResponse(
            json.dumps(final_payload),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception("Crash inside function")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


# ===============================================
# CATEGORY PROMPTS
# ===============================================
def build_prompt(category):

    prompts = {
        "dresscode": """
        Analyze employee dress code:
        - Shirt black/white?
        - Pants black?
        - Shoes present?
        - Beard present?
        List violations.
        Return JSON only.
        """,

        "dustbin": """
        Analyze dustbin cleanliness:
        - Visible?
        - Clean or messy?
        - Poly cover present?
        - Overflowing?
        Return JSON only.
        """,

        "lightscheck": """
        Analyze lights:
        - Lights ON/OFF
        - Dim or faulty lights
        Return JSON only.
        """,

        "floorcheck": """
        Floor analysis:
        - Dust, hair, stains
        - Wet/dry
        Give cleanliness rating.
        Return JSON only.
        """,

        "nailpolishtray": """
        Nail polish tray:
        - Bottles arranged?
        - Caps missing?
        - Spills?
        Return JSON only.
        """,

        "shampoobottles": """
        Shampoo bottle rack:
        - Arrangement
        - Messy surroundings
        - Spills
        Return JSON only.
        """,

        "restroomcheck": """
        Restroom analysis:
        - Toilet clean?
        - Basin clean?
        - Stains/hair?
        - Handwash?
        - Freshener?
        Return JSON only.
        """
    }

    return prompts.get(category, "General analysis. Return JSON only.")


# ===============================================
# MOONDREAM INFERENCE
# ===============================================
def run_moondream_inference(prompt, image_path):

    final_prompt = f"""
{prompt}

Image Path: {image_path}

{JSON_RULE}
"""

    payload = {
        "model": "moondream:latest",
        "prompt": final_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
    except Exception as e:
        return {"error": "Ollama unreachable", "details": str(e)}

    try:
        return response.json()
    except:
        try:
            return json.loads(response.text)
        except:
            return {"error": "Invalid JSON from model", "raw": response.text}
