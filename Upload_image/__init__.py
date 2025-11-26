import logging
import azure.functions as func
import json
import os
import requests
from datetime import datetime
from adls_utils import upload_json_to_adls

# ==============================
# OLLAMA ENDPOINT
# ==============================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# ==============================
# STRICT JSON RULE ENFORCEMENT
# ==============================
JSON_RULE = """
You MUST respond ONLY in valid JSON.
Do NOT include explanations, markdown, or extra text.
Output must match this format:
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

# ==============================
# CATEGORY PROMPTS
# ==============================
CATEGORY_PROMPTS = {
    "dresscode": """
Analyze employee dress code from image:
- Shirt must be black or white
- Pants must be black
- Shoes must be present
- Beard should not be present
List violations and give rating.
Return JSON only.
""",
    "dustbin": """
Analyze dustbin:
- Is dustbin visible?
- Clean or untidy?
- Poly cover present?
- Overflowing or OK?
Return JSON only.
""",
    "lightscheck": """
Analyze lighting in the room:
- Which lights are ON?
- Which lights are OFF?
- Any dim or faulty lights?
Return JSON only.
""",
    "floorcheck": """
Analyze floor cleanliness:
- Hair, dust, stains, spills, marks
- Is the floor dry and clean?
Give a cleanliness rating.
Return JSON only.
""",
    "nailpolishtray": """
Analyze nail polish tray:
- Are bottles arranged neatly?
- Any bottles missing caps?
- Any spills or stains?
Return JSON only.
""",
    "shampoobottles": """
Analyze shampoo bottle arrangement:
- Are bottles arranged properly?
- Any messy surroundings?
- Any spills or stains?
Return JSON only.
""",
    "restroomcheck": """
Analyze restroom:
- Is toilet clean?
- Is basin clean?
- Any stains or hair?
- Handwash available?
- Room freshener available?
Give rating.
Return JSON only.
"""
}


# ==============================
# MAIN FUNCTION
# ==============================
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function received a request.")

    try:
        category = req.params.get("category")
        store_id = req.params.get("store_id")

        if not category:
            return func.HttpResponse(
                json.dumps({"error": "Missing ?category="}),
                mimetype="application/json",
                status_code=400
            )

        if not store_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing ?store_id="}),
                mimetype="application/json",
                status_code=400
            )

        file = req.files.get("file")
        if not file:
            return func.HttpResponse(
                json.dumps({"error": "Missing file upload"}),
                mimetype="application/json",
                status_code=400
            )

        # Save temp file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(file.read())

        logging.info(f"Saved uploaded file at {tmp_path}")

        # Build prompt
        prompt = CATEGORY_PROMPTS.get(category, "General analysis. Return JSON only.")
        ai_result = run_moondream_inference(prompt, tmp_path)

        # Upload to ADLS
        adls_path = upload_json_to_adls(
            data=ai_result,
            category=category,
            timestamp=timestamp,
            store_id=store_id
        )

        response_payload = {
            "filename": file.filename,
            "category": category,
            "status": "success" if "error" not in ai_result else "fail",
            "adls_path": adls_path,
            "result": ai_result
        }

        return func.HttpResponse(
            json.dumps(response_payload),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception("Function failed")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


# ==============================
# AI INFERENCE
# ==============================
def run_moondream_inference(prompt, image_path):
    complete_prompt = f"""
{prompt}

Image File Path: {image_path}

{JSON_RULE}
"""
    payload = {
        "model": "moondream:latest",
        "prompt": complete_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json()
    except Exception as e:
        return {"error": "Unable to reach OLLAMA server", "details": str(e)}
