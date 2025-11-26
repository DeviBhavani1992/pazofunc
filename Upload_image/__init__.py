import logging
import azure.functions as func
import json
import requests
import os

# ===============================================
#  OLLAMA ENDPOINT (supports VM + local testing)
# ===============================================
# If running INSIDE Azure VM → localhost works
# If running from LOCAL laptop → set env OLLAMA_URL=http://104.211.66.125:21434/api/generate
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# ===============================================
# STRICT JSON RULE ENFORCEMENT
# ===============================================
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

# ===============================================
# MAIN FUNCTION — ENTRY POINT
# ===============================================
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function received a request.")

    try:
        # 1. Validate category
        category = req.params.get("category")
        if not category:
            return func.HttpResponse(
                json.dumps({"error": "Missing ?category="}),
                mimetype="application/json",
                status_code=400
            )

        # 2. Validate file
        file = req.files.get("file")
        if not file:
            return func.HttpResponse(
                json.dumps({"error": "Missing file upload"}),
                mimetype="application/json",
                status_code=400
            )

        # 3. Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.read())

        logging.info(f"Saved uploaded file at {file_path}")

        # 4. Generate the correct prompt
        prompt = build_prompt(category)

        # 5. Run AI model (Moondream via Ollama)
        ai_result = run_moondream_inference(prompt, file_path)

        # 6. Build final response for frontend
        final_payload = {
            "filename": file.filename,
            "category": category,
            "status": "success",
            "result": ai_result
        }

        return func.HttpResponse(
            json.dumps(final_payload),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception("Function crashed unexpectedly")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


# ===============================================
# CATEGORY PROMPTS (ALL 7 CATEGORIES)
# ===============================================
def build_prompt(category):

    prompts = {
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

    return prompts.get(category, "General analysis. Return JSON only.")


# ===============================================
#  MOONDREAM INFERENCE (OLLAMA CALL)
# ===============================================
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
    except Exception as e:
        return {
            "error": "Unable to reach OLLAMA server",
            "url": OLLAMA_URL,
            "details": str(e)
        }

    # Try parsing clean JSON
    try:
        return response.json()
    except:
        pass

    # Try fallback parsing
    try:
        return json.loads(response.text)
    except:
        return {"error": "Model returned invalid JSON", "raw_output": response.text}
