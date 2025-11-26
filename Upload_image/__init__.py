import logging
import azure.functions as func
import json
import os
import requests
import base64
import tempfile
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
        # Validate environment variables
        required_env_vars = ["ADLS_CONNECTION_STRING", "OLLAMA_URL"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
            logging.error(error_msg)
            return func.HttpResponse(
                json.dumps({"filename": "", "category": "", "status": "error", "message": error_msg}),
                mimetype="application/json",
                status_code=500
            )

        category = req.params.get("category")
        store_id = req.params.get("store_id")

        if not category:
            return func.HttpResponse(
                json.dumps({"filename": "", "category": "", "status": "error", "message": "Missing ?category="}),
                mimetype="application/json",
                status_code=400
            )

        if not store_id:
            return func.HttpResponse(
                json.dumps({"filename": "", "category": category, "status": "error", "message": "Missing ?store_id="}),
                mimetype="application/json",
                status_code=400
            )

        file = req.files.get("file")
        if not file:
            return func.HttpResponse(
                json.dumps({"filename": "", "category": category, "status": "error", "message": "Missing file upload"}),
                mimetype="application/json",
                status_code=400
            )

        logging.info(f"Processing file: {file.filename}, category: {category}, store_id: {store_id}")

        # Read file content
        file_content = file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build prompt and run AI inference
        prompt = CATEGORY_PROMPTS.get(category, "General analysis. Return JSON only.")
        ai_result = run_moondream_inference(prompt, file_content)
        
        logging.info(f"AI inference result: {ai_result}")

        # Check if AI inference was successful
        if "error" in ai_result:
            return func.HttpResponse(
                json.dumps({
                    "filename": file.filename,
                    "category": category,
                    "status": "error",
                    "message": ai_result.get("details", "AI inference failed")
                }),
                mimetype="application/json",
                status_code=200
            )

        # Upload to ADLS
        try:
            adls_path = upload_json_to_adls(
                data=ai_result,
                category=category,
                timestamp=timestamp,
                store_id=store_id
            )
            logging.info(f"Successfully uploaded to ADLS: {adls_path}")
        except Exception as adls_error:
            logging.error(f"ADLS upload failed: {str(adls_error)}")
            return func.HttpResponse(
                json.dumps({
                    "filename": file.filename,
                    "category": category,
                    "status": "error",
                    "message": f"ADLS upload failed: {str(adls_error)}"
                }),
                mimetype="application/json",
                status_code=200
            )

        response_payload = {
            "filename": file.filename,
            "category": category,
            "status": "success",
            "message": "Analysis completed successfully",
            "adls_path": adls_path,
            "result": ai_result
        }

        return func.HttpResponse(
            json.dumps(response_payload),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        error_msg = f"Function failed: {str(e)}"
        logging.exception(error_msg)
        return func.HttpResponse(
            json.dumps({
                "filename": getattr(file, 'filename', '') if 'file' in locals() else "",
                "category": category if 'category' in locals() else "",
                "status": "error",
                "message": error_msg
            }),
            mimetype="application/json",
            status_code=500
        )


# ==============================
# AI INFERENCE
# ==============================
def run_moondream_inference(prompt, image_content):
    try:
        # Encode image to base64
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        complete_prompt = f"""
{prompt}

{JSON_RULE}
"""
        
        payload = {
            "model": "moondream:latest",
            "prompt": complete_prompt,
            "images": [image_base64],
            "stream": False
        }

        logging.info(f"Sending request to OLLAMA: {OLLAMA_URL}")
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        
        if response.status_code != 200:
            logging.error(f"OLLAMA API error: {response.status_code} - {response.text}")
            return {"error": "OLLAMA API error", "details": f"Status: {response.status_code}, Response: {response.text}"}
        
        result = response.json()
        logging.info(f"OLLAMA response: {result}")
        
        # Extract the response text from OLLAMA's response format
        if "response" in result:
            try:
                # Try to parse the response as JSON
                ai_response = json.loads(result["response"])
                return ai_response
            except json.JSONDecodeError:
                # If not valid JSON, return the raw response
                return {"error": "Invalid JSON response from AI", "details": result["response"]}
        else:
            return {"error": "Unexpected response format from OLLAMA", "details": str(result)}
            
    except requests.exceptions.Timeout:
        return {"error": "OLLAMA request timeout", "details": "Request took longer than 60 seconds"}
    except requests.exceptions.ConnectionError:
        return {"error": "Unable to connect to OLLAMA server", "details": f"Connection failed to {OLLAMA_URL}"}
    except Exception as e:
        logging.exception("OLLAMA inference failed")
        return {"error": "OLLAMA inference failed", "details": str(e)}
