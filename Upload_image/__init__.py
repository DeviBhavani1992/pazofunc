import logging
import azure.functions as func
import requests
import json
import os

# Get environment variables
GEMMA_ENDPOINT = os.getenv("GEMMA_ENDPOINT")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

def run_gemma_inference(image_url, prompt):
    try:
        logging.info("Preparing payload for Gemma inference...")
        payload = {
            "model": "gemma2:27b",
            "prompt": f"{prompt}\n\nImage URL: {image_url}",
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
        }
        if GEMMA_API_KEY:
            headers["Authorization"] = f"Bearer {GEMMA_API_KEY}"

        logging.info(f"Sending request to Gemma endpoint: {GEMMA_ENDPOINT}/api/generate")
        logging.info(f"Payload snippet: {json.dumps(payload)[:300]} ...")

        resp = requests.post(
            f"{GEMMA_ENDPOINT}/api/generate",
            headers=headers,
            data=json.dumps(payload),
            timeout=240
        )

        logging.info(f"Gemma status code: {resp.status_code}")
        logging.info(f"Gemma response snippet: {resp.text[:300]} ...")

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        logging.error("Gemma call TIMED OUT")
        return {"error": "timeout"}

    except Exception as e:
        logging.error(f"Gemma call failed: {e}")
        return {"error": str(e)}


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        category = req.params.get('category')
        if not category:
            return func.HttpResponse(
                "Missing 'category' query parameter.",
                status_code=400
            )

        # Assuming file is uploaded in 'file' field
        file = req.files.get('file')
        if not file:
            return func.HttpResponse(
                "No file uploaded in 'file' field.",
                status_code=400
            )

        # Save file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.read())

        logging.info(f"Received file: {file.filename}, category: {category}")

        # Choose default prompts based on category
        prompts = {
            "dresscode": "from this image you have to validate the dress code "
                         "shirt: black or white, pant: black, shoe: must be there "
                         "colour optional, beards: No. If criteria not met, say dress code is inappropriate "
                         "and give rating score based on the criteria.",
            "dustbin": "check whether there is a dustbin in a image or not. "
                       "If dustbin is there validate it is clean or not clean "
                       "and give score based on image. Is dustbin available with poly cover inside and not overfilled?",
            "lightscheck": "you have to check all the lights are on or not. If on, give the score."
        }
        prompt = prompts.get(category, "")

        # Build image URL if your blob storage is used, or pass local path
        image_url = f"{file_path}"

        gemma_result = run_gemma_inference(image_url, prompt)

        response_payload = {
            "filename": file.filename,
            "category": category,
            "status": "success" if "error" not in gemma_result else "error",
            "result": gemma_result
        }

        return func.HttpResponse(json.dumps(response_payload), mimetype="application/json")

    except Exception as e:
        logging.exception("Function execution failed")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
