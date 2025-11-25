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
            "model": "gemma:2b",
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

        # All prompts including NEW restroom check
        prompts = {
            "dresscode": (
                "From this image validate the dress code: "
                "shirt: black or white, pant: black, shoe: must be there (colour optional), "
                "beards: No. If criteria not met, say dress code is inappropriate "
                "and give rating score based on the criteria."
            ),

            "dustbin": (
                "Check whether there is a dustbin in the image. "
                "If dustbin is present validate if it is clean or not clean "
                "and give a score. Check for poly cover inside and if it is not overfilled."
            ),

            "lightscheck": (
                "Check whether all the lights in the image are ON or OFF. "
                "If ON, give the score. If not, mention which lights are off."
            ),

            "floorcheck": (
                "From the image, check whether the floor is clean. "
                "Verify if the floor is free from hair, stains, dust, marks or spill areas. "
                "If you find any stains or hair, list them clearly. "
                "Give an overall cleanliness rating from 1 to 10."
            ),

            "nailpolishtray": (
                "Check whether all nail polish bottles are placed properly inside a plastic box or tray. "
                "If arranged well, give a rating. If not arranged properly, describe what you see clearly."
            ),

            "shampoobottles": (
                "Check whether shampoo bottles are placed neatly. "
                "Ensure there are no stains, spill overs, or messy surroundings. "
                "Give a rating for arrangement and cleanliness. "
                "If not neat, explain what you observe and provide improvement suggestions."
            ),

            "restroomcheck": (
                "Check whether the rest room is clean or not and free from stains and hair. "
                "Verify if the wash basin is clean, if handwash is available, and "
                "if a room freshener is present. List any missing items. "
                "Finally give an overall rating based on all criteria."
            ),
        }

        prompt = prompts.get(category, "")

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
