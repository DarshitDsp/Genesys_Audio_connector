from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import time
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

@app.post("/postUtterance")
async def post_utterance(request: Request):
    start_time = time.time()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Extract fields
    utterance = body.get("utterance", "")
    session_id = body.get("sessionId", "")
    session_vars = body.get("sessionVars", {})

    member_id = session_vars.get("memberId")
    auth_state = session_vars.get("authState")
    current_page = session_vars.get("currentPage")
    source_nudge_id = session_vars.get("sourceNudgeId")
    pre_seeded_question = session_vars.get("preSeededQuestion")

    # Log request
    request_size = len(str(body).encode("utf-8"))
    logging.info(f"Incoming Request: {body}")
    logging.info(f"Request Size: {request_size} bytes")

    # Business logic (POC)
    text = utterance.lower()

    if "agent" in text:
        response = {
            "responseType": "TRANSFER_TO_AGENT",
            "escalationReason": "User requested human agent"
        }

    elif "bye" in text:
        response = {
            "responseType": "CONVERSATION_COMPLETE",
            "text": "Thank you! Goodbye."
        }

    else:
        response = {
            "responseType": "MORE_DATA",
            "text": f"You said: {utterance}",
            "debugContext": {
                "sessionId": session_id,
                "memberId": member_id,
                "authState": auth_state,
                "currentPage": current_page,
                "sourceNudgeId": source_nudge_id,
                "preSeededQuestion": pre_seeded_question
            }
        }

    # Log response
    response_size = len(str(response).encode("utf-8"))
    latency = time.time() - start_time

    logging.info(f"Response: {response}")
    logging.info(f"Response Size: {response_size} bytes")
    logging.info(f"Latency: {latency:.3f} sec")

    return JSONResponse(content=response)


# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "API is running"}
