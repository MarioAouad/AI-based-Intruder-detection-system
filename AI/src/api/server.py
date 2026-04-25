import base64
import datetime
import glob
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from src.database.db_manager import delete_person

load_dotenv()

app = FastAPI()

AI_API_KEY = os.getenv("AI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY")


# ── Models ────────────────────────────────────────────────────────────────────

class PhotoItem(BaseModel):
    type: str   # "face", "left_profile", "right_profile"
    data: str   # base64-encoded image


class RegisterPersonRequest(BaseModel):
    person_id: int
    property_id: int
    photos: list[PhotoItem]


# ── Receive from backend: register a person ───────────────────────────────────

@app.post("/persons/register")
def register_person(payload: RegisterPersonRequest, x_api_key: str = Header(...)):
    if x_api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Define where to drop the images
    TARGETS_DIR = os.path.join("data", "captured_targets")
    os.makedirs(TARGETS_DIR, exist_ok=True)

    # Clean the queue: Delete any pending processing files for this specific person AND property
    existing = glob.glob(os.path.join(TARGETS_DIR, f"owner_{payload.property_id}_{payload.person_id}_*.jpg"))
    for f in existing:
        os.remove(f)

    # Clean the database: Wipe their old 512-D vectors to ensure a fresh update
    delete_person(payload.property_id, payload.person_id)

    decoded_photos = {}
    for photo in payload.photos:
        # Decode base64 to binary image data
        image_bytes = base64.b64decode(photo.data)
        decoded_photos[photo.type] = image_bytes
        
        # CREATE THE FILENAME: "owner_{property_id}_{person_id}_{photo_type}.jpg"
        # Example: "owner_42_105_left_profile.jpg"
        filename = f"owner_{payload.property_id}_{payload.person_id}_{photo.type}.jpg"
        filepath = os.path.join(TARGETS_DIR, filename)
        
        # Save the image to captured_targets/ so face_processor.py can see it
        with open(filepath, "wb") as f:
            f.write(image_bytes)

    print(f"Registered/Updated person {payload.person_id} at property {payload.property_id}. Dropped {len(decoded_photos)} images to {TARGETS_DIR}.")
    return {"status": "registered", "person_id": payload.person_id}

# ── Receive from backend: deregister a person ────────────────────────────────

@app.delete("/properties/{property_id}/persons/{person_id}")
def deregister_person(property_id: int, person_id: int, x_api_key: str = Header(...)):
    if x_api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Tell the database manager to wipe this person from this specific property
    delete_person(property_id, person_id)

    print(f"Deregistered person {person_id} from property {property_id}. Removed from DB.")
    return {
        "status": "deregistered", 
        "property_id": property_id, 
        "person_id": person_id
    }


# ── Send to backend: report a detection ───────────────────────────────────────

def send_detection_event(
    property_id: int,
    similarity_score: float,       # 0.0 to 100.0
    snapshot_path: str,            # path to snapshot image file on disk
    person_id: int | None = None,  # None if nobody was recognized
):
    with open(snapshot_path, "rb") as f:
        snapshot_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "property_id": property_id,
        "similarity_score": similarity_score,
        "snapshot_base64": snapshot_base64,
        "person_id": person_id,
        "occurred_at": datetime.datetime.utcnow().isoformat(),
    }

    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/webhooks/intruder",
            json=payload,
            headers={"X-Webhook-Api-Key": WEBHOOK_API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        print(f"Detection recorded: event_id={data['event_id']}, status={data['status']}")
        return data
    except httpx.HTTPStatusError as e:
        print(f"Backend rejected detection: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"Failed to send detection: {e}")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

