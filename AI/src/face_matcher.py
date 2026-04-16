import os
import sys
import glob
import time
import json
import base64
import sqlite3
import datetime
import numpy as np
import httpx
from dotenv import load_dotenv

# 1. Setup & Pathing
load_dotenv()

WEBHOOK_API_KEY = os.environ.get("WEBHOOK_API_KEY", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGNED_DIR = os.path.join(BASE_DIR, "data", "faces_aligned")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "alert_snapshots")
DB_PATH = os.path.join(BASE_DIR, "data", "faces.db")

# Append src/ to access database.embedding_utils
sys.path.append(os.path.join(BASE_DIR, "src"))
from database.embedding_utils import get_embedding

COSINE_THRESHOLD = 0.65


# 2. Helper Functions
def compute_cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    """Calculate the cosine distance between two 512-D lists."""
    a = np.array(vec1)
    b = np.array(vec2)
    # Cosine distance = 1 - Cosine Similarity
    return float(1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

def fetch_property_owners(property_id: int):
    """Fetch property owner rows from SQLite and parse JSON embeddings back to lists."""
    results = []
    if not os.path.exists(DB_PATH):
        return results

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Select rows filtering by property_id
    cursor.execute('''
        SELECT property_id, person_id, photo_type, embedding 
        FROM owners WHERE property_id = ?
    ''', (property_id,))
    
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        prop_id, pers_id, p_type, emb_str = row
        try:
            emb_list = json.loads(emb_str)
            results.append({
                "property_id": prop_id,
                "person_id": pers_id,
                "photo_type": p_type,
                "embedding": emb_list
            })
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode embedding JSON for Person ID: {pers_id}")

    return results


# 3. Webhook Function (STRICT PAYLOAD)
def send_webhook(property_id: int, similarity_score: float, snapshot_base64: str, person_id: int | None):
    """Payload rigorously omitting the note field and including UTC ISO string."""
    payload = {
        "property_id": property_id,
        "similarity_score": similarity_score,
        "snapshot_base64": snapshot_base64,
        "person_id": person_id,
        "occurred_at": datetime.datetime.utcnow().isoformat()
    }

    url = f"{BACKEND_URL}/api/v1/webhooks/intruder"
    headers = {
        "X-Webhook-Api-Key": WEBHOOK_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        return response
    except Exception as e:
        print(f"  [ERROR] Failed to send webhook payload: {e}")
        return None


# 4. Core Processing Loop
def run_matcher():
    print("============================================================")
    print("  FACE MATCHER ENGINE — Monitoring ALIGNED_DIR")
    print("============================================================")
    
    try:
        while True:
            # Grab only files at the root of ALIGNED_DIR (ignore subdirectories/owner/ etc)
            images = glob.glob(os.path.join(ALIGNED_DIR, "*.jpg"))
            
            for img_path in images:
                basename = os.path.basename(img_path)
                
                # Ignore anything with 'owner' in the filename
                if "owner" in basename.lower():
                    continue

                # Expected Format: target_p{property_id}_id{track_id}_{ts}.jpg
                # e.g., target_p42_id105_2026...jpg
                parts = basename.replace(".jpg", "").split("_")
                property_id = None
                
                if len(parts) >= 2:
                    p_segment = parts[1]
                    if p_segment.startswith("p"):
                        try:
                            property_id = int(p_segment[1:])
                        except ValueError:
                            pass
                
                if property_id is None:
                    print(f"  [WARN] Could not parse property_id from {basename}. Skipping.")
                    continue

                # Retrieve specific full-body snapshot converting directly to base64
                snapshot_path = os.path.join(SNAPSHOTS_DIR, basename)
                snapshot_base64 = ""
                if os.path.exists(snapshot_path):
                    with open(snapshot_path, "rb") as sf:
                        snapshot_base64 = base64.b64encode(sf.read()).decode("utf-8")
                
                # Math extraction through Facenet512 utility
                try:
                    target_embedding = get_embedding(img_path)
                except Exception as e:
                    print(f"  [ERROR] DeepFace embedding failed: {e}. Deleting corrupted image.")
                    os.remove(img_path)
                    if os.path.exists(snapshot_path):
                        os.remove(snapshot_path)
                    continue

                # Compare extracted math locally against SQLite DB
                owners = fetch_property_owners(property_id)
                best_distance = 2.0 # Max possible cosine distance
                best_person = None

                for owner in owners:
                    dist = compute_cosine_distance(target_embedding, owner["embedding"])
                    if dist < best_distance:
                        best_distance = dist
                        best_person = owner["person_id"]

                # Translate distance scale to confidence percentage (0.0=100%, 1.0=0%)
                similarity_score = max(0.0, (1.0 - best_distance) * 100.0)

                # Validation Branch
                if best_person is not None and best_distance <= COSINE_THRESHOLD:
                    print(f"  [SAFE] Matched Owner {best_person} on Prop {property_id} | Dist: {best_distance:.3f}")
                    send_webhook(property_id, similarity_score, snapshot_base64, best_person)
                else:
                    print(f"  [ALERT] Intruder Detected on Prop {property_id} | Best Dist: {best_distance:.3f}")
                    # If person_id doesn't pass threshold, send None
                    send_webhook(property_id, similarity_score, snapshot_base64, None)

                # Tear-down raw files after transmission validation
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(snapshot_path):
                    os.remove(snapshot_path)

            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[Face Matcher] Shutting down.")

if __name__ == "__main__":
    run_matcher()
