import os
import sys
import glob
import time

# Tell Python to look one folder up (in 'src') so it can find config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from db_manager import init_db, save_owner
from embedding_utils import get_embedding

OWNER_DIR = os.path.join(BASE_DIR, "data", "faces_aligned", "owner")

def run_db_worker():
    print("============================================================")
    print("  DATABASE WORKER — Listening for new aligned owners...")
    print("============================================================")
    
    # Ensure the database is ready
    init_db()
    
    try:
        while True:
            # Grab all JPGs in the owner folder
            images = glob.glob(os.path.join(OWNER_DIR, "*.jpg"))
            
            for img_path in images:
                basename = os.path.basename(img_path)
                
                # Filename format: owner_{property_id}_{person_id}_{photo_type}.jpg
                # Example: owner_42_105_left_profile.jpg
                parts = basename.replace(".jpg", "").split("_")
                
                if len(parts) >= 4:
                    property_id = int(parts[1])
                    person_id = int(parts[2])
                    # Rejoin the end in case the type has an underscore (like 'left_profile')
                    photo_type = "_".join(parts[3:])
                    
                    print(f"[DB Worker] Processing: Property {property_id} | Person {person_id} | {photo_type}")
                    
                    # 1. Turn the image into math using our utility
                    embedding = get_embedding(img_path)
                    
                    # 2. Save it to SQLite
                    save_owner(property_id, person_id, photo_type, embedding)
                    
                    # 3. Delete the image so we don't process it twice
                    os.remove(img_path)
                    print(f"  └─> Saved to DB and deleted image.")
                
            # Wait 2 seconds before checking the folder again
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n[DB Worker] Shutting down.")

if __name__ == "__main__":
    run_db_worker()