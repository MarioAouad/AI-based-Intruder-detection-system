import sqlite3
import json
import os
import config

# Put the database right next to the image folders
DB_PATH = os.path.join(config.BASE_DIR, "data", "faces.db")

def init_db():
    """Creates the SQLite database and the owners table with a Composite Primary Key."""
    # Ensure the data folder exists before making the database
    os.makedirs(os.path.join(config.BASE_DIR, "data"), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Your Composite Primary Key logic is applied here
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS owners (
            property_id INTEGER,
            person_id INTEGER,
            photo_type TEXT,
            embedding TEXT,
            PRIMARY KEY (property_id, person_id, photo_type)
        )
    ''')
    conn.commit()
    conn.close()

def save_owner(property_id: int, person_id: int, photo_type: str, embedding: list[float]):
    """Converts the embedding list to JSON and saves it to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # SQLite cannot store raw Python lists, so we turn the list into a JSON string
    embedding_json = json.dumps(embedding)
    
    # INSERT OR REPLACE handles re-registrations safely without crashing
    cursor.execute('''
        INSERT OR REPLACE INTO owners (property_id, person_id, photo_type, embedding)
        VALUES (?, ?, ?, ?)
    ''', (property_id, person_id, photo_type, embedding_json))
    
    conn.commit()
    conn.close()