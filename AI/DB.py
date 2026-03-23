"""
setup_enterprise_db.py
Generates the fully normalized, 10-table SQLite database for the AI Intruder System.
"""

import sqlite3

def initialize_database():
    conn = sqlite3.connect('enterprise_security.db')
    
    # ⚠️ CRITICAL: Must be enabled in SQLite for CASCADE DELETE to work
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # 1. USER
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firebase_uid TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT,
            phone_number TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. USER_CONSENT
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_consents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            consent_type TEXT,
            accepted BOOLEAN DEFAULT 0,
            accepted_at DATETIME,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # 3. PROPERTY
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT,
            address TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # 4. CAMERA_STREAM (1-to-1 with Property)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_streams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER UNIQUE NOT NULL,
            source_url TEXT,
            stream_type TEXT,
            is_enabled BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (property_id) REFERENCES properties(id) ON DELETE CASCADE
        )
    ''')

    # 5. PERSON (AI Biometric Whitelist)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            name TEXT,
            face_embedding BLOB,  -- AI Math Vector Storage restored
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (property_id) REFERENCES properties(id) ON DELETE CASCADE
        )
    ''')

    # 6. PERSON_PHOTO (Training & UI Avatar)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            is_display BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        )
    ''')

    # 7. PROTOCOL (Lookup Table)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS protocols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
    ''')

    # 8. PROTOCOL_ASSIGNMENT (Junction Table)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS protocol_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            protocol_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY (protocol_id) REFERENCES protocols(id) ON DELETE CASCADE
        )
    ''')

    # 9. EVENT (The 30-Second AI Trigger Ledger)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            person_id INTEGER,    -- Nullable: AI might not recognize the face
            similarity_score REAL,
            ai_status TEXT,       -- 'AUTHORIZED', 'INTRUDER', 'HUMAN_REVIEW'
            snapshot_path TEXT,
            occurred_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            note TEXT,
            verified_intruder BOOLEAN DEFAULT 0,
            protocols_activated BOOLEAN DEFAULT 0,
            distance_meters REAL,
            dwell_time_seconds REAL,
            expires_at DATETIME,  -- The 72-hour cleanup target
            FOREIGN KEY (property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
        )
    ''')

    # 10. NOTIFICATION_LOG
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            channel TEXT,
            status TEXT,
            detail TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print("Database 'enterprise_security.db' created successfully with full cascade rules.")

if __name__ == "__main__":
    initialize_database()