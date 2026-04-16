# AI-Based Intruder Detection System

> **University CCE Project** — A multi-phase computer vision pipeline that detects humans, tracks their proximity, extracts and preprocesses faces, and performs identity verification to distinguish owners from intruders in real-time.

---

## System Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    REAL-TIME DETECTION LOOP                        │
 │                                                                     │
 │   Webcam Frame                                                      │
 │       │                                                             │
 │       ▼                                                             │
 │   YOLOv8s Person Detection ──► ByteTrack ID Assignment              │
 │       │                            │                                │
 │       ▼                            ▼                                │
 │   Distance Estimation         Zone Timer (10s threshold)            │
 │   (Triangle Similarity)           │                                 │
 │       │                           ▼                                 │
 │       └──────────► TRIGGER ──► Save Torso Crop                      │
 │                                   │                                 │
 └───────────────────────────────────┼─────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    FACE PROCESSING WORKER                          │
 │                                                                     │
 │   captured_targets/                                                 │
 │       │                                                             │
 │       ▼                                                             │
 │   YuNet Gatekeeper (0.7 threshold)                                  │
 │       │                                                             │
 │       ▼                                                             │
 │   Step 1: Affine Eye-Alignment                                      │
 │   Step 2: 160×160 Face Crop (20% margin)                            │
 │   Step 3: LAB/CLAHE Lighting Normalisation                          │
 │       │                                                             │
 │       ▼                                                             │
 │   faces_aligned/ (owner/ or intruder staging)                       │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                  IDENTITY VERIFICATION & API                        │
 │                                                                     │
 │   Facenet512 + Cosine Distance                                      │
 │   1-to-N Gallery Matching (threshold = 0.65)                        │
 │       │                                                             │
 │       ├──► OWNER  → HTTP POST Owner Registration                    │
 │       └──► INTRUDER → HTTP POST Intruder Alert                      │
 │                  (Sent to Backend API)                              │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AI/
│
├── main.py                    ← Core Server Orchestrator. Launches face processing and DB backend.
├── deployments/               ← Edge Node Launchers
│   ├── run_p1_live.py         ← Starts live camera for Property 1
│   ├── run_p2_live.py         ← Starts live camera for Property 2
│   └── run_p3_video.py        ← Starts video feed for Property 3 from test_feeds
├── src/                       ← Core production modules
│   ├── api/                   ← Backend API integration
│   │   └── server.py          ← FastAPI server for backend communication
│   ├── database/              ← SQLite & Embedding generation
│   │   ├── db_manager.py      ← SQLite schema and save logic
│   │   ├── db_worker.py       ← Background pipeline saving embeddings to DB
│   │   └── embedding_utils.py ← Facenet512 512-D vector extraction
│   ├── config.py              ← Single source of truth for all pipeline constants
│   ├── watchdog.py            ← Phase 2: Real-time detection + tracking + trigger loop
│   ├── spatial_math.py        ← Distance estimation via triangle similarity
│   ├── threat_timer.py        ← Per-ID zone timer (10s threshold)
│   ├── face_verifier.py       ← YuNet face detection wrapper (get_face_data → bbox + eyes)
│   ├── face_processor.py      ← Phase 4: Production face preprocessing worker
│   ├── face_detection_yunet_2023mar.onnx  ← YuNet face detection model weights
│   └── yolov8s.pt             ← YOLOv8s person detection model weights
│
├── data/                      ← Runtime data directories
│   ├── captured_targets/      ← Raw torso crops saved by watchdog
│   ├── test_feeds/            ← Test videos for simulated edge deployments
│   ├── faces_aligned/         ← Final 160×160 AI-ready faces
│   │   └── owner/             ← Owner-routed faces
│   └── faces.db               ← SQLite database storing 512-D facial embeddings
│
├── benchmarks/                ← Experiment and scale analysis
│   ├── phase1_camera_calibration/
│   ├── phase2_person_detection/
│   ├── phase3_face_detection/
│   ├── phase4_face_preprocessing/
│   ├── phase5_face_recognition/
│   └── phase6_gallery_test/
│
├── .gitignore                 ← Blocks images, models, caches from version control
├── .env.example               ← Template for API keys and environment variables
└── requirements.txt           ← Root Python dependencies
```

---

## Pipeline Phases

| Phase | Objective | Key Files |
|-------|-----------|-----------|
| **Phase 1** | Camera calibration — derive intrinsic matrix | `benchmarks/phase1_camera_calibration/` |
| **Phase 2** | Person detection + tracking + distance-based trigger | `src/watchdog.py`, `src/spatial_math.py`, `src/threat_timer.py` |
| **Phase 3** | Face detection benchmarking (YuNet selected) | `benchmarks/phase3_face_detection/benchmark_faces.py` |
| **Phase 3.5** | Face preprocessing sandbox with debug outputs | `benchmarks/phase4_face_preprocessing/faces_preprocessing.py` |
| **Phase 4** | Production face processor (in-memory pipeline) | `src/face_processor.py` |
| **Phase 5** | Recognition model benchmarking (Facenet512 selected) | `benchmarks/phase5_face_recognition/custom_benchmark.py` |
| **Phase 6** | 1-to-N gallery matching + margin-of-safety analysis | `benchmarks/phase6_gallery_test/gallery_test.py` |
| **Phase 7** | Backend API & Database Integration | `src/api/server.py`, `src/database/` |

---

### Core vs. Edge Architecture

To maximize scalability, the system is strictly split between a **Core Server** and **Edge Nodes**:
- **The Core Server (`main.py`)**: Responsible for handling the heavy Backend processing. It runs the Face Preprocessing worker and the Database SQLite worker concurrently, operating completely headlessly while waiting for images to arrive in the shared `data/` folder.
- **The Edge Nodes (`deployments/`)**: Every physical camera acts as an independent Edge Node. Scripts inside `deployments/` launch instances of `src/watchdog.py` parameterised with their respective `--property` IDs and `--camera` feeds. They monitor their video sources in real time, and when an intruder is detected, they stamp the target ID and Property ID onto the cropped image and drop it into `data/captured_targets/` where the Core Server will pick it up.

---

## Core Module Descriptions

### `main.py` (Core Server Orchestrator)
The backend entrypoint for the system. It leverages the `subprocess` module to spin up `face_processor.py` and `db_worker.py` concurrently. It acts as the central AI Brain that runs continuously without consuming camera feed resources directly. It safely stops workers on script exit via `KeyboardInterrupt`.

### `src/config.py`
Central configuration file. Contains the **camera intrinsic matrix**, YOLO model settings, spatial/temporal thresholds, display colours, and filesystem paths. Every other module reads from here.

### `src/watchdog.py`
The parameter-driven Edge Node detection loop. Configurable via `argparse` for specific `--property` IDs and `--camera` streams. Captures frames, runs YOLOv8s person detection with ByteTrack, estimates distance using `spatial_math.py`, accumulates zone time via `threat_timer.py`, and saves tagged torso crops to `data/captured_targets/` when the 10-second trigger fires.

### `src/face_processor.py`
Production Phase 4 worker. Polls `data/captured_targets/` every second, runs YuNet gatekeeper, then passes the image through a fully in-memory 3-step chain. Only the final face is written to disk. Routes filenames containing `"owner"` to `data/faces_aligned/owner/`.

### `src/face_verifier.py`
Lightweight wrapper around OpenCV's YuNet ONNX model. Exposes a `get_face_data(image) → (bbox, left_eye, right_eye) | None` method used by the face processor to extract the best face bounding box and eye landmarks for downstream alignment.

### `src/spatial_math.py`
`DistanceEstimator` class that calculates real-world distance from the bounding-box width using Triangle Similarity and the calibrated camera matrix.

### `src/threat_timer.py`
`ZoneTimer` class that tracks how long each tracked person ID has been continuously inside the threat zone. Fires a trigger after `TRIGGER_TIME_SECONDS` (default 10s).

### Database Layer (`src/database/`)
- **`db_manager.py`**: Manages the SQLite `faces.db` using a Composite Primary Key of `(property_id, person_id, photo_type)` to ensure data integrity.
- **`db_worker.py`**: A background process that monitors the owner folder, extracts embeddings, and saves to SQLite before deleting the image.
- **`embedding_utils.py`**: A unified utility using Facenet512 to generate 512-D vectors.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create environment file from template
cp .env.example .env  # Then edit .env to add your API keys

# --- TERMINAL 1: Start the Core Server ---
# Start the main AI pipeline (Face Processing + DB Worker)
python main.py

# --- TERMINAL 2: Start an Edge Node ---
# Start camera deployment for Property 1
python deployments/run_p1_live.py

# --- TERMINAL 3: Start the Backend API (Optional) ---
uvicorn src.api.server:app --host 0.0.0.0 --port 8001
```

> **Note for Developers**: You can still run the modules in isolation for targeted debugging by passing the specific `argparse` arguments directly, for example: `python src/watchdog.py --property 1 --camera 0`.

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- **ultralytics** — YOLOv8 / YOLO11 inference
- **opencv-python** — Image processing, YuNet face detection
- **deepface** — Face recognition (Facenet512)
- **numpy** — Numerical operations
- **matplotlib** — Benchmark visualisation

---

## Licence

University project — for educational and research purposes.
