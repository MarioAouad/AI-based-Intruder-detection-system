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
 │                  IDENTITY VERIFICATION                              │
 │                                                                     │
 │   Facenet512 + Cosine Distance                                      │
 │   1-to-N Gallery Matching (threshold = 0.65)                        │
 │       │                                                             │
 │       ├──► OWNER  → Unlock / Allow                                  │
 │       └──► INTRUDER → Alert / Block                                 │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AI /
│
├── src/                       ← Core production modules
│   ├── config.py              ← Single source of truth for all pipeline constants
│   ├── watchdog.py            ← Phase 2: Real-time detection + tracking + trigger loop
│   ├── spatial_math.py        ← Distance estimation via triangle similarity
│   ├── threat_timer.py        ← Per-ID zone timer (10s threshold)
│   ├── face_verifier.py       ← YuNet face detection wrapper (get_face_data → bbox + eyes)
│   ├── face_processor.py      ← Phase 4: Production face preprocessing worker
│   └── face_detection_yunet_2023mar.onnx
│
├── data/                      ← Runtime data directories
│   ├── captured_targets/      ← Raw torso crops saved by watchdog
│   └── faces_aligned/         ← Final 160×160 AI-ready faces
│       └── owner/             ← Owner-routed faces
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

---

## Core Module Descriptions

### `src/config.py`
Central configuration file. Contains the **camera intrinsic matrix**, YOLO model settings, spatial/temporal thresholds, display colours, and filesystem paths. Every other module reads from here.

### `src/watchdog.py`
The real-time detection loop. Captures webcam frames, runs YOLOv8s person detection with ByteTrack, estimates distance using `spatial_math.py`, accumulates zone time via `threat_timer.py`, and saves torso crops to `data/captured_targets/` when the 10-second trigger fires.

### `src/face_processor.py`
Production Phase 4 worker. Polls `data/captured_targets/` every second, runs YuNet gatekeeper, then passes the image through a fully in-memory 3-step chain. Only the final face is written to disk. Routes filenames containing `"owner"` to `data/faces_aligned/owner/`.

### `src/face_verifier.py`
Lightweight wrapper around OpenCV's YuNet ONNX model. Exposes a `get_face_data(image) → (bbox, left_eye, right_eye) | None` method used by the face processor to extract the best face bounding box and eye landmarks for downstream alignment.

### `src/spatial_math.py`
`DistanceEstimator` class that calculates real-world distance from the bounding-box width using Triangle Similarity and the calibrated camera matrix.

### `src/threat_timer.py`
`ZoneTimer` class that tracks how long each tracked person ID has been continuously inside the threat zone. Fires a trigger after `TRIGGER_TIME_SECONDS` (default 10s).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (Real-time detection + Face processing)
python main.py
```

> **Note for Developers**: You can still run the modules in isolation for targeted debugging by opening separate terminals and executing `python src/watchdog.py` and `python src/face_processor.py`.

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
