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
AI Part/
│
├── config.py                  ← Single source of truth for all pipeline constants
├── main_watchdog.py           ← Phase 2: Real-time detection + tracking + trigger loop
├── spatial_math.py            ← Distance estimation via triangle similarity
├── threat_timer.py            ← Per-ID zone timer (10s threshold)
├── face_verifier.py           ← YuNet face detection wrapper (verify True/False)
├── face_processor.py          ← Phase 4: Production face preprocessing worker
├── .gitignore                 ← Blocks images, models, caches from version control
├── requirements.txt           ← Root Python dependencies
│
├── captured_targets/          ← Raw torso crops saved by main_watchdog
├── faces_aligned/             ← Final 160×160 AI-ready faces
│   └── owner/                 ← Owner-routed faces
│
├── camera_calibration_testing/    ← Phase 1: Camera intrinsic calibration experiments
│   ├── method_checkerboard.py
│   ├── method_auto_yolo.py
│   └── compare_distances.py
│
├── PersonDetection_Benchmarking/  ← Phase 2: YOLO / RT-DETR speed vs. accuracy tests
│   ├── config.py
│   ├── main_benchmark.py
│   ├── video_tracker.py
│   └── benchmark_logger.py
│
├── face_detection_benchmarking/   ← Phase 3: Face detector comparison (YuNet vs MTCNN)
│   └── benchmark_faces.py
│
├── faces_preprocessing/           ← Phase 3.5: Sandbox preprocessing pipeline (debug)
│   └── faces_preprocessing.py
│
├── phase5_benchmarking/           ← Phase 5: Face recognition model benchmarking
│   └── custom_benchmark.py
│
└── phase6_gallery_test/           ← Phase 6: 1-to-N gallery matching test suite
    └── gallery_test.py
```

---

## Pipeline Phases

| Phase | Objective | Key Files |
|-------|-----------|-----------|
| **Phase 1** | Camera calibration — derive intrinsic matrix | `camera_calibration_testing/` |
| **Phase 2** | Person detection + tracking + distance-based trigger | `main_watchdog.py`, `spatial_math.py`, `threat_timer.py` |
| **Phase 3** | Face detection benchmarking (YuNet selected) | `face_detection_benchmarking/benchmark_faces.py` |
| **Phase 3.5** | Face preprocessing sandbox with debug outputs | `faces_preprocessing/faces_preprocessing.py` |
| **Phase 4** | Production face processor (in-memory pipeline) | `face_processor.py` |
| **Phase 5** | Recognition model benchmarking (Facenet512 selected) | `phase5_benchmarking/custom_benchmark.py` |
| **Phase 6** | 1-to-N gallery matching + margin-of-safety analysis | `phase6_gallery_test/gallery_test.py` |

---

## Core Module Descriptions

### `config.py`
Central configuration file. Contains the **camera intrinsic matrix** (from Auto-YOLO calibration), YOLO model settings, spatial/temporal thresholds, display colours, and filesystem paths. Every other module reads from here.

### `main_watchdog.py`
The real-time detection loop. Captures webcam frames, runs YOLOv8s person detection with ByteTrack, estimates distance using `spatial_math.py`, accumulates zone time via `threat_timer.py`, and saves torso crops to `captured_targets/` when the 10-second trigger fires.

### `face_processor.py`
Production Phase 4 worker. Polls `captured_targets/` every second, runs YuNet gatekeeper, then passes the image through a fully in-memory 3-step chain (affine alignment → 160×160 crop → CLAHE lighting fix). Only the final face is written to disk. Routes filenames containing `"owner"` to `faces_aligned/owner/`.

### `face_verifier.py`
Lightweight wrapper around OpenCV's YuNet ONNX model. Exposes a `verify(crop_img) → bool` method used as a binary face/no-face gatekeeper.

### `spatial_math.py`
`DistanceEstimator` class that calculates real-world distance from the bounding-box width using Triangle Similarity and the calibrated camera matrix.

### `threat_timer.py`
`ZoneTimer` class that tracks how long each tracked person ID has been continuously inside the threat zone. Fires a trigger after `TRIGGER_TIME_SECONDS` (default 10s).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run real-time detection
python main_watchdog.py

# (In a second terminal) Run face processor
python face_processor.py
```

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
