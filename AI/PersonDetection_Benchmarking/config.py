"""
config.py
---------
Central configuration & constants for the object detection benchmarking suite.
All tuneable parameters live here so that the other modules never hard-code values.
"""

import os

# ---------------------------------------------------------------------------
# GPU / Device settings
# ---------------------------------------------------------------------------
DEVICE = "cuda"          # Use "cpu" if no CUDA-capable GPU is available.
HALF_PRECISION = True    # FP16 – cuts VRAM and increases speed on RTX cards.

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
TARGET_CLASS = 0         # COCO class 0 = 'person'
CONF_THRESHOLD = 0.70    # Minimum object-confidence to keep a detection.
IOU_THRESHOLD = 0.50     # IoU threshold used during NMS.

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
# ByteTrack is bundled with Ultralytics and requires no extra install.
# Alternatives: "botsort.yaml"
TRACKER_CONFIG = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH    = os.path.join(BASE_DIR, "test_video.mp4")   # 60-second test clip
OUTPUT_CSV    = os.path.join(BASE_DIR, "benchmark_results.csv")
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "output_videos") # Optional annotated outputs

# ---------------------------------------------------------------------------
# Model list
# Each entry is a dict with:
#   "name"    - human-readable label used in CSVs / console output
#   "weights" - Ultralytics model identifier (auto-downloaded if not cached)
#   "type"    - "yolo" | "rtdetr" | "rfdetr" | "rtmdet"
# ---------------------------------------------------------------------------
MODELS = [
    # ── YOLOv8 family ────────────────────────────────────────────────────
    {"name": "YOLOv8-Nano",   "weights": "yolov8n.pt",  "type": "yolo"},
    {"name": "YOLOv8-Small",  "weights": "yolov8s.pt",  "type": "yolo"},
    {"name": "YOLOv8-Medium", "weights": "yolov8m.pt",  "type": "yolo"},

    # ── YOLO11 family ─────────────────────────────────────────────────────
    {"name": "YOLO11-Nano",   "weights": "yolo11n.pt",  "type": "yolo"},
    {"name": "YOLO11-Small",  "weights": "yolo11s.pt",  "type": "yolo"},
    {"name": "YOLO11-Medium", "weights": "yolo11m.pt",  "type": "yolo"},

    # ── Transformer / Hybrid architectures ───────────────────────────────
    # RT-DETR: Real-Time DEtection TRansformer (Ultralytics wrapper)
    {"name": "RT-DETR-L",     "weights": "rtdetr-l.pt", "type": "rtdetr"},

    # RF-DETR: Roboflow DETR – install via  pip install rfdetr
    # Uses the rfdetr Python API instead of Ultralytics directly.
    {"name": "RF-DETR",       "weights": "rf-detr",     "type": "rfdetr"},

    # RTMDet: mmdeploy-based detector – install via  pip install mmdet
    # Requires an MMDET config; the "weights" field is used as a tag only.
    {"name": "RTMDet",        "weights": "rtmdet",      "type": "rtmdet"},
]

# ---------------------------------------------------------------------------
# Benchmark display settings
# ---------------------------------------------------------------------------
# How many characters wide the console summary table should be.
CONSOLE_TABLE_WIDTH = 110
