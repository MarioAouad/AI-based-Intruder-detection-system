"""
config.py  –  Phase 2 Intruder Detection Pipeline
===================================================
Single source of truth for every tunable constant.
Edit this file to calibrate the system for your specific camera/environment.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_WEIGHTS   = "yolov8s.pt"   # Auto-downloaded by Ultralytics if not cached
CONF_THRESHOLD  = 0.70           # High threshold → fewer false positives
TARGET_CLASS    = 0              # COCO class 0 = 'person'
TRACKER_CONFIG  = "bytetrack.yaml"
DEVICE          = "cuda"         # Change to "cpu" if no CUDA GPU is present

# ---------------------------------------------------------------------------
# Spatial constants  (Triangle Similarity distance estimation)
# ---------------------------------------------------------------------------
# Average adult shoulder width in centimetres.
# Used as the known real-world reference measurement.
REAL_TORSO_WIDTH_CM = 45.0

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CALIBRATION REQUIRED                                                   │
# │  Stand exactly 1 metre in front of your camera, run the script, and     │
# │  measure the pixel width of your torso/shoulders in the bounding box.   │
# │  Enter that value below.                                                │
# └─────────────────────────────────────────────────────────────────────────┘
CALIBRATED_PIXEL_WIDTH_1M = 300   # ← Replace with your measured pixel width

# Distance (metres) at which the threat zone activates.
ZONE_RADIUS_METERS = 2.0

# ---------------------------------------------------------------------------
# Temporal constants
# ---------------------------------------------------------------------------
# How many consecutive seconds inside the zone before TRIGGER fires.
TRIGGER_TIME_SECONDS = 30.0

# ---------------------------------------------------------------------------
# Display / visual constants
# ---------------------------------------------------------------------------
# Bounding box colours (BGR format for OpenCV)
COLOR_SAFE    = (0,   220,  0)    # Green  – person outside 2 m zone
COLOR_THREAT  = (0,    0, 255)    # Red    – person inside 2 m zone
COLOR_TRIGGER = (0,  165, 255)    # Orange – timer has fired (TRIGGER)

FONT          = 0                 # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.55
FONT_THICKNESS = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR  = os.path.join(BASE_DIR, "captured_targets")  # Cropped face saves
os.makedirs(TARGETS_DIR, exist_ok=True)
