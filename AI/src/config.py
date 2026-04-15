"""
config.py  –  Phase 2 Intruder Detection Pipeline
===================================================
Single source of truth for every tunable constant.
Edit this file to calibrate the system for your specific camera/environment.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_WEIGHTS   = "yolov8s.pt"   # Auto-downloaded by Ultralytics if not cached
CONF_THRESHOLD  = 0.70           # High threshold → fewer false positives
TARGET_CLASS    = 0              # COCO class 0 = 'person'
TRACKER_CONFIG  = "bytetrack.yaml"
DEVICE          = "cuda"         # Change to "cpu" if no CUDA GPU is present

# ---------------------------------------------------------------------------
# Camera Intrinsic Matrix  (obtained from Auto-YOLO calibration experiment)
# ---------------------------------------------------------------------------
# This 3×3 matrix was produced by camera_calibration_testing/method_auto_yolo.py
# using Triangle Similarity with a person at 1 m and adjusted torso width.
#
#   f_x = f_y = 1108.9 px   (focal length in pixels)
#   c_x = 640.0 px          (principal point x, image centre)
#   c_y = 360.0 px          (principal point y, image centre)
#
CAMERA_MATRIX = np.array([
    [1108.9,      0.0,    640.0],
    [   0.0,   1108.9,    360.0],
    [   0.0,      0.0,      1.0],
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Spatial constants
# ---------------------------------------------------------------------------
# Adjusted torso/shoulder width in centimetres (validated during calibration
# experiments — 49.5 cm gave the most accurate distance readings).
REAL_TORSO_WIDTH_CM = 49.5

# Distance (metres) at which the threat zone activates.
ZONE_RADIUS_METERS = 2.0

# ---------------------------------------------------------------------------
# Temporal constants
# ---------------------------------------------------------------------------
# How many consecutive seconds inside the zone before TRIGGER fires.
TRIGGER_TIME_SECONDS = 10.0

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
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR       = os.path.join(BASE_DIR, "captured_targets")        # Raw torso crops
# TARGETS_FACES_DIR = os.path.join(BASE_DIR, "captured_targets_faces")  # Verified face crops
os.makedirs(TARGETS_DIR, exist_ok=True)
# os.makedirs(TARGETS_FACES_DIR, exist_ok=True)
