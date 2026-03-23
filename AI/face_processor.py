"""
face_processor.py  –  Phase 4: Production Face Preprocessing Worker
=====================================================================

Production-optimised version of the sandbox (faces_preprocessing/).
Monitors captured_targets/ for new torso crops, runs the 3-step
preprocessing chain entirely in-memory, and saves only the final
160×160 AI-ready face to faces_aligned/.

Pipeline
--------
  captured_targets/  (raw torso crops from main_watchdog)
         │
    ┌────▼──── Gatekeeper ────────────────────────────────────┐
    │  YuNet @ 0.7 threshold                                  │
    │  No face? → delete file & skip                          │
    └────┬────────────────────────────────────────────────────┘
         │  face + eye landmarks
         ▼
    Step 1: Affine eye-alignment      (in-memory only)
    Step 2: 160×160 face crop         (in-memory only)
    Step 3: LAB/CLAHE lighting fix    (in-memory only)
         ▼
    Final save → faces_aligned/       (owner/ or root)
         ▼
    Cleanup  → delete raw from captured_targets/

Routing
-------
    "owner"  in filename → faces_aligned/owner/
    anything else        → faces_aligned/  (intruder staging)

Usage
-----
    python face_processor.py

Press Ctrl+C to stop.
"""

import os
import glob
import time
import math
import base64

import cv2
import numpy as np

import config
from face_verifier import FaceVerifier

# ---------------------------------------------------------------------------
# Paths  (all derived from config.py — single source of truth)
# ---------------------------------------------------------------------------
FACES_ALIGNED_DIR         = os.path.join(config.BASE_DIR, "faces_aligned")
FACES_ALIGNED_OWNER_DIR   = os.path.join(FACES_ALIGNED_DIR, "owner")
FACES_ALIGNED_INTRUDER_DIR = os.path.join(FACES_ALIGNED_DIR, "intruder")

for d in [FACES_ALIGNED_DIR, FACES_ALIGNED_OWNER_DIR, FACES_ALIGNED_INTRUDER_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_SIZE      = 160         # 160×160 for FaceNet / ArcFace
POLL_INTERVAL  = 1.0         # Seconds between scans
IMAGE_EXTS     = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     PREPROCESSING FUNCTIONS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def detect_face_with_landmarks(image: np.ndarray, verifier: FaceVerifier):
    """
    Run YuNet and return the best detection + eye landmarks.

    YuNet row: [x, y, w, h,
                right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                nose_x, nose_y,
                mouth_right_x, mouth_right_y, mouth_left_x, mouth_left_y,
                score]

    Returns (bbox, left_eye, right_eye) or None.
    """
    if verifier.detector is None:
        return None

    h, w = image.shape[:2]
    verifier.detector.setInputSize((w, h))
    _, detections = verifier.detector.detect(image)

    if detections is None or len(detections) == 0:
        return None

    best = max(detections, key=lambda d: d[14])
    x, y, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    right_eye = (float(best[4]), float(best[5]))
    left_eye  = (float(best[6]), float(best[7]))

    return (x, y, bw, bh), left_eye, right_eye


# ── Step 1: Affine Eye-Alignment ──────────────────────────────────────────

def step1_affine_alignment(
    image: np.ndarray,
    left_eye: tuple,
    right_eye: tuple,
) -> np.ndarray:
    """
    Rotate the image so both eyes are perfectly horizontal.
    Eyes are X-sorted to guarantee dx ≥ 0 (prevents 180° flip).
    """
    eye1, eye2 = sorted([left_eye, right_eye], key=lambda p: p[0])

    dx = eye2[0] - eye1[0]
    dy = eye2[1] - eye1[1]
    angle_deg = math.degrees(math.atan2(dy, dx))

    cx = (eye1[0] + eye2[0]) / 2.0
    cy = (eye1[1] + eye2[1]) / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale=1.0)

    h, w = image.shape[:2]
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ── Step 2: 160×160 Face Crop ────────────────────────────────────────────

def step2_crop_face(
    aligned_image: np.ndarray,
    bbox: tuple,
    target_size: int = FACE_SIZE,
) -> np.ndarray:
    """Crop the face region with 20% margin and resize to target_size²."""
    x, y, w, h = bbox
    img_h, img_w = aligned_image.shape[:2]

    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    crop = aligned_image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = aligned_image

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ── Step 3: LAB / CLAHE Lighting Normalisation ───────────────────────────

def step3_clahe_lighting(image_160: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L-channel of LAB colour space."""
    lab = cv2.cvtColor(image_160, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l_ch)

    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# ── Helper: base64 encoding for webhook payloads ─────────────────────────

def to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a BGR image as a base64 string."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          FILE HELPERS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def delete_file(path: str) -> None:
    """Delete a file with Windows-safe error handling."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except PermissionError:
        print(f"[ERROR] Could not delete {path} — file is locked (Windows).")
    except OSError as e:
        print(f"[ERROR] Could not delete {path}: {e}")


def scan_for_images() -> list:
    """Return a sorted list of image paths in captured_targets/."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(config.TARGETS_DIR, ext)))
    files.sort()
    return files


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       PROCESS IMAGE                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def process_image(img_path: str, verifier: FaceVerifier) -> bool:
    """
    Run the full 3-step preprocessing chain on a single image.
    All intermediate arrays stay in memory — only the final 160×160
    face is written to disk.

    Routing:
        "owner"  in filename → faces_aligned/owner/
        anything else        → faces_aligned/  (intruder staging)

    Returns True on success, False if rejected.
    """
    basename = os.path.basename(img_path)

    print(f"\n[DETECTION] New raw target detected in captured_targets...")

    image = cv2.imread(img_path)
    if image is None:
        print(f"  [SKIP] Cannot read: {basename}")
        delete_file(img_path)
        return False

    # ── Gatekeeper: YuNet face check ──────────────────────────────────
    result = detect_face_with_landmarks(image, verifier)
    if result is None:
        print(f"  [GATEKEEPER] Rejected: {basename} - No face detected.")
        delete_file(img_path)
        return False

    bbox, left_eye, right_eye = result

    # ── Step 1 → Step 2 → Step 3  (in-memory pipeline) ───────────────
    aligned     = step1_affine_alignment(image, left_eye, right_eye)
    print("[STEP 1] Eye-alignment completed.")

    cropped_160 = step2_crop_face(aligned, bbox, FACE_SIZE)
    print("[STEP 2] Converted to 160x160.")

    lit_fixed   = step3_clahe_lighting(cropped_160)
    print("[STEP 3] Lighting fixed.")

    # ── Routing ───────────────────────────────────────────────────────
    lower_name = basename.lower()

    if "owner" in lower_name:
        dest_dir = FACES_ALIGNED_OWNER_DIR
        print("[ROUTING] Owner detected. Destination: /faces_aligned/owner/.")
    else:
        dest_dir = FACES_ALIGNED_DIR
        print("[ROUTING] Unknown target. Destination: /faces_aligned/.")

    # ── Final save (the ONLY cv2.imwrite in the pipeline) ─────────────
    final_path = os.path.join(dest_dir, basename)
    cv2.imwrite(final_path, lit_fixed)

    # ── Cleanup: delete the raw target ────────────────────────────────
    delete_file(img_path)

    print("[SUCCESS] Aligned face saved & raw target cleaned.")
    return True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          WORKER LOOP                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_worker():
    """
    Entry point.  Polls captured_targets/ every POLL_INTERVAL seconds
    and processes any new images through the production pipeline.
    """
    print("=" * 60)
    print("  FACE PROCESSOR — Phase 4 (Production)")
    print("=" * 60)
    print(f"  Monitoring : {config.TARGETS_DIR}")
    print(f"  Output     : {FACES_ALIGNED_DIR}")
    print(f"  Poll rate  : every {POLL_INTERVAL}s")
    print(f"  Face size  : {FACE_SIZE}×{FACE_SIZE}")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")

    verifier = FaceVerifier()
    if not verifier.is_loaded():
        print("[Worker] FATAL: YuNet failed to load. Cannot proceed.")
        raise SystemExit(1)

    processed_count = 0

    try:
        while True:
            images = scan_for_images()

            for img_path in images:
                if process_image(img_path, verifier):
                    processed_count += 1

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[Worker] Stopped. Total faces processed: {processed_count}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_worker()
