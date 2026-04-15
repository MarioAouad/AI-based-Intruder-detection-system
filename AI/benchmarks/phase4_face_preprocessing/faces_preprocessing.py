"""
faces_preprocessing.py  –  Automated Face Preprocessing Worker
================================================================

Standalone experiment script that monitors captured_targets/ every 1 second,
gates images through YuNet face detection, then runs a 3-step preprocessing
chain before saving a clean 160×160 AI-ready face.

Pipeline
--------
  captured_targets/  (raw torso crops from main_watchdog)
         │
    ┌────▼──── Phase 3 Gatekeeper ────────────────────────────┐
    │  YuNet @ 0.7 threshold                                  │
    │  No face found? → delete file from captured_targets/     │
    └────┬────────────────────────────────────────────────────┘
         │  face detected + landmarks
         ▼
    Step 1: Affine eye-alignment → step1_faces_debug_affine/
         ▼
    Step 2: 160×160 crop         → step2_faces_debug_crop/
         ▼
    Step 3: LAB/CLAHE lighting   → step3_faces_debug_lighting/
         ▼
    Final: faces_aligned/        (production-ready)
         ▼
    Debug: faces_debug_comparison/  (side-by-side original vs final)
         ▼
    Cleanup: delete from captured_targets/ + captured_targets_faces/

Usage
-----
    python faces_preprocessing/faces_preprocessing.py

Press Ctrl+C to stop the worker loop.
"""

import os
import sys
import glob
import time
import math
import base64
import shutil

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths — FULL SANDBOX ISOLATION
# ALL folders (input + output) live inside faces_preprocessing/ (SCRIPT_DIR)
# so the script never touches files outside its own sandbox.
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)               # AI Part/

# Add project root to sys.path so we can import face_verifier from AI Part/.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from face_verifier import FaceVerifier  # noqa: E402 (after sys.path edit)

# ---------------------------------------------------------------------------
# Input / output folders  (Bug Fix 1: everything under SCRIPT_DIR)
# ---------------------------------------------------------------------------
CAPTURED_TARGETS_DIR = os.path.join(SCRIPT_DIR, "captured_targets")
CAPTURED_FACES_DIR   = os.path.join(SCRIPT_DIR, "captured_targets_faces")

# Debug + output folders (all inside faces_preprocessing/)
FACES_ALIGNED_DIR        = os.path.join(SCRIPT_DIR, "faces_aligned")
FACES_ALIGNED_OWNER_DIR  = os.path.join(FACES_ALIGNED_DIR, "owner")
FACES_ALIGNED_INTRUDER_DIR = os.path.join(FACES_ALIGNED_DIR, "intruder")
DEBUG_AFFINE_DIR         = os.path.join(SCRIPT_DIR, "step1_faces_debug_affine")
DEBUG_CROP_DIR           = os.path.join(SCRIPT_DIR, "step2_faces_debug_crop")
DEBUG_LIGHTING_DIR       = os.path.join(SCRIPT_DIR, "step3_faces_debug_lighting")
DEBUG_COMPARISON_DIR     = os.path.join(SCRIPT_DIR, "faces_debug_comparison")

for d in [
    CAPTURED_TARGETS_DIR, CAPTURED_FACES_DIR,
    FACES_ALIGNED_DIR, FACES_ALIGNED_OWNER_DIR, FACES_ALIGNED_INTRUDER_DIR,
    DEBUG_AFFINE_DIR, DEBUG_CROP_DIR,
    DEBUG_LIGHTING_DIR, DEBUG_COMPARISON_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_SIZE      = 160         # Final output size (160×160 for FaceNet / ArcFace)
POLL_INTERVAL  = 1.0         # Seconds between scans
IMAGE_EXTS     = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     PREPROCESSING FUNCTIONS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def detect_face_with_landmarks(
    image: np.ndarray, verifier: FaceVerifier
):
    """
    Run YuNet on the image and return the best detection + eye landmarks.

    YuNet detection row format:
        [x, y, w, h,
         right_eye_x, right_eye_y,
         left_eye_x,  left_eye_y,
         nose_x, nose_y,
         mouth_right_x, mouth_right_y,
         mouth_left_x,  mouth_left_y,
         score]

    Returns
    -------
    (bbox, left_eye, right_eye) or None if no face found.
    bbox = (x, y, w, h)
    left_eye / right_eye = (px, py) — pixel coordinates
    """
    if verifier.detector is None:
        return None

    h, w = image.shape[:2]
    verifier.detector.setInputSize((w, h))
    _, detections = verifier.detector.detect(image)

    if detections is None or len(detections) == 0:
        return None

    # Pick the detection with the highest confidence score.
    best = max(detections, key=lambda d: d[14])

    x, y, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])

    # YuNet labels: index 4-5 = right eye, 6-7 = left eye.
    # In face-alignment convention, "left eye" = the person's left eye
    # which appears on the RIGHT side of the image (mirrored).
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
    Rotate the image so that both eyes sit on a perfectly horizontal line.

    Bug Fix 2 — Eye Sorting:
        YuNet may return left/right eyes in either order depending on
        head pose.  If dx becomes negative, atan2 yields ~±180° and the
        face flips upside-down.  We fix this by sorting the two points
        by their X-coordinate so that dx is ALWAYS positive.

    After sorting:
        eye1 = leftmost point on the image  (smaller X)
        eye2 = rightmost point on the image (larger X)
        dx   = eye2.x − eye1.x  →  guaranteed ≥ 0
    """
    # Bug Fix 2: sort eyes by X so dx is always positive → no 180° flip.
    eye1, eye2 = sorted([left_eye, right_eye], key=lambda p: p[0])

    dx = eye2[0] - eye1[0]
    dy = eye2[1] - eye1[1]
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Rotation centre = midpoint between the eyes.
    cx = (eye1[0] + eye2[0]) / 2.0
    cy = (eye1[1] + eye2[1]) / 2.0

    # Build the 2×3 affine rotation matrix (no scaling).
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale=1.0)

    h, w = image.shape[:2]
    aligned = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


# ── Step 2: Standardised 160×160 Crop ────────────────────────────────────

def step2_crop_face(
    aligned_image: np.ndarray,
    bbox: tuple,
    target_size: int = FACE_SIZE,
) -> np.ndarray:
    """
    Crop the face region from the aligned image and resize to target_size².

    We add a small margin (20%) around the bounding box so the crop isn't
    too tight — this helps downstream recognition models.
    """
    x, y, w, h = bbox
    img_h, img_w = aligned_image.shape[:2]

    # Add 20% margin on each side.
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    crop = aligned_image[y1:y2, x1:x2]
    if crop.size == 0:
        # Fallback: use the whole image if crop is degenerate.
        crop = aligned_image

    # Resize to exactly 160×160 with INTER_AREA (best for downscaling).
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


# ── Step 3: LAB / CLAHE Lighting Normalisation ───────────────────────────

def step3_clahe_lighting(image_160: np.ndarray) -> np.ndarray:
    """
    Normalise lighting using CLAHE applied to the L-channel of the LAB
    colour space.

    LAB separates luminance (L) from colour (A, B).  By equalising only
    the L channel we fix uneven lighting without distorting skin tones.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) is preferred
    over standard equalizeHist because it works in small tiles, preventing
    over-amplification of noise.
    """
    lab = cv2.cvtColor(image_160, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result


# ── Debug: side-by-side comparison ────────────────────────────────────────

def build_comparison(original: np.ndarray, final_160: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side image: original (resized to 160 high) | final 160×160.
    """
    h_orig, w_orig = original.shape[:2]
    scale  = 160.0 / h_orig
    w_new  = max(1, int(w_orig * scale))
    left   = cv2.resize(original, (w_new, 160), interpolation=cv2.INTER_AREA)

    # Draw labels
    cv2.putText(left, "ORIGINAL", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
    final_labelled = final_160.copy()
    cv2.putText(final_labelled, "FINAL", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

    # Separator line (white, 2px)
    sep = np.full((160, 2, 3), 255, dtype=np.uint8)
    comparison = np.hstack([left, sep, final_labelled])
    return comparison


# ── Helper: convert image to base64 for webhook payloads ─────────────────

def to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """
    Encode a BGR image as a base64 string (JPEG by default).
    Ready to embed in a JSON webhook payload.

    Returns
    -------
    str
        Base64-encoded string of the image bytes.
    """
    success, buffer = cv2.imencode(ext, image)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CLEANUP HELPERS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def delete_file(path: str) -> None:
    """
    Attempt to delete a file.  On Windows, files opened by cv2.imread()
    may still be locked briefly; we catch that and log an explicit error
    instead of crashing.  (Bug Fix 4: Windows file locking)
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
    except PermissionError:
        print(f"[ERROR] Could not delete {path} — file is locked (Windows).")
    except OSError as e:
        print(f"[ERROR] Could not delete {path}: {e}")


def clean_captured_faces_dir() -> None:
    """Delete every file in captured_targets_faces/ (the raw YuNet-verified folder)."""
    if not os.path.isdir(CAPTURED_FACES_DIR):
        return
    for f in glob.glob(os.path.join(CAPTURED_FACES_DIR, "*")):
        delete_file(f)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          WORKER LOOP                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def scan_for_images() -> list:
    """Return a sorted list of image paths in captured_targets/."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(CAPTURED_TARGETS_DIR, ext)))
    files.sort()
    return files


def process_image(img_path: str, verifier: FaceVerifier) -> bool:
    """
    Run the full preprocessing chain on a single image.

    Routing rules:
        - filename contains "owner"  → faces_aligned/owner/
        - filename contains "target" → faces_aligned/  (intruder staging)

    Returns True if a face was successfully processed and saved,
    False if the image was rejected (no face / unreadable).
    """
    basename    = os.path.basename(img_path)
    name_no_ext = os.path.splitext(basename)[0]
    ext         = os.path.splitext(basename)[1]    # e.g. ".jpg"

    print(f"\n[DETECTION] New raw target detected in captured_targets...")

    image = cv2.imread(img_path)
    if image is None:
        print(f"  [SKIP] Cannot read: {basename}")
        delete_file(img_path)
        return False

    # Bug Fix 4: release the file handle immediately so Windows won't
    # lock it when we try to delete later.  cv2.imread returns a numpy
    # copy, so the file is already fully read — but we delete any
    # internal reference just in case.
    # (numpy array 'image' is our only reference from here on.)

    # ── Phase 3 Gatekeeper: YuNet face check ──────────────────────────
    result = detect_face_with_landmarks(image, verifier)
    if result is None:
        # Bug Fix 3: explicit rejection log BEFORE deletion.
        print(f"  [GATEKEEPER] Rejected: {basename} - No face detected.")
        delete_file(img_path)
        return False

    bbox, left_eye, right_eye = result

    # ── Step 1: Affine eye-alignment ──────────────────────────────────
    aligned = step1_affine_alignment(image, left_eye, right_eye)
    debug1_path = os.path.join(DEBUG_AFFINE_DIR, f"{name_no_ext}_affine.jpg")
    cv2.imwrite(debug1_path, aligned)
    print("[STEP 1] Eye-alignment completed.")

    # ── Step 2: 160×160 crop ──────────────────────────────────────────
    cropped_160 = step2_crop_face(aligned, bbox, FACE_SIZE)
    debug2_path = os.path.join(DEBUG_CROP_DIR, f"{name_no_ext}_crop160.jpg")
    cv2.imwrite(debug2_path, cropped_160)
    print("[STEP 2] Converted to 160x160.")

    # ── Step 3: LAB / CLAHE lighting fix (last step before final save)
    lit_fixed = step3_clahe_lighting(cropped_160)
    debug3_path = os.path.join(DEBUG_LIGHTING_DIR, f"{name_no_ext}_clahe.jpg")
    cv2.imwrite(debug3_path, lit_fixed)
    print("[STEP 3] Lighting fixed.")

    # ── Routing: decide destination folder ─────────────────────────────
    # CRITICAL: keep the original filename — no added prefixes.
    lower_name = basename.lower()

    if "owner" in lower_name:
        dest_dir = FACES_ALIGNED_OWNER_DIR
        print("[ROUTING] Owner detected. Destination: /faces_aligned/owner/.")
    else:
        # Any "target" file (or anything else) → root of faces_aligned/
        # This is the "intruder staging area" for Phase 5 recognition.
        dest_dir = FACES_ALIGNED_DIR
        print("[ROUTING] Unknown target. Destination: /faces_aligned/.")

    # Final save — filename is preserved exactly as-is (e.g. target_ID1_timestamp.jpg)
    final_path = os.path.join(dest_dir, basename)
    cv2.imwrite(final_path, lit_fixed)

    # ── Debug comparison: original vs final ───────────────────────────
    comparison = build_comparison(image, lit_fixed)
    comp_path  = os.path.join(DEBUG_COMPARISON_DIR, f"{name_no_ext}_comparison.jpg")
    cv2.imwrite(comp_path, comparison)

    # ── Double-folder cleanup ─────────────────────────────────────────
    # 1. Delete the original raw file from captured_targets/
    delete_file(img_path)
    # 2. Delete everything in captured_targets_faces/ (redundant after alignment)
    clean_captured_faces_dir()

    print("[SUCCESS] Aligned face saved & cleaning captured_targets and captured_targets_faces...")
    return True


def run_worker():
    """
    Entry point.  Polls captured_targets/ every POLL_INTERVAL seconds
    and processes any new images through the preprocessing chain.
    """
    print("=" * 60)
    print("  FACE PREPROCESSING WORKER")
    print("=" * 60)
    print(f"  Monitoring : {CAPTURED_TARGETS_DIR}")
    print(f"  Output     : {FACES_ALIGNED_DIR}")
    print(f"  Poll rate  : every {POLL_INTERVAL}s")
    print(f"  Face size  : {FACE_SIZE}×{FACE_SIZE}")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")

    # Load YuNet once for the entire session.
    verifier = FaceVerifier()
    if not verifier.is_loaded():
        print("[Worker] FATAL: YuNet failed to load. Cannot proceed.")
        sys.exit(1)

    processed_count = 0

    try:
        while True:
            images = scan_for_images()

            for img_path in images:
                success = process_image(img_path, verifier)
                if success:
                    processed_count += 1

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[Worker] Stopped. Total faces processed: {processed_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_worker()
