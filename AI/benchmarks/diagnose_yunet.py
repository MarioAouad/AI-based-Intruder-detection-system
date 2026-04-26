"""
Diagnostic script: Test YuNet face detection on images.
Drop images into data/captured_targets/ BEFORE running this.
This will NOT delete images - it only tests detection.
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageOps

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from src.face_verifier import FaceVerifier

TARGETS_DIR = os.path.join(BASE_DIR, "data", "captured_targets")

def diagnose_image(img_path, verifier):
    basename = os.path.basename(img_path)
    print(f"\n{'='*60}")
    print(f"  DIAGNOSING: {basename}")
    print(f"{'='*60}")
    
    # 1. Raw file info
    file_size = os.path.getsize(img_path)
    print(f"  File size: {file_size:,} bytes")
    
    # 2. Try reading with raw cv2.imread (OLD method)
    raw_cv = cv2.imread(img_path)
    if raw_cv is None:
        print(f"  [cv2.imread] FAILED - cannot read file")
    else:
        print(f"  [cv2.imread] OK - shape={raw_cv.shape}, dtype={raw_cv.dtype}")
        result_raw = verifier.get_face_data(raw_cv)
        if result_raw is None:
            print(f"  [YuNet on raw cv2] REJECTED - no face detected")
        else:
            bbox, le, re = result_raw
            print(f"  [YuNet on raw cv2] ACCEPTED - bbox={bbox}, left_eye={le}, right_eye={re}")
    
    # 3. Try reading with PIL + EXIF transpose (NEW method)
    try:
        pil_img = Image.open(img_path)
        print(f"  [PIL] Format={pil_img.format}, Size={pil_img.size}, Mode={pil_img.mode}")
        
        # Check EXIF orientation
        exif = pil_img.getexif()
        orientation = exif.get(274, None)  # 274 = Orientation tag
        print(f"  [EXIF] Orientation tag = {orientation}  (None=not set, 1=normal, 6=90CW, 8=90CCW, 3=180)")
        
        pil_transposed = ImageOps.exif_transpose(pil_img)
        pil_rgb = pil_transposed.convert("RGB")
        cv_fixed = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
        
        print(f"  [PIL→BGR] shape={cv_fixed.shape}, dtype={cv_fixed.dtype}")
        
        result_fixed = verifier.get_face_data(cv_fixed)
        if result_fixed is None:
            print(f"  [YuNet on PIL-fixed] REJECTED - no face detected")
        else:
            bbox, le, re = result_fixed
            print(f"  [YuNet on PIL-fixed] ACCEPTED - bbox={bbox}, left_eye={le}, right_eye={re}")

        # 4. Try with downscale to 1024 max dimension
        h, w = cv_fixed.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            resized = cv2.resize(cv_fixed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            print(f"  [DOWNSCALE] {w}×{h} → {resized.shape[1]}×{resized.shape[0]}")
            result_resized = verifier.get_face_data(resized)
            if result_resized is None:
                print(f"  [YuNet on DOWNSCALED] REJECTED - no face detected")
            else:
                bbox, le, re = result_resized
                print(f"  [YuNet on DOWNSCALED] ACCEPTED ✓ - bbox={bbox}, left_eye={le}, right_eye={re}")
            
    except Exception as e:
        print(f"  [PIL] ERROR: {e}")

    print()


if __name__ == "__main__":
    verifier = FaceVerifier()
    if not verifier.is_loaded():
        print("FATAL: YuNet model not loaded.")
        sys.exit(1)
    
    # Scan for images
    import glob
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        images.extend(glob.glob(os.path.join(TARGETS_DIR, ext)))
    
    if not images:
        print(f"No images found in {TARGETS_DIR}")
        print(f"Please drop test images there and run again.")
    else:
        for img in sorted(images):
            diagnose_image(img, verifier)
    
    print("\n--- DIAGNOSTIC COMPLETE ---")
