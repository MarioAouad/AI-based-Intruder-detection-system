"""
method_auto_yolo.py  –  AI Auto-Calibrator via Triangle Similarity
====================================================================

THEORY: Triangle Similarity vs Pinhole Model
----------------------------------------------
The full Pinhole Camera Model (used in method_checkerboard.py) solves for the
complete 3×3 intrinsic matrix K using a calibration pattern and many views.

Triangle Similarity is a simplified, single-measurement shortcut:

    Known:
        W  = real-world width of an object  (e.g. 45 cm torso)
        D  = known distance to that object  (e.g. 100 cm = 1 m)
        P  = pixel width of that object in the image at distance D

    Derived:
        F  = (P × D) / W          ← "apparent" focal length in pixels

    This F approximates f_x from the full pinhole model along the
    horizontal axis.  We then construct a synthetic intrinsic matrix:

        K = ┌  F    0   c_x ┐
            │  0    F   c_y │
            └  0    0    1  ┘

    where c_x, c_y = image centre (width/2, height/2).

    Advantage : no calibration pattern needed — just a person at a known distance.
    Limitation: ignores lens distortion, assumes square pixels, and only
                estimates ONE focal length (assumes f_x ≈ f_y).

THIS SCRIPT
-----------
1. Opens your webcam and runs YOLOv8-Small to detect a person (class 0).
2. You stand exactly 100 cm (1 m) from the camera.
3. Press 'c' — it reads your bounding box pixel width, applies Triangle
   Similarity, and prints the synthesised 3×3 Camera Intrinsic Matrix.

Controls:
    C  –  capture & calculate
    Q  –  quit
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS      = "yolov8s.pt"
CONF_THRESHOLD     = 0.70
TARGET_CLASS       = 0              # COCO 'person'
DEVICE             = "cuda"         # "cpu" if no GPU

# Known physical parameters for the Triangle Similarity equation.
KNOWN_DISTANCE_CM  = 100.0          # Person stands 1 m from camera
REAL_TORSO_WIDTH_CM = 45.0          # Average adult shoulder width


def main():
    # ── Load model ────────────────────────────────────────────────────
    print("[AutoCalib] Loading YOLOv8-Small …")
    model = YOLO(MODEL_WEIGHTS)
    model.to(DEVICE)
    print(f"[AutoCalib] Model ready on {DEVICE.upper()}.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[AutoCalib] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│  Stand exactly 1 METRE from the camera.                  │")
    print("│  Make sure your full torso/shoulders are visible.         │")
    print("│  Press C to capture.    Q to quit.                       │")
    print("└──────────────────────────────────────────────────────────┘\n")

    focal_length = None       # Will be calculated on capture.
    camera_matrix = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display = frame.copy()

        # ── Run YOLO detection (no tracking needed here) ──────────────
        results = model.predict(
            source=frame,
            classes=[TARGET_CLASS],
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        # Find the widest person detection (assumed to be the calibration
        # subject, closest to camera).
        best_box   = None
        best_width = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().tolist()

            for i, (x1, y1, x2, y2) in enumerate(xyxys):
                conf      = confs[i]
                box_width = x2 - x1

                # Draw every detection so the user can see what YOLO sees.
                colour = (0, 255, 0) if box_width >= best_width else (180, 180, 180)
                cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    display,
                    f"W={box_width}px  {conf:.0%}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA,
                )

                if box_width > best_width:
                    best_width = box_width
                    best_box   = (x1, y1, x2, y2)

        # ── HUD ───────────────────────────────────────────────────────
        status = (
            f"Widest person: {best_width} px"
            if best_width > 0
            else "No person detected"
        )
        cv2.putText(
            display, status,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Stand 1m away  |  C=capture  Q=quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

        if focal_length is not None:
            cv2.putText(
                display,
                f"Last f_x = {focal_length:.2f} px  (captured!)",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

        cv2.imshow("YOLO Auto-Calibrator", display)
        key = cv2.waitKey(1) & 0xFF

        # ── Capture ───────────────────────────────────────────────────
        if key == ord("c"):
            if best_width <= 0:
                print("[AutoCalib] No person detected — cannot capture.")
                continue

            # ┌────────────────────────────────────────────────────────┐
            # │  Triangle Similarity:  F = (P × D) / W               │
            # │                                                        │
            # │  P = best_width    (pixel width of the torso box)     │
            # │  D = 100 cm        (known standing distance)          │
            # │  W = 45 cm         (average shoulder width)           │
            # └────────────────────────────────────────────────────────┘
            focal_length = (best_width * KNOWN_DISTANCE_CM) / REAL_TORSO_WIDTH_CM

            # Build a synthetic 3×3 Camera Intrinsic Matrix.
            # We assume square pixels (f_x = f_y) and that the principal
            # point is at the image centre.
            c_x = w / 2.0
            c_y = h / 2.0
            camera_matrix = np.array([
                [focal_length,  0.0,           c_x],
                [0.0,           focal_length,  c_y],
                [0.0,           0.0,           1.0],
            ], dtype=np.float64)

            print("\n" + "=" * 60)
            print(" AUTO-YOLO CALIBRATION RESULT")
            print("=" * 60)
            print(f"\n  Person pixel width : {best_width} px")
            print(f"  Known distance     : {KNOWN_DISTANCE_CM} cm")
            print(f"  Real torso width   : {REAL_TORSO_WIDTH_CM} cm")
            print(f"  Calculated f_x     : {focal_length:.2f} px")
            print(f"\n  Camera Intrinsic Matrix (K):")
            print(repr(camera_matrix))
            print(f"\n  f_x = f_y = {focal_length:.2f} px")
            print(f"  c_x = {c_x:.2f} px")
            print(f"  c_y = {c_y:.2f} px")
            print("\n" + "=" * 60)
            print("Copy the matrix above into compare_distances.py  →  MATRIX_METHOD_AUTO_YOLO")
            print("=" * 60 + "\n")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[AutoCalib] Done.")


if __name__ == "__main__":
    main()
