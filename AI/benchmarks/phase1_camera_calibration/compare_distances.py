"""
compare_distances.py  –  Live A/B Distance Comparison Sandbox
==============================================================

PURPOSE
-------
This script lets you physically walk around in front of your webcam and see
**both calibration methods' distance estimates side-by-side** in real time.

HOW IT WORKS
------------
Both calibration methods ultimately give us one key number: the camera's
focal length in pixels (f_x).  Using that focal length we apply the same
underlying distance formula:

        distance_cm = (REAL_TORSO_WIDTH_CM × f_x) / pixel_width_of_person
        distance_m  = distance_cm / 100

The difference between the two methods is *how* f_x was obtained:

    ┌─────────────────┬──────────────────────────────────────────────────┐
    │ Checkerboard    │ f_x comes from the full Pinhole Camera Model    │
    │                 │ solved by OpenCV's Zhang-method calibration.     │
    │                 │ Accounts for lens distortion.  More accurate.   │
    ├─────────────────┼──────────────────────────────────────────────────┤
    │ Auto-YOLO       │ f_x is approximated via Triangle Similarity     │
    │                 │ from a single measurement (person at 1 m).      │
    │                 │ Ignores distortion.  Quick & easy.              │
    └─────────────────┴──────────────────────────────────────────────────┘

SETUP
-----
1. Run method_checkerboard.py → copy the printed 3×3 matrix below.
2. Run method_auto_yolo.py    → copy the printed 3×3 matrix below.
3. Run this script.  Press '1' or '2' to toggle which method is active.

Controls:
    1  –  use Checkerboard matrix
    2  –  use Auto-YOLO matrix
    Q  –  quit
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────
# Model & detection settings
# ──────────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS       = "yolov8s.pt"
CONF_THRESHOLD      = 0.70
TARGET_CLASS        = 0              # 'person'
DEVICE              = "cuda"
TRACKER_CONFIG      = "bytetrack.yaml"
REAL_TORSO_WIDTH_CM = 49.5           # Same constant used in both methods

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PASTE YOUR CALIBRATION MATRICES HERE                                  ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  Replace the dummy np.eye(3) matrices with the real output from        ║
# ║  method_checkerboard.py and method_auto_yolo.py.                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# From method_checkerboard.py — the full OpenCV calibration result.
MATRIX_METHOD_CHECKERBOARD = np.array([
    [882.89978178,   0.        , 660.50837331],
    [  0.        , 885.05049294, 363.29063902],
    [  0.        ,   0.        ,   1.        ]
], dtype=np.float64)

# From method_auto_yolo.py — the Triangle Similarity approximation.
MATRIX_METHOD_AUTO_YOLO = np.array([
    [1108.9,      0.0,    640.0],
    [   0.0,   1108.9,    360.0],
    [   0.0,      0.0,      1.0]
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────
# Distance calculation
# ──────────────────────────────────────────────────────────────────────────

def estimate_distance(pixel_width: int, camera_matrix: np.ndarray) -> float | None:
    """
    Estimate the distance of a person from the camera using a given
    intrinsic matrix.

    Formula (Triangle Similarity rearranged from the Pinhole Model):

        f_x = camera_matrix[0, 0]
        distance_cm = (REAL_TORSO_WIDTH_CM × f_x) / pixel_width
        distance_m  = distance_cm / 100

    Returns distance in metres, or None if inputs are degenerate.
    """
    if pixel_width <= 0:
        return None

    f_x = camera_matrix[0, 0]   # Focal length in pixels (horizontal)
    if f_x <= 1.0:
        return None  # Matrix is still the dummy identity

    distance_cm = (REAL_TORSO_WIDTH_CM * f_x) / pixel_width
    return round(distance_cm / 100.0, 2)


# ──────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("[Compare] Loading YOLOv8-Small …")
    model = YOLO(MODEL_WEIGHTS)
    model.to(DEVICE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Compare] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Method selection state ─────────────────────────────────────────
    # 1 = Checkerboard,  2 = Auto-YOLO
    active_method = 1
    method_names  = {1: "Checkerboard (OpenCV)", 2: "Auto-YOLO (Triangle Sim.)"}
    method_mats   = {1: MATRIX_METHOD_CHECKERBOARD, 2: MATRIX_METHOD_AUTO_YOLO}
    method_colors = {1: (255, 200, 0), 2: (0, 220, 0)}   # Cyan / Green

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│  Press 1 = Checkerboard matrix   2 = Auto-YOLO matrix   │")
    print("│  Walk around and compare the distance reading!           │")
    print("│  Q = quit                                                │")
    print("└──────────────────────────────────────────────────────────┘\n")

    dummy_warning_shown = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cam_mtx = method_mats[active_method]

        # Warn once if matrix is still the dummy identity.
        if cam_mtx[0, 0] <= 1.0 and not dummy_warning_shown:
            print(
                "[Compare] WARNING: The active matrix is still the dummy identity.\n"
                "  Paste your real calibration matrix at the top of this script."
            )
            dummy_warning_shown = True

        # ── Run YOLO with tracking for persistent IDs ─────────────────
        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[TARGET_CLASS],
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().tolist()
            ids   = boxes.id

            for i, (x1, y1, x2, y2) in enumerate(xyxys):
                conf     = confs[i]
                track_id = int(ids[i].item()) if (ids is not None and ids[i] is not None) else -1

                pixel_w  = x2 - x1
                dist_m   = estimate_distance(pixel_w, cam_mtx)
                dist_txt = f"{dist_m:.2f} m" if dist_m is not None else "-- m"

                colour = method_colors[active_method]
                cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)

                label = f"ID:{track_id}  {dist_txt}  W={pixel_w}px"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1)
                cv2.putText(
                    display, label,
                    (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )

        # ── HUD: active method + helper ───────────────────────────────
        hud_color = method_colors[active_method]

        cv2.putText(
            display,
            f"Active: [{active_method}] {method_names[active_method]}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            hud_color, 2, cv2.LINE_AA,
        )

        f_x_val = cam_mtx[0, 0]
        cv2.putText(
            display,
            f"f_x = {f_x_val:.2f} px" if f_x_val > 1.0 else "f_x = NOT SET (dummy matrix)",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            hud_color, 1, cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "Press 1=Checkerboard  2=AutoYOLO  Q=Quit",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow("Distance Comparison – Checkerboard vs Auto-YOLO", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            active_method = 1
            dummy_warning_shown = False
            print(f"[Compare] Switched to: {method_names[1]}")
        elif key == ord("2"):
            active_method = 2
            dummy_warning_shown = False
            print(f"[Compare] Switched to: {method_names[2]}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Compare] Done.")


if __name__ == "__main__":
    main()
