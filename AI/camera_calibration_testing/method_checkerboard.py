"""
method_checkerboard.py  –  OpenCV Standard Checkerboard Calibration
====================================================================

THEORY: The Pinhole Camera Model
----------------------------------
A perfect pinhole camera maps 3D world points (X, Y, Z) to 2D image points
(u, v) through a 3×3 intrinsic matrix K:

        ┌ u ┐       ┌ f_x   0   c_x ┐   ┌ X ┐
    s · │ v │  =    │  0   f_y  c_y │ × │ Y │
        └ 1 ┘       └  0    0    1  ┘   └ Z ┘

    f_x, f_y  = focal lengths in pixels (horizontal / vertical)
    c_x, c_y  = principal point (optical centre, ideally image centre)
    s         = scaling factor (usually absorbed)

Real lenses also introduce radial and tangential distortion.  OpenCV
parameterises these as the distortion coefficient vector:

    dist_coeffs = [k1, k2, p1, p2, k3]

By showing OpenCV many views of a known-geometry planar pattern (the
checkerboard), it can solve for K and dist_coeffs using Zhang's method.

THIS SCRIPT
-----------
1. Opens your webcam and shows a live feed.
2. Press 'c' to capture a frame — it finds the 9×6 inner corners.
3. After ~15 captures, press 'q' to run calibration.
4. Prints the Camera Intrinsic Matrix and Distortion Coefficients.

IMPORTANT: You are using a checkerboard on your PHONE SCREEN.
           Measure one physical square with a ruler and fill in SQUARE_SIZE_MM.
           Watch out for screen glare — the reflections can confuse corner detection!

Controls:
    C  –  capture current frame
    Q  –  finish capturing & run calibration
"""

import sys
import cv2
import numpy as np

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CALIBRATION PARAMETERS – YOU MUST SET THESE                           ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║                                                                        ║
# ║  Take a physical ruler, hold it next to your phone screen, and         ║
# ║  measure the side-length of ONE checkerboard square in millimetres.    ║
# ║  Enter that value here.                                                ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SQUARE_SIZE_MM = 10        # ← MEASURE & REPLACE  (e.g. 18.5)

# Convert to metres — OpenCV uses metres for the 3D object points.
SQUARE_SIZE_M = SQUARE_SIZE_MM / 1000.0

# Inner corner count of the checkerboard grid.
# A 10×7 square board has 9×6 = 54 inner corners.
CHECKERBOARD = (8, 6)

# How many good captures before we run calibration.
MIN_CAPTURES = 15

# ──────────────────────────────────────────────────────────────────────────
# Sub-pixel corner refinement criteria (termination of iterative solver).
# This is standard OpenCV practice for improved accuracy.
# ──────────────────────────────────────────────────────────────────────────
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,     # max iterations
    0.001,  # epsilon (desired accuracy)
)


def build_object_points() -> np.ndarray:
    """
    Construct the array of 3D world coordinates for each inner corner.

    We assume the checkerboard lies on the Z=0 plane, so each point is:
        (col * SQUARE_SIZE_M, row * SQUARE_SIZE_M, 0)

    Returns shape (CHECKERBOARD[0]*CHECKERBOARD[1], 3) of float32.
    """
    cols, rows = CHECKERBOARD
    objp = np.zeros((cols * rows, 3), np.float32)

    # np.mgrid returns coordinate matrices; reshape to (N, 2).
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Scale from "grid units" to real-world metres.
    objp *= SQUARE_SIZE_M
    return objp


def run_capture_loop(cap: cv2.VideoCapture):
    """
    Show the live feed. On 'c', attempt to find checkerboard corners.
    Returns (object_points_list, image_points_list, image_size).
    """
    objp           = build_object_points()
    obj_points_3d  = []   # list of objp copies (one per good capture)
    img_points_2d  = []   # list of detected corner arrays
    image_size     = None
    capture_count  = 0

    print(f"\n[Checkerboard] Looking for a {CHECKERBOARD[0]}×{CHECKERBOARD[1]} inner-corner grid.")
    print(f"[Checkerboard] Capture at least {MIN_CAPTURES} frames, then press Q.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Checkerboard] Webcam read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # ── Glare warning ──────────────────────────────────────────────
        cv2.putText(
            display,
            "Watch out for screen glare!",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Captures: {capture_count}/{MIN_CAPTURES}  |  C=capture  Q=calibrate",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 200, 0), 2, cv2.LINE_AA,
        )

        # Try to find corners in real-time for visual feedback.
        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            # Refine corner positions to sub-pixel accuracy.
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), CRITERIA
            )
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners_refined, found)

        cv2.imshow("Checkerboard Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if not found:
                print("[Checkerboard] No corners detected in this frame. "
                      "Tilt phone / reduce glare and try again.")
                continue

            # Save the 3D↔2D correspondence pair for this view.
            obj_points_3d.append(objp)
            img_points_2d.append(corners_refined)
            image_size = gray.shape[::-1]   # (width, height)
            capture_count += 1
            print(f"[Checkerboard] ✓ Captured frame {capture_count}")

        elif key == ord("q"):
            break

    return obj_points_3d, img_points_2d, image_size, capture_count


def calibrate_camera(obj_points_3d, img_points_2d, image_size):
    """
    Run OpenCV's camera calibration solver (Zhang's method).

    Returns (camera_matrix, dist_coeffs, rvecs, tvecs, rms_error).
    """
    print("\n[Checkerboard] Running calibration solver …")

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_3d,
        img_points_2d,
        image_size,
        None,          # initial camera matrix guess (None = auto)
        None,          # initial distortion guess
    )
    return camera_matrix, dist_coeffs, rvecs, tvecs, rms_error


def main():
    # ── Safety check ──────────────────────────────────────────────────
    if SQUARE_SIZE_MM <= 0:
        print(
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  ERROR: SQUARE_SIZE_MM is not set!                      ║\n"
            "║                                                         ║\n"
            "║  Open this file, measure one checkerboard square on     ║\n"
            "║  your phone screen with a physical ruler, and enter     ║\n"
            "║  the size in millimetres at the top of the script.      ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Checkerboard] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    obj_pts, img_pts, img_size, count = run_capture_loop(cap)
    cap.release()
    cv2.destroyAllWindows()

    if count < 3:
        print(f"[Checkerboard] Only {count} captures — need at least 3. Aborting.")
        sys.exit(1)

    if count < MIN_CAPTURES:
        print(f"[Checkerboard] Warning: only {count}/{MIN_CAPTURES} captures. "
              "Accuracy may suffer.\n")

    # ── Calibrate ──────────────────────────────────────────────────────
    cam_mtx, dist, rvecs, tvecs, rms = calibrate_camera(obj_pts, img_pts, img_size)

    # ── Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" CHECKERBOARD CALIBRATION RESULTS")
    print("=" * 60)
    print(f"\nRMS Re-projection Error: {rms:.4f}  (lower = better, <0.5 is great)\n")
    print("Camera Intrinsic Matrix (K):")
    print(repr(cam_mtx))
    print(f"\n  f_x = {cam_mtx[0, 0]:.2f} px")
    print(f"  f_y = {cam_mtx[1, 1]:.2f} px")
    print(f"  c_x = {cam_mtx[0, 2]:.2f} px")
    print(f"  c_y = {cam_mtx[1, 2]:.2f} px")
    print("\nDistortion Coefficients [k1, k2, p1, p2, k3]:")
    print(repr(dist))
    print("\n" + "=" * 60)
    print("Copy the matrix above into compare_distances.py  →  MATRIX_METHOD_CHECKERBOARD")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
