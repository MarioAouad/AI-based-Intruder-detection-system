"""
main_watchdog.py  –  Phase 2 Intruder Detection Orchestrator
=============================================================

Ties together:
  ┌──────────────────────────────────────────────────────────┐
  │  YOLOv8-Small  →  ByteTrack  →  DistanceEstimator        │
  │                                        ↓                  │
  │                               ZoneTimer (per ID)          │
  │                                        ↓                  │
  │                          TRIGGER → save crop + alert      │
  └──────────────────────────────────────────────────────────┘

Run:
    python main_watchdog.py

Controls:
    Q  –  quit the live window

Calibration
-----------
Distance estimation uses the Camera Intrinsic Matrix stored in config.py
(CAMERA_MATRIX).  The focal length f_x is extracted at runtime by
spatial_math.DistanceEstimator.  To re-calibrate, run
camera_calibration_testing/method_auto_yolo.py and paste the output matrix
into config.py → CAMERA_MATRIX.
"""

import os
import time
import datetime
import argparse

import cv2
import torch
from ultralytics import YOLO

import config
from spatial_math   import DistanceEstimator
from threat_timer   import ZoneTimer


# ---------------------------------------------------------------------------
# Helper: draw one person's annotation on the frame
# ---------------------------------------------------------------------------

def _annotate_person(
    frame,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    distance_m,          # float | None
    elapsed_s: float,
    triggered: bool,
    in_zone: bool,
) -> None:
    """
    Draw a bounding box and a multi-line info label for a single detected person.

    Colour logic
    ------------
    Green  → outside the 2 m warning zone (or distance unknown)
    Red    → inside the 2 m zone (countdown running)
    Orange → countdown has just fired (TRIGGER)
    """
    if triggered:
        colour = config.COLOR_TRIGGER
    elif in_zone:
        colour = config.COLOR_THREAT
    else:
        colour = config.COLOR_SAFE

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Build label lines
    dist_txt    = f"{distance_m:.2f} m" if distance_m is not None else "-- m (uncal.)"
    timer_txt   = f"Zone: {elapsed_s:.1f}s / {config.TRIGGER_TIME_SECONDS:.0f}s" if in_zone else ""
    trigger_txt = "!! TRIGGER !!" if triggered else ""

    lines = [f"ID:{track_id}  {dist_txt}"]
    if timer_txt:
        lines.append(timer_txt)
    if trigger_txt:
        lines.append(trigger_txt)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    fscale     = config.FONT_SCALE
    fthick     = config.FONT_THICKNESS
    line_h     = 18
    padding    = 4
    max_w      = max(cv2.getTextSize(l, font, fscale, fthick)[0][0] for l in lines)
    box_top    = y1 - len(lines) * line_h - padding * 2
    box_top    = max(box_top, 0)

    # Filled background rectangle behind all label lines
    cv2.rectangle(
        frame,
        (x1, box_top),
        (x1 + max_w + padding * 2, y1),
        colour, -1
    )

    for idx, line in enumerate(lines):
        ty = box_top + padding + (idx + 1) * line_h - 2
        cv2.putText(
            frame, line,
            (x1 + padding, ty),
            font, fscale, (255, 255, 255), fthick, cv2.LINE_AA
        )


# ---------------------------------------------------------------------------
# Helper: save a cropped image of the triggered person
# ---------------------------------------------------------------------------

def _save_crop(frame, x1: int, y1: int, x2: int, y2: int, track_id: int, property_id: int) -> str:
    """
    Crop the bounding box region from the frame and save it to TARGETS_DIR.
    Returns the full path of the saved file.
    """
    h, w = frame.shape[:2]
    # Clamp coordinates so we never exceed frame bounds
    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(w, x2);  y2c = min(h, y2)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return ""

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"target_p{property_id}_id{track_id}_{ts}.jpg"
    filepath = os.path.join(config.TARGETS_DIR, filename)
    cv2.imwrite(filepath, crop)
    return filepath


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_watchdog(property_id: int = 1, video_source=0) -> None:
    """
    Entry point.  Opens the webcam, loads the model, and runs the
    detection / tracking / alerting loop until the user presses Q.
    """

    # ── 1. Load YOLOv8-Small onto GPU ─────────────────────────────────────
    print("[Watchdog] Loading YOLOv8-Small …")
    model = YOLO(config.MODEL_WEIGHTS)
    model.to(config.DEVICE)
    print(f"[Watchdog] Model ready on {config.DEVICE.upper()}.")

    # ── 2. Initialise distance estimator ──────────────────────────────────
    estimator = DistanceEstimator()   # reads config constants internally

    # ── 3. Initialise zone timer ───────────────────────────────────────────
    zone_timer = ZoneTimer()

    # ── 4. Open webcam ────────────────────────────────────────────────────
    print(f"[Watchdog] Opening video source (index {video_source}) …")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[Watchdog] ERROR: Cannot open webcam. Exiting.")
        return

    # Optional: request a higher resolution from the camera driver.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Watchdog] Running.  Press Q in the video window to quit.\n")

    # FPS display
    fps_timer      = time.perf_counter()
    display_fps    = 0.0
    frame_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Watchdog] Failed to read from webcam. Exiting.")
            break

        # ── 5. Run track() on the current frame ───────────────────────────
        # persist=True is essential: it tells ByteTrack to carry its
        # internal state (Kalman filters, ID assignments) across frames,
        # giving each person a stable integer ID throughout the session.
        results = model.track(
            source=frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[config.TARGET_CLASS],
            conf=config.CONF_THRESHOLD,
            device=config.DEVICE,
            verbose=False,
        )

        active_ids = set()   # IDs seen this frame; used to purge stale timers

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().tolist()
            ids   = boxes.id   # None if tracker hasn't assigned IDs yet

            for i, (x1, y1, x2, y2) in enumerate(xyxys):
                conf     = confs[i]
                track_id = int(ids[i].item()) if (ids is not None and ids[i] is not None) else -1

                if track_id < 0:
                    continue   # Skip detections without a confirmed track ID

                active_ids.add(track_id)

                # ── 6. Calculate distance ──────────────────────────────────
                pixel_width = x2 - x1     # Width of bounding box in pixels
                distance_m  = estimator.calculate_distance(pixel_width)

                # ── 7. Update zone timer ───────────────────────────────────
                triggered, elapsed_s = zone_timer.update(track_id, distance_m)

                in_zone = (
                    distance_m is not None
                    and distance_m <= config.ZONE_RADIUS_METERS
                )

                # ── 8. Handle TRIGGER ──────────────────────────────────────
                if triggered:
                    # ---- Console alert ----
                    print(
                        f"\n{'!'*60}\n"
                        f"  *** TARGET ACQUIRED: ID {track_id} ***\n"
                        f"  Distance : {distance_m:.2f} m\n"
                        f"  In zone  : {elapsed_s:.1f} s\n"
                        f"{'!'*60}\n"
                    )

                    # ---- Save cropped bounding box to disk ----
                    saved_path = _save_crop(frame, x1, y1, x2, y2, track_id, property_id)
                    if saved_path:
                        print(f"[Watchdog] Saved crop → {saved_path}")

                    # ---- MUST reset timer immediately so we don't spam saves ----
                    # The ID will need another full 30 s in zone before re-triggering.
                    zone_timer.reset(track_id)

                # ── 9. Draw annotation on frame ────────────────────────────
                _annotate_person(
                    frame,
                    x1, y1, x2, y2,
                    track_id,
                    distance_m,
                    elapsed_s,
                    triggered,
                    in_zone,
                )

        # ── 10. Purge timers for IDs that vanished this frame ─────────────
        zone_timer.purge_stale(active_ids)

        # ── 11. FPS counter overlay ─────────────────────────────────────────
        frame_count += 1
        now = time.perf_counter()
        if now - fps_timer >= 0.5:                    # update twice a second
            display_fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer   = now

        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f}  |  Zone active: {zone_timer.active_count()}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 0), 2, cv2.LINE_AA,
        )

        # Calibration banner if not yet set up
        if not estimator.is_calibrated():
            cv2.putText(
                frame,
                "DISTANCE UNCALIBRATED  –  set CAMERA_MATRIX in config.py",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 100, 255), 2, cv2.LINE_AA,
            )

        # ── 12. Display ────────────────────────────────────────────────────
        cv2.imshow("Watchdog – Intruder Detection (Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Watchdog] User requested quit.")
            break

    # ── Clean-up ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Watchdog] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watchdog Edge Node")
    parser.add_argument("--property", type=int, required=True, help="Property ID for tagging saved frames")
    parser.add_argument("--camera", type=str, default="0", help="Video source (e.g. '0' for webcam, or video path)")
    args = parser.parse_args()

    # Parse numerical strings to integers for physical webcams
    try:
        parsed_camera_source = int(args.camera)
    except ValueError:
        parsed_camera_source = args.camera

    run_watchdog(property_id=args.property, video_source=parsed_camera_source)
