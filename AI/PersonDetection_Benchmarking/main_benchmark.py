"""
main_benchmark.py
-----------------
Orchestrator that iterates through every model in config.MODELS, runs the
test video through VideoTracker, collects metrics with BenchmarkLogger, and
finally prints a summary table plus writes all results to a CSV.

Flow
----
  for each model in config.MODELS:
      1. Instantiate VideoTracker  (loads model onto GPU)
      2. Instantiate BenchmarkLogger
      3. Open test video with cv2.VideoCapture
      4. Frame loop → VideoTracker.process_frame() → BenchmarkLogger.log_frame()
      5. BenchmarkLogger.end_run() → store stats dict
      6. logger.export_to_csv()
      7. tracker.release()        (free GPU VRAM)
      8. torch.cuda.empty_cache() (paranoid clean-up between models)
  Print summary table with tabulate
"""

import os
import sys
import gc

import cv2
import torch

# ── Local modules ──────────────────────────────────────────────────────────
import config
from video_tracker   import VideoTracker
from benchmark_logger import BenchmarkLogger


# ---------------------------------------------------------------------------
# Optional: tabulate gives prettier console tables.
# Falls back to a simple ASCII table if not installed.
# ---------------------------------------------------------------------------
try:
    from tabulate import tabulate
    _TABULATE_AVAILABLE = True
except ImportError:
    _TABULATE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _check_video(path: str) -> None:
    """Abort early with a helpful message if the test video is missing."""
    if not os.path.isfile(path):
        print(f"\n[ERROR] Test video not found: {path}")
        print("Please place a 60-second video at this path and re-run.\n")
        sys.exit(1)


def _print_summary(all_stats: list) -> None:
    """Print a nicely formatted summary table to stdout."""
    if not all_stats:
        print("No results to display.")
        return

    headers = [
        "Model", "Frames", "Time(s)", "Avg FPS",
        "Avg Conf", "Unique IDs", "Peak VRAM (MiB)"
    ]
    rows = [
        [
            s["model_name"],
            s["frames_processed"],
            s["total_time_sec"],
            s["avg_fps"],
            f"{s['avg_confidence']:.2%}",
            s["unique_track_ids"],
            s["peak_vram_mib"],
        ]
        for s in all_stats
    ]

    print("\n" + "=" * config.CONSOLE_TABLE_WIDTH)
    print(" BENCHMARK SUMMARY ".center(config.CONSOLE_TABLE_WIDTH, "="))
    print("=" * config.CONSOLE_TABLE_WIDTH)

    if _TABULATE_AVAILABLE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline",
                       stralign="left", numalign="right"))
    else:
        # Fallback: minimal ASCII table
        col_w = [max(len(str(headers[i])), max(len(str(r[i])) for r in rows)) + 2
                 for i in range(len(headers))]
        separator = "+" + "+".join("-" * w for w in col_w) + "+"
        fmt_row   = lambda row: "|" + "|".join(
            f" {str(v):<{w - 1}}" for v, w in zip(row, col_w)
        ) + "|"
        print(separator)
        print(fmt_row(headers))
        print(separator)
        for row in rows:
            print(fmt_row(row))
        print(separator)

    print(f"\nDetailed results saved to: {config.OUTPUT_CSV}\n")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """
    Entry point. Iterates all models defined in config.MODELS,
    processes the test video through each, and exports results to CSV.
    """
    _check_video(config.VIDEO_PATH)

    # Ensure output video directory exists (optional annotated video saving).
    os.makedirs(config.OUTPUT_VIDEO_DIR, exist_ok=True)

    # Remove stale CSV so we start fresh on each full benchmark run.
    if os.path.isfile(config.OUTPUT_CSV):
        os.remove(config.OUTPUT_CSV)
        print(f"[Orchestrator] Cleared existing CSV: {config.OUTPUT_CSV}")

    all_stats = []

    for model_idx, model_cfg in enumerate(config.MODELS):
        model_name = model_cfg["name"]
        print("\n" + "-" * 70)
        print(f"[Orchestrator] ({model_idx + 1}/{len(config.MODELS)}) "
              f"Starting run for: {model_name}")
        print("-" * 70)

        # ── 1. Load model ────────────────────────────────────────────────
        try:
            tracker = VideoTracker(model_cfg)
        except Exception as exc:
            print(f"[Orchestrator] Failed to load {model_name}: {exc}. Skipping.")
            continue

        # ── 2. Prepare logger ────────────────────────────────────────────
        logger = BenchmarkLogger(model_name=model_name, gpu_index=0)

        # ── 3. Open video ────────────────────────────────────────────────
        cap = cv2.VideoCapture(config.VIDEO_PATH)
        if not cap.isOpened():
            print(f"[Orchestrator] Cannot open video: {config.VIDEO_PATH}. Skipping.")
            tracker.release()
            continue

        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Orchestrator] Video: {frame_w}x{frame_h} @ {fps_source:.1f} fps, "
              f"{total_frames} frames")

        # Optionally write an annotated output video.
        safe_name  = model_name.replace(" ", "_").replace("/", "-")
        out_path   = os.path.join(config.OUTPUT_VIDEO_DIR, f"{safe_name}.mp4")
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        writer     = cv2.VideoWriter(out_path, fourcc, fps_source, (frame_w, frame_h))

        # ── 4. Frame loop ────────────────────────────────────────────────
        logger.start_run()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break   # End of video

            # Run detection + tracking; returns annotated frame and metrics.
            annotated_frame, fps, confidences, track_ids = tracker.process_frame(frame)

            # Feed raw metrics into the logger (no display logic here).
            logger.log_frame(fps=fps, confidences=confidences, track_ids=track_ids)

            # Write the annotated frame to the output video file.
            writer.write(annotated_frame)

            frame_idx += 1
            # Progress indicator every 100 frames.
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[Orchestrator]   Frame {frame_idx}/{total_frames} "
                      f"({progress:.0f}%) – FPS: {fps:.1f}")

        cap.release()
        writer.release()

        # ── 5. Finalise stats ────────────────────────────────────────────
        stats = logger.end_run()
        all_stats.append(stats)

        # ── 6. Export row to CSV ─────────────────────────────────────────
        logger.export_to_csv(stats, config.OUTPUT_CSV)

        # ── 7. Release model from VRAM ───────────────────────────────────
        tracker.release()

        # ── 8. Aggressive GPU memory clean-up between models ─────────────
        # del + gc.collect() ensures Python's reference counter drops the
        # tensors before we call torch.cuda.empty_cache(), which only frees
        # memory that has no remaining Python references.
        del tracker, logger
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()   # Wait for all CUDA kernels to finish.
        print(f"[Orchestrator] GPU memory cleared after {model_name}.")

    # ── Final summary ────────────────────────────────────────────────────────
    _print_summary(all_stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_benchmark()
