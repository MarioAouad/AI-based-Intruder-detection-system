"""
benchmark_logger.py
-------------------
Silent telemetry collector that runs alongside video_tracker.py.

Metrics captured per run
------------------------
  - Average FPS over the entire video
  - Average confidence score of detected persons
  - Unique tracking ID count (proxy for ID switches / tracking drops)
  - Peak GPU VRAM usage in MiB (via pynvml with torch fallback)
  - Total frames processed
"""

import csv
import os
import time
from typing import Optional, List

import torch

# pynvml gives fine-grained NVIDIA GPU stats.
# Falls back to torch.cuda memory stats when pynvml is absent.
try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False
    print("[BenchmarkLogger] pynvml not found – falling back to torch CUDA memory stats.")


class BenchmarkLogger:
    """
    Collects per-frame statistics during a single model's video run and
    exports them to a shared CSV file once the run is complete.

    Usage
    -----
    logger = BenchmarkLogger(model_name="YOLOv8-Nano", gpu_index=0)
    logger.start_run()
    for frame in frames:
        ...inference...
        logger.log_frame(fps=30.1, confidences=[0.87, 0.91], track_ids=[1, 2])
    row = logger.end_run()
    logger.export_to_csv(row, output_csv_path)
    """

    def __init__(self, model_name: str, gpu_index: int = 0):
        self.model_name = model_name
        self.gpu_index  = gpu_index

        # Per-frame accumulators (reset by start_run)
        self._fps_samples:    List[float] = []
        self._conf_samples:   List[float] = []
        self._all_track_ids:  set         = set()
        self._frame_count:    int         = 0

        # Timing for the overall run
        self._run_start_time: Optional[float] = None

        # VRAM peak tracking
        self._peak_vram_mib: float = 0.0
        self._nvml_handle           = None

        # Initialise pynvml if available so we can query the correct GPU.
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except pynvml.NVMLError as e:
                print(f"[BenchmarkLogger] pynvml init failed: {e}. Falling back to torch stats.")
                self._nvml_handle = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Reset all accumulators and begin wall-clock timing."""
        self._fps_samples   = []
        self._conf_samples  = []
        self._all_track_ids = set()
        self._frame_count   = 0
        self._peak_vram_mib = 0.0
        self._run_start_time = time.perf_counter()

        # Reset torch CUDA peak-memory counters so we get a clean run peak.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.gpu_index)

        print(f"[BenchmarkLogger] Run started for: {self.model_name}")

    def log_frame(
        self,
        fps: float,
        confidences: List[float],
        track_ids: List[int],
    ) -> None:
        """
        Called once per processed video frame.

        Parameters
        ----------
        fps          : Instantaneous FPS for this frame.
        confidences  : List of confidence scores for each detected person.
        track_ids    : List of integer tracker IDs assigned in this frame.
        """
        self._frame_count += 1
        self._fps_samples.append(fps)

        # Accumulate confidence scores.
        self._conf_samples.extend(confidences)

        # Every unique ID seen across the whole video is logged.
        # In a perfect run the set size == number of distinct people.
        # ID switches inflate this count beyond the true people count.
        self._all_track_ids.update(track_ids)

        # Sample VRAM after each frame for accurate peak measurement.
        self._peak_vram_mib = max(self._peak_vram_mib, self._sample_vram_mib())

    def end_run(self) -> dict:
        """
        Finalise the run, compute aggregate statistics, and return them
        as a dictionary ready for CSV export.
        """
        elapsed = time.perf_counter() - self._run_start_time

        avg_fps   = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0.0
        avg_conf  = sum(self._conf_samples) / len(self._conf_samples) if self._conf_samples else 0.0
        unique_ids = len(self._all_track_ids)

        # If pynvml path failed, query torch as a final safety net.
        if self._peak_vram_mib == 0.0 and torch.cuda.is_available():
            self._peak_vram_mib = torch.cuda.max_memory_allocated(device=self.gpu_index) / (1024 ** 2)

        stats = {
            "model_name":        self.model_name,
            "frames_processed":  self._frame_count,
            "total_time_sec":    round(elapsed, 2),
            "avg_fps":           round(avg_fps, 2),
            "avg_confidence":    round(avg_conf, 4),
            "unique_track_ids":  unique_ids,   # Low = stable tracking; high = many ID switches
            "peak_vram_mib":     round(self._peak_vram_mib, 1),
        }
        print(f"[BenchmarkLogger] Run ended  → avg_fps={stats['avg_fps']}, "
              f"peak_vram={stats['peak_vram_mib']} MiB, unique_ids={unique_ids}")
        return stats

    def export_to_csv(self, stats: dict, csv_path: str) -> None:
        """
        Append a single model's stats as one row to the CSV.
        Creates the file and writes the header if it does not yet exist.
        """
        file_exists = os.path.isfile(csv_path)
        fieldnames  = list(stats.keys())

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

        print(f"[BenchmarkLogger] Metrics appended to: {csv_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_vram_mib(self) -> float:
        """
        Return the current GPU VRAM usage in MiB.

        Priority:
          1. pynvml (whole-GPU view, includes drivers + other processes)
          2. torch.cuda.memory_allocated (only this process's tensors – lighter)
        """
        if self._nvml_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                # mem_info.used is in bytes → convert to MiB
                return mem_info.used / (1024 ** 2)
            except pynvml.NVMLError:
                pass  # Graceful degradation to torch path

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device=self.gpu_index) / (1024 ** 2)

        return 0.0

    def __del__(self):
        """Cleanly shut down pynvml when the logger is garbage-collected."""
        if _PYNVML_AVAILABLE and self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
