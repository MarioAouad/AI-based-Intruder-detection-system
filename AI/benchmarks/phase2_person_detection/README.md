# Person Detection Benchmarking

> **Phase 2** — Performance comparison of YOLOv8, YOLO11, and RT-DETR models for real-time person detection. Measures FPS throughput vs. detection accuracy (mAP) to select the optimal model for the intruder detection pipeline.

---

## Purpose

The main watchdog system requires a detector that is:
1. **Fast enough** for real-time webcam processing (≥ 15 FPS).
2. **Accurate enough** to avoid false detections that would trigger unnecessary alerts.

This benchmark runs each model variant on a standardised test video and logs per-frame metrics.

---

## Models Tested

| Model | Size | Architecture | Weights File |
|-------|------|-------------|-------------|
| YOLOv8n | Nano | YOLOv8 | `yolov8n.pt` |
| YOLOv8s | Small | YOLOv8 | `yolov8s.pt` |
| YOLOv8m | Medium | YOLOv8 | `yolov8m.pt` |
| YOLO11n | Nano | YOLO11 | `yolo11n.pt` |
| YOLO11s | Small | YOLO11 | `yolo11s.pt` |
| YOLO11m | Medium | YOLO11 | `yolo11m.pt` |
| RT-DETR-l | Large | Transformer | `rtdetr-l.pt` |

---

## Key Components

### `config.py`
Model list, weight paths (resolved relative to `BASE_DIR`), and benchmark settings. Paths use `os.path.join(BASE_DIR, "filename.pt")` for portability.

### `main_benchmark.py`
Entry point. Iterates over the model list, runs inference on the test video, and collects FPS + detection counts.

### `video_tracker.py`
Video processing loop with ByteTrack integration. Handles frame-by-frame inference, bounding box rendering, and optional output video saving.

### `benchmark_logger.py`
CSV logger that records per-model metrics. Generates summary tables and comparison graphs.

---

## Winning Selection

**YOLOv8s** was selected as the production model:
- **FPS:** ~25–30 on CUDA GPU — sufficient for real-time operation.
- **Accuracy:** Higher mAP than nano variants, with far less latency than RT-DETR.
- **Trade-off:** Best balance of speed and accuracy for the 2-metre zone detection use case.

---

## Folder Structure

```
PersonDetection_Benchmarking/
├── config.py                  ← Model list & benchmark settings
├── main_benchmark.py          ← Entry point
├── video_tracker.py           ← Frame-by-frame inference loop
├── benchmark_logger.py        ← CSV logging & graph generation
├── benchmark_results.csv      ← Raw benchmark data
├── requirements.txt           ← Local dependencies
├── test_video.mp4             ← Standard test footage
├── output_videos/             ← Annotated output videos
└── *.pt                       ← Model weight files (gitignored)
```

---

## Usage

```bash
cd PersonDetection_Benchmarking
pip install -r requirements.txt
python main_benchmark.py
```
