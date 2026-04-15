# Face Detection Benchmarking

> **Phase 3** — Compares four face detection models to select the fastest and most reliable detector for real-time face verification on torso crops.

---

## Purpose

After the watchdog saves a torso crop, we need to confirm that a face is actually present before expensive preprocessing begins. This benchmark tests four detectors on actual captured torso images to find the one with the best speed-to-accuracy ratio.

---

## Models Benchmarked

| Model | Library | Speed | Notes |
|-------|---------|-------|-------|
| **Haar Cascade** | OpenCV | Fast | High false positive rate, poor on angled faces |
| **YuNet** | OpenCV DNN | Very Fast | ONNX model, excellent accuracy, minimal dependencies |
| **MTCNN** | `mtcnn` | Slow | Multi-stage CNN, accurate but heavy |
| **MediaPipe** | `mediapipe` | Moderate | Google's pipeline, good but adds large dependency |

---

## Key Script

### `benchmark_faces.py`

Loads a folder of test images and runs each through all four detectors. For every image it:
1. Runs detection and records inference time.
2. Draws bounding boxes on a 2×2 comparison grid.
3. Saves the grid to `benchmark_results/`.

At the end it prints a summary table and generates a **dual Y-axis performance graph** (FPS bar chart + success rate line) saved to `benchmark_graphs/`.

---

## Winning Selection

**YuNet** was selected for the production pipeline:

- **Speed:** ~3–5 ms per frame — 10× faster than MTCNN.
- **Accuracy:** 0.7 score threshold eliminates virtually all false positives.
- **Dependencies:** Ships as a single `.onnx` file — no TensorFlow or PyTorch required.
- **Landmark Output:** Returns 5 facial landmarks (eyes, nose, mouth corners), which are essential for the affine alignment step in Phase 4.

**Technical Justification:** In a real-time system where the face check runs on every trigger event, latency matters more than marginal accuracy gains. YuNet's sub-5ms inference makes it the clear winner.

---

## Folder Structure

```
face_detection_benchmarking/
├── benchmark_faces.py                     ← Main benchmarking script
├── face_detection_yunet_2023mar.onnx      ← YuNet ONNX weights
├── benchmark_results/                     ← 2×2 comparison grids
├── benchmark_graphs/                      ← Performance graph PNGs
└── captured_targets/                      ← Test images (torso crops)
```

---

## Requirements

```
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
mtcnn>=0.1.1
mediapipe>=0.10.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Usage

```bash
# Download YuNet model (if not present)
# https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

cd face_detection_benchmarking
python benchmark_faces.py
```
