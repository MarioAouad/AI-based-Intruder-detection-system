# Camera Calibration Testing

> **Phase 1** — Determines the camera's intrinsic matrix so that pixel measurements can be converted into accurate real-world distances.

---

## Why Calibration Matters

Face recognition accuracy depends on consistent facial geometry. Lens distortion warps shapes, making bounding boxes unreliable for distance estimation. A calibrated intrinsic matrix corrects this, giving sub-10% distance error at operational ranges (1–3 m).

---

## Methods Tested

### 1. Checkerboard Pattern (`method_checkerboard.py`)

The classical approach. Uses a printed checkerboard target photographed at multiple angles. OpenCV's `findChessboardCorners()` + `calibrateCamera()` solves for the full 3×3 intrinsic matrix and distortion coefficients.

**Pros:** Mathematically rigorous, produces distortion coefficients.
**Cons:** Requires a physical printed pattern; impractical for quick field deployments.

### 2. Auto-YOLO (`method_auto_yolo.py`)

A field-expedient method using Triangle Similarity. A person stands at a known distance (1 m), and the YOLO bounding-box width in pixels is used to back-calculate the focal length:

```
f = (pixel_width × known_distance) / real_world_width
```

**Pros:** No special equipment — just a person and a tape measure.
**Cons:** Single-point calibration; no radial distortion correction.

### 3. Distance Comparison (`compare_distances.py`)

Runs both calibration methods side-by-side and plots the estimated distance against ground truth at multiple standoff positions, quantifying the error of each approach.

---

## Winning Result

The **Auto-YOLO** method was selected. With an adjusted torso width constant of **49.5 cm**, it delivered consistent distance readings (< 8% error) across 1–3 metres — the operational range of the intruder detection system.

The resulting camera matrix:

```python
CAMERA_MATRIX = np.array([
    [1108.9,    0.0,  640.0],
    [   0.0, 1108.9,  360.0],
    [   0.0,    0.0,    1.0],
])
```

This value is stored in `config.py` and consumed by `spatial_math.py`.

---

## Folder Structure

```
camera_calibration_testing/
├── method_checkerboard.py     ← OpenCV checkerboard calibration
├── method_auto_yolo.py        ← Triangle similarity with YOLO
└── compare_distances.py       ← Side-by-side accuracy comparison
```

---

## Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
matplotlib>=3.7.0
```
