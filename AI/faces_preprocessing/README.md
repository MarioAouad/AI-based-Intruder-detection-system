# Faces Preprocessing — Sandbox Environment

> **Phase 3.5** — Experimental sandbox for developing and debugging the 3-step face preprocessing pipeline before integration into the production `face_processor.py`.

---

## Purpose

Raw torso crops from the watchdog contain full upper-body images with inconsistent head angles, varying lighting, and background clutter. This sandbox script develops the preprocessing chain that transforms those crops into standardised **160×160 AI-ready face images** suitable for embedding models like Facenet512.

**Important:** This is a development/testing sandbox. The production version lives at `face_processor.py` in the project root (stripped of all debug outputs).

---

## The 3-Step Preprocessing Pipeline

### Step 1: Affine Eye-Alignment

Uses YuNet's 5-point facial landmarks to extract the left and right eye coordinates. The rotation angle is calculated via `atan2(dy, dx)`, and the image is rotated so both eyes sit on a perfectly horizontal line.

**Bug Fix Applied:** Eye coordinates are **sorted by X-value** before calculating `dx` and `dy`. This prevents 180° flips when YuNet returns eyes in inconsistent left/right order depending on head pose.

```
eye1, eye2 = sorted([left_eye, right_eye], key=lambda p: p[0])
```

### Step 2: 160×160 Face Crop

The face bounding box (from YuNet) is expanded by a **20% margin** on all sides to include forehead and chin context while eliminating background noise. The region is then resized to exactly 160×160 pixels using `cv2.INTER_AREA` interpolation.

**Why 160×160?** This is the native input size for FaceNet and compatible with ArcFace, ensuring zero information loss during embedding extraction.

**Why 20% margin?** Testing showed that tighter crops cut off foreheads on tilted faces, while larger margins reintroduced background. 20% was the empirical sweet spot.

### Step 3: CLAHE Lighting Normalisation

Converts the image to **LAB colour space**, applies CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the L-channel only, and converts back to BGR. This normalises lighting across day/night captures without affecting colour hue.

**Parameters:** `clipLimit=2.0`, `tileGridSize=(4, 4)` — tuned to avoid over-amplifying noise in dark regions.

---

## Debug Output Folders

This sandbox saves intermediate results at every step for visual inspection:

```
faces_preprocessing/
├── faces_preprocessing.py          ← The sandbox worker script
├── captured_targets/               ← Input: raw torso crops (sandbox copy)
├── captured_targets_faces/         ← YuNet-verified crops (pipeline prep)
├── faces_aligned/                  ← Final output: 160×160 faces
│   ├── owner/                      ← Owner-routed faces
│   └── intruder/                   ← Intruder-routed faces
├── step1_faces_debug_affine/       ← After eye-alignment rotation
├── step2_faces_debug_crop/         ← After 160×160 crop (before CLAHE)
├── step3_faces_debug_lighting/     ← After CLAHE normalisation
└── faces_debug_comparison/         ← Side-by-side "Original vs. Final"
```

---

## Routing Logic

- Filename contains `"owner"` → saved to `faces_aligned/owner/`
- Filename contains `"target"` or anything else → saved to `faces_aligned/` (intruder staging)

**Critical:** Original filenames are preserved — no prefixes like `aligned_` or `_clahe` are added.

---

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
```

*(Also requires `face_verifier.py` and `face_detection_yunet_2023mar.onnx` from the project root.)*
