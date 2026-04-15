"""
face_verifier.py  –  YuNet-Based Face Detection Module
==========================================================

Follows the same class pattern as spatial_math.py:
  - Loads a model at __init__ time (YuNet ONNX via OpenCV DNN).
  - Exposes a single primary method: get_face_data(image) → tuple | None.
  - Gracefully degrades if the .onnx file is missing.

The YuNet detector is a lightweight CNN trained for face detection.
It is used here to extract the best face bounding box and eye landmarks
from a cropped torso image, enabling downstream alignment and
identity verification.

Download the ONNX weights from:
  https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Place the file in the same directory as this script.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np

# Path to YuNet ONNX model – expected alongside this file.
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_YUNET_ONNX_PATH = os.path.join(_SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")


class FaceVerifier:
    """
    YuNet face detection wrapper for the preprocessing pipeline.

    Usage
    -----
    verifier = FaceVerifier()
    result   = verifier.get_face_data(cropped_bgr_image)
    if result is not None:
        bbox, left_eye, right_eye = result

    Parameters
    ----------
    model_path : str
        Path to the YuNet ONNX weight file.
    score_threshold : float
        Minimum confidence for a detection to count as a face.
    """

    def __init__(
        self,
        model_path: str      = _YUNET_ONNX_PATH,
        score_threshold: float = 0.7,
    ):
        self._detector: Optional[cv2.FaceDetectorYN] = None

        if not os.path.isfile(model_path):
            print(
                f"[FaceVerifier] ERROR: YuNet ONNX file not found at:\n"
                f"  {model_path}\n"
                f"  Download from:\n"
                f"  https://github.com/opencv/opencv_zoo/blob/main/models/"
                f"face_detection_yunet/face_detection_yunet_2023mar.onnx\n"
                f"  Face detection will be DISABLED (get_face_data() returns None)."
            )
            return

        # Initialise the detector with a dummy input size.
        # setInputSize() is called per-image inside get_face_data() to match
        # the actual image dimensions (YuNet requires this).
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),      # Placeholder; overridden per-call
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5,
        )
        print(f"[FaceVerifier] YuNet loaded (score_threshold={score_threshold}).")

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def get_face_data(
        self, image: np.ndarray,
    ) -> Optional[Tuple[tuple, tuple, tuple]]:
        """
        Run YuNet on the given image and return the best detection's
        bounding box and eye landmarks.

        YuNet row format:
            [x, y, w, h,
             right_eye_x, right_eye_y, left_eye_x, left_eye_y,
             nose_x, nose_y,
             mouth_right_x, mouth_right_y, mouth_left_x, mouth_left_y,
             score]

        Parameters
        ----------
        image : np.ndarray
            BGR image (typically a bounding-box crop from main_watchdog).

        Returns
        -------
        tuple or None
            ``(bbox, left_eye, right_eye)`` where *bbox* = ``(x, y, w, h)``
            and each eye is a ``(float, float)`` coordinate pair.
            Returns ``None`` if no face is found or the model failed to load.
        """
        if self._detector is None:
            return None  # Model not loaded — graceful degradation

        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]

        # YuNet requires setInputSize() to match the input dimensions
        # before every detect() call; mismatched sizes cause silent failures.
        self._detector.setInputSize((w, h))

        _, detections = self._detector.detect(image)

        if detections is None or len(detections) == 0:
            return None

        best = max(detections, key=lambda d: d[14])
        bbox      = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
        right_eye = (float(best[4]), float(best[5]))
        left_eye  = (float(best[6]), float(best[7]))

        return bbox, left_eye, right_eye

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True if the YuNet model was successfully initialised."""
        return self._detector is not None
