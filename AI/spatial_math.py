"""
spatial_math.py  –  Distance Estimation via Triangle Similarity
================================================================

This module dynamically extracts the focal length f_x from the Camera
Intrinsic Matrix defined in config.py and uses it in the Triangle
Similarity formula to estimate real-world distance.

Background  (for the university report)
-----------------------------------------
Triangle Similarity is a monocular depth estimation technique based on
the geometry of similar triangles formed by the camera's optical axis.

Given:
    f_x  = focal length in pixels, extracted from CAMERA_MATRIX[0, 0]
    W    = known real-world width of the object (REAL_TORSO_WIDTH_CM)
    P    = observed pixel width of the object's bounding box

The distance formula is:

        distance_cm = (W × f_x) / P
        distance_m  = distance_cm / 100

The f_x value was obtained from the Auto-YOLO calibration experiment
(camera_calibration_testing/method_auto_yolo.py) which uses the same
Triangle Similarity principle to compute f_x once at a known distance:

        f_x = (P_calibration × D_known) / W

After calibration, f_x is stored inside the 3×3 intrinsic matrix K:

        K = ┌  f_x   0   c_x ┐
            │   0   f_y  c_y │
            └   0    0    1  ┘

This module reads K at runtime so that if the camera or calibration
changes, only config.py needs to be updated.
"""

import numpy as np

from config import CAMERA_MATRIX, REAL_TORSO_WIDTH_CM


class DistanceEstimator:
    """
    Computes real-world distance from the camera to a detected person
    using Triangle Similarity, dynamically extracting f_x from the
    Camera Intrinsic Matrix.

    Parameters
    ----------
    camera_matrix : np.ndarray (3×3)
        The camera intrinsic matrix K.  f_x is read from K[0, 0].
    real_width_cm : float
        Known real-world width of a human torso/shoulders in cm.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray = CAMERA_MATRIX,
        real_width_cm: float      = REAL_TORSO_WIDTH_CM,
    ):
        self.real_width_cm = real_width_cm

        # ── Extract f_x directly from the intrinsic matrix ────────────
        # K[0, 0] = f_x  (horizontal focal length in pixels)
        self.focal_length = float(camera_matrix[0, 0])

        if self.focal_length <= 0:
            print(
                "[DistanceEstimator] WARNING: f_x from CAMERA_MATRIX is ≤ 0.\n"
                "  Distance estimation is DISABLED.\n"
                "  Run camera_calibration_testing/method_auto_yolo.py to get\n"
                "  a valid matrix and paste it into config.py → CAMERA_MATRIX."
            )
        else:
            print(
                f"[DistanceEstimator] f_x extracted from intrinsic matrix: "
                f"{self.focal_length:.2f} px  "
                f"(torso reference = {real_width_cm} cm)"
            )

    # ------------------------------------------------------------------
    # Triangle Similarity distance formula
    # ------------------------------------------------------------------
    #   distance_cm = (REAL_TORSO_WIDTH_CM × f_x) / pixel_width
    #   distance_m  = distance_cm / 100.0
    #
    # This is the rearranged form of:
    #   f_x = (pixel_width × distance_cm) / REAL_TORSO_WIDTH_CM
    # which was used during calibration (person at 100 cm).
    # ------------------------------------------------------------------

    def calculate_distance(self, pixel_width: float) -> float | None:
        """
        Estimate the person's distance from the camera in metres.

        Uses Triangle Similarity, dynamically extracting f_x from the
        intrinsic matrix set during __init__:

            distance_cm = (W × f_x) / P
            distance_m  = distance_cm / 100

        Parameters
        ----------
        pixel_width : float
            Width (in pixels) of the detected person's bounding box.

        Returns
        -------
        float | None
            Estimated distance in metres (rounded to 2 d.p.), or None
            if calibration is missing or pixel_width is zero.
        """
        if self.focal_length <= 0:
            return None  # No valid calibration

        if pixel_width <= 0:
            return None  # Guard against division by zero

        distance_cm = (self.real_width_cm * self.focal_length) / pixel_width
        distance_m  = distance_cm / 100.0
        return round(distance_m, 2)

    def is_calibrated(self) -> bool:
        """Return True if a valid focal length was extracted from the matrix."""
        return self.focal_length > 0
