"""Pitch-keypoint homography: frame pixels -> tactical-map pixels.

Port of the homography logic that used to live inside ``detect()``. The matrix is only
recomputed when the detected keypoints have moved (on average) more than a tolerance since
the previous frame, which keeps the tactical map stable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

MIN_KEYPOINTS = 4  # a homography needs at least 4 point correspondences


def mean_squared_displacement(prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
    """Mean squared difference over all coordinates (same as sklearn's mean_squared_error)."""
    prev = np.asarray(prev_pts, dtype=float)
    curr = np.asarray(curr_pts, dtype=float)
    return float(np.mean((prev - curr) ** 2))


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (n, 2) array of points; returns an (n, 2) float array."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    projected = homogeneous @ np.asarray(matrix, dtype=float).T
    return projected[:, :2] / projected[:, 2:3]


class HomographyTracker:
    """Keeps the current frame->map homography and decides when to recompute it."""

    def __init__(self, displacement_tol: float = 7.0) -> None:
        self.displacement_tol = displacement_tol
        self.matrix: np.ndarray | None = None
        self._prev: dict[str, np.ndarray] | None = None  # keypoint label -> frame position

    def reset(self) -> None:
        self.matrix = None
        self._prev = None

    def update(
        self,
        labels: Sequence[str],
        src_pts: np.ndarray,
        dst_lookup: Mapping[str, tuple[int, int]],
    ) -> np.ndarray | None:
        """Feed one frame's detected keypoints; returns the current matrix (or None).

        ``labels[i]`` is the keypoint name detected at ``src_pts[i]`` (frame pixels) and
        ``dst_lookup`` maps each name to its position on the tactical map.
        """
        src = np.asarray(src_pts, dtype=float).reshape(-1, 2)

        # First occurrence wins if the model detects the same keypoint twice.
        current: dict[str, np.ndarray] = {}
        for label, pt in zip(labels, src, strict=True):
            current.setdefault(label, pt)

        if len(labels) >= MIN_KEYPOINTS:
            recompute = True
            if self._prev is not None:
                common = sorted(set(self._prev) & set(current))
                if len(common) >= MIN_KEYPOINTS:
                    prev_common = np.array([self._prev[name] for name in common])
                    curr_common = np.array([current[name] for name in common])
                    error = mean_squared_displacement(prev_common, curr_common)
                    recompute = error > self.displacement_tol
            if recompute:
                dst = np.array([dst_lookup[name] for name in labels], dtype=float)
                matrix, _ = cv2.findHomography(src, dst)
                if matrix is not None:  # degenerate point sets return None: keep the old one
                    self.matrix = matrix

        if self.matrix is not None:
            self._prev = current
        return self.matrix
