"""Ball track history shown on the tactical map."""

from __future__ import annotations

import numpy as np

from footballmap.config import BallTrackParams


class BallTracker:
    """Keeps the recent ball positions (frame and tactical-map coordinates)."""

    def __init__(self, params: BallTrackParams | None = None) -> None:
        self.params = params or BallTrackParams()
        self.src: list[tuple[int, int]] = []  # positions on the video frame
        self.dst: list[tuple[int, int]] = []  # positions on the tactical map
        self._frames_without_ball = 0

    def reset_track(self) -> None:
        self.src = []
        self.dst = []

    def update(
        self,
        src_pos: np.ndarray | None,
        dst_pos: np.ndarray | None,
        record: bool = True,
    ) -> None:
        """Call once per frame.

        ``src_pos`` is the ball on the frame (None if not detected), ``dst_pos`` the same
        point on the tactical map (None if there is no homography yet). With
        ``record=False`` the track is not stored (used when the track is switched off).
        """
        p = self.params
        if self._frames_without_ball > p.reset_after_frames:
            self.reset_track()

        if src_pos is None:
            self._frames_without_ball += 1
            return
        self._frames_without_ball = 0
        if dst_pos is None or not record:
            return

        if self.src:
            jump = np.linalg.norm(np.asarray(src_pos, dtype=float) - np.asarray(self.src[-1]))
            if jump >= p.max_jump_px:  # too far from the last point: start a new track
                self.reset_track()

        self.src.append((int(src_pos[0]), int(src_pos[1])))
        self.dst.append((int(dst_pos[0]), int(dst_pos[1])))
        while len(self.src) > p.max_length:
            self.src.pop(0)
            self.dst.pop(0)

    @property
    def dst_points(self) -> np.ndarray:
        """Track on the tactical map as an int32 array shaped (n, 1, 2) for cv2.polylines."""
        return np.array(self.dst, dtype=np.int32).reshape(-1, 1, 2)
