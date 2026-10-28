"""Helpers for the 'Team Colors' tab: player thumbnails and picking a colour from them."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from footballmap.config import PLAYER_CLASS_ID

THUMB_SIZE = (60, 80)  # (width, height) of each player thumbnail


def rgb_to_hex(rgb: Sequence[int]) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def player_thumbnails(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    classes: np.ndarray,
    size: tuple[int, int] = THUMB_SIZE,
) -> list[np.ndarray]:
    """RGB thumbnails of every detected player (class 0) in the frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    thumbs = []
    for box, cls in zip(xyxy, classes, strict=True):
        if int(cls) != PLAYER_CLASS_ID:
            continue
        x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
        x2, y2 = min(int(box[2]), width), min(int(box[3]), height)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size:
            thumbs.append(cv2.resize(crop, size))
    return thumbs


def build_thumbnail_grid(
    thumbs: Sequence[np.ndarray], size: tuple[int, int] = THUMB_SIZE
) -> np.ndarray | None:
    """Two rows of thumbnails (white padding keeps the rows equally wide); None if empty."""
    if len(thumbs) == 0:
        return None
    half = len(thumbs) // 2
    row1, row2 = list(thumbs[:half]), list(thumbs[half:])
    if len(thumbs) % 2:
        row1.append(np.full((size[1], size[0], 3), 255, dtype=np.uint8))
    return cv2.vconcat([cv2.hconcat(row1), cv2.hconcat(row2)])


def pick_color(grid: np.ndarray, x: int, y: int) -> str | None:
    """Hex colour of the pixel at (x, y); None if the click is outside the image."""
    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
        return rgb_to_hex(grid[y, x])
    return None
