"""Drawing helpers for the video frame and the tactical map (all colours are BGR).

The drawing functions modify the image they are given, like the cv2 functions they wrap.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

BGR = tuple[int, int, int]
RGB = tuple[int, int, int]

WHITE: BGR = (255, 255, 255)
BLACK: BGR = (0, 0, 0)
BALL_COLOR: BGR = (0, 0, 255)
TRACK_COLOR: BGR = (0, 0, 100)
PALETTE_BOX_SIZE = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX


def rgb_to_bgr(color: RGB) -> BGR:
    return (int(color[2]), int(color[1]), int(color[0]))


def _int_box(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return x1, y1, x2, y2


# ----------------------------- video frame ----------------------------- #
def draw_labeled_box(frame: np.ndarray, bbox: Sequence[float], label: str, color: BGR) -> None:
    """Bounding box (x1, y1, x2, y2) with a text label above it."""
    x1, y1, x2, y2 = _int_box(bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, label, (x1, y1 - 10), FONT, 0.5, color, 2)


def draw_palette(frame: np.ndarray, bbox: Sequence[float], palette_rgb: Sequence[RGB]) -> None:
    """Small colour squares next to a player's box showing the extracted palette."""
    _, y1, x2, _ = _int_box(bbox)
    for k, color in enumerate(palette_rgb):
        top_left = (x2 + 3, y1 + k * PALETTE_BOX_SIZE)
        bottom_right = (x2 + PALETTE_BOX_SIZE, y1 + PALETTE_BOX_SIZE * (k + 1))
        cv2.rectangle(frame, top_left, bottom_right, rgb_to_bgr(color), -1)


def draw_keypoint_boxes(frame: np.ndarray, bboxes: np.ndarray) -> None:
    for bbox in bboxes:
        x1, y1, x2, y2 = _int_box(bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 1)


# ----------------------------- tactical map ----------------------------- #
def draw_tactical_player(tac_map: np.ndarray, pt: Sequence[float], color: BGR) -> None:
    center = (int(pt[0]), int(pt[1]))
    cv2.circle(tac_map, center, radius=5, color=color, thickness=-1)
    cv2.circle(tac_map, center, radius=5, color=BLACK, thickness=1)


def draw_tactical_ball(tac_map: np.ndarray, pt: Sequence[float]) -> None:
    cv2.circle(tac_map, (int(pt[0]), int(pt[1])), radius=5, color=BALL_COLOR, thickness=3)


def draw_ball_track(tac_map: np.ndarray, dst_points: np.ndarray) -> None:
    """``dst_points`` is an int32 array shaped (n, 1, 2); does nothing when empty."""
    if len(dst_points) == 0:
        return
    cv2.polylines(tac_map, [dst_points], isClosed=False, color=TRACK_COLOR, thickness=2)


# ------------------------------ final image ------------------------------ #
def compose_output(frame: np.ndarray, tac_map: np.ndarray, fps: float | None = None) -> np.ndarray:
    """Frame and tactical map side by side, with white borders, a title and optional FPS."""
    framed = cv2.copyMakeBorder(frame, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=WHITE)
    tac = cv2.copyMakeBorder(tac_map, 70, 50, 10, 10, cv2.BORDER_CONSTANT, value=WHITE)
    tac = cv2.resize(tac, (tac.shape[1], framed.shape[0]))
    final = cv2.hconcat((framed, tac))

    # Title centred over the map; the 70 px top border was scaled with the resize above.
    title = "Tactical Map"
    (text_w, text_h), _ = cv2.getTextSize(title, FONT, 0.9, 2)
    top_band = round(70 * framed.shape[0] / (tac_map.shape[0] + 120))
    x = framed.shape[1] + (tac.shape[1] - text_w) // 2
    y = max(int(top_band * 0.75), text_h + 5)
    cv2.putText(final, title, (x, y), FONT, 0.9, BLACK, 2)

    if fps is not None:
        cv2.putText(final, f"FPS: {int(fps)}", (20, 30), FONT, 0.9, BLACK, 2)
    return final
