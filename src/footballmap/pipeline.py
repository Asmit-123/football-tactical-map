"""Per-frame analysis and rendering, plus a generator that runs a whole video.

Nothing in here depends on Streamlit: the UI (or a script, or a test) just feeds frames in
and gets annotated images out.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from footballmap import annotate
from footballmap.ball import BallTracker
from footballmap.config import (
    BALL_CLASS_ID,
    PLAYER_CLASS_ID,
    TACTICAL_MAP_PATH,
    BallTrackParams,
    DetectionParams,
    PlotOptions,
    load_keypoint_positions,
    load_pitch_keypoint_names,
    load_player_class_names,
)
from footballmap.homography import HomographyTracker, transform_points
from footballmap.teams import RGB, TeamClassifier


@dataclass(frozen=True)
class PipelineSettings:
    detection: DetectionParams = field(default_factory=DetectionParams)
    ball: BallTrackParams = field(default_factory=BallTrackParams)
    plot: PlotOptions = field(default_factory=PlotOptions)


@dataclass
class PlayerDetection:
    bbox: np.ndarray  # x1, y1, x2, y2 on the frame
    conf: float
    team_idx: int | None  # 0 / 1, None when no colour information was available
    palette: list[RGB]
    map_pos: np.ndarray | None  # position on the tactical map (None without homography)


@dataclass
class OtherDetection:
    bbox: np.ndarray
    conf: float
    label: str  # 'referee' or 'ball'


@dataclass
class FrameAnalysis:
    players: list[PlayerDetection]
    others: list[OtherDetection]
    ball_map_pos: np.ndarray | None
    keypoint_boxes: np.ndarray
    homography: np.ndarray | None


@dataclass
class ProcessedFrame:
    index: int  # frames read so far (1-based)
    total: int  # frame count reported by the video (0 if unknown)
    image: np.ndarray
    fps: float | None


def _to_numpy(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.cpu().numpy())


class FrameProcessor:
    """Detects players/ball/keypoints in a frame and renders the annotated output image."""

    def __init__(
        self,
        players_model: Any,
        keypoints_model: Any,
        classifier: TeamClassifier,
        settings: PipelineSettings | None = None,
        tac_map: np.ndarray | None = None,
    ) -> None:
        self.players_model = players_model
        self.keypoints_model = keypoints_model
        self.classifier = classifier
        self.settings = settings or PipelineSettings()

        if tac_map is None:
            tac_map = cv2.imread(str(TACTICAL_MAP_PATH))
            if tac_map is None:
                raise FileNotFoundError(f"Could not read {TACTICAL_MAP_PATH}")
        self._tac_map = tac_map

        self.homography = HomographyTracker(self.settings.detection.keypoint_displacement_tol)
        self.ball = BallTracker(self.settings.ball)
        self._keypoint_positions = load_keypoint_positions()
        self._keypoint_names = load_pitch_keypoint_names()
        self._class_names = load_player_class_names()

    # ------------------------------------------------------------------ #
    def analyze(self, frame_bgr: np.ndarray) -> FrameAnalysis:
        det = self.settings.detection
        players_boxes = self.players_model(frame_bgr, conf=det.player_conf, verbose=False)[0].boxes
        keypoint_boxes = self.keypoints_model(
            frame_bgr, conf=det.keypoint_conf, verbose=False
        )[0].boxes

        xyxy = _to_numpy(players_boxes.xyxy).reshape(-1, 4)
        xywh = _to_numpy(players_boxes.xywh).reshape(-1, 4)
        classes = _to_numpy(players_boxes.cls).astype(int).reshape(-1)
        confs = _to_numpy(players_boxes.conf).reshape(-1)

        kp_xyxy = _to_numpy(keypoint_boxes.xyxy).reshape(-1, 4)
        kp_xywh = _to_numpy(keypoint_boxes.xywh).reshape(-1, 4)
        kp_classes = _to_numpy(keypoint_boxes.cls).astype(int).reshape(-1)

        # Frame -> tactical map transformation, from the detected pitch keypoints
        kp_names = [self._keypoint_names[c] for c in kp_classes]
        kp_src = np.round(kp_xywh[:, :2]).astype(int)
        matrix = self.homography.update(kp_names, kp_src, self._keypoint_positions)

        # Players: the point on the pitch is the bottom-centre of the box (the feet)
        player_idx = np.flatnonzero(classes == PLAYER_CLASS_ID)
        feet = np.column_stack([xywh[player_idx, 0], xywh[player_idx, 1] + xywh[player_idx, 3] / 2])
        player_map = transform_points(matrix, feet) if matrix is not None else None

        # Ball: centre of the first detected ball
        ball_idx = np.flatnonzero(classes == BALL_CLASS_ID)
        ball_src = xywh[ball_idx[0], :2] if len(ball_idx) else None
        ball_map = None
        if matrix is not None and ball_src is not None:
            ball_map = transform_points(matrix, ball_src)[0]
        self.ball.update(ball_src, ball_map, record=self.settings.plot.show_ball_track)

        # Team of every player, from the colours of the torso area
        players: list[PlayerDetection] = []
        if len(player_idx):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            for n, i in enumerate(player_idx):
                x1, y1, x2, y2 = xyxy[i]
                crop = frame_rgb[max(int(y1), 0) : min(int(y2), height),
                                 max(int(x1), 0) : min(int(x2), width)]
                team_idx, palette = self.classifier.classify_crop(crop)
                map_pos = None if player_map is None else player_map[n]
                players.append(
                    PlayerDetection(xyxy[i], float(confs[i]), team_idx, palette, map_pos)
                )

        others = [
            OtherDetection(xyxy[i], float(confs[i]), self._class_names.get(int(c), str(c)))
            for i, c in enumerate(classes)
            if c != PLAYER_CLASS_ID
        ]
        return FrameAnalysis(players, others, ball_map, kp_xyxy, matrix)

    # ------------------------------------------------------------------ #
    def render(
        self, frame_bgr: np.ndarray, analysis: FrameAnalysis, fps: float | None = None
    ) -> np.ndarray:
        plot = self.settings.plot
        image = frame_bgr.copy()
        tac = self._tac_map.copy()

        for player in analysis.players:
            if player.team_idx is None:
                name, color = "Unknown", annotate.WHITE
            else:
                name = self.classifier.team_name(player.team_idx)
                color = annotate.rgb_to_bgr(self.classifier.team_color(player.team_idx))
            if plot.show_palette and player.palette:
                annotate.draw_palette(image, player.bbox, player.palette)
            if plot.show_players:
                annotate.draw_labeled_box(image, player.bbox, f"{name} {player.conf:.2f}", color)
            if player.map_pos is not None:
                annotate.draw_tactical_player(tac, player.map_pos, color)

        for other in analysis.others:
            annotate.draw_labeled_box(
                image, other.bbox, f"{other.label} {other.conf:.2f}", annotate.WHITE
            )
        if analysis.ball_map_pos is not None:
            annotate.draw_tactical_ball(tac, analysis.ball_map_pos)
        if plot.show_keypoints:
            annotate.draw_keypoint_boxes(image, analysis.keypoint_boxes)
        annotate.draw_ball_track(tac, self.ball.dst_points)

        return annotate.compose_output(image, tac, fps)

    def process(self, frame_bgr: np.ndarray, fps: float | None = None) -> np.ndarray:
        return self.render(frame_bgr, self.analyze(frame_bgr), fps)


def iter_processed_frames(
    video_path: Path | str, processor: FrameProcessor, frame_stride: int = 1
) -> Iterator[ProcessedFrame]:
    """Yield annotated frames; with ``frame_stride`` N only every Nth frame is processed."""
    stride = max(int(frame_stride), 1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        index = 0
        fps: float | None = None
        while True:
            if index % stride:  # skipped frame: advance without decoding it
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            started = time.perf_counter()
            image = processor.process(frame, fps)
            fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            yield ProcessedFrame(index, total, image, fps)
    finally:
        cap.release()
