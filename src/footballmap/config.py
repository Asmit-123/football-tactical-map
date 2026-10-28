"""Central configuration: paths, default hyperparameters and demo presets.

Everything that used to be a hard-coded path or a magic number in the old
``main.py`` / ``detection.py`` lives here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Paths (pathlib => identical behaviour on Windows, macOS and Linux)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
DEMO_DIR = ASSETS_DIR / "demo"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TACTICAL_MAP_PATH = ASSETS_DIR / "tactical_map.jpg"
KEYPOINTS_POSITIONS_PATH = ASSETS_DIR / "pitch_keypoints.json"
CLASSES_PATH = ASSETS_DIR / "classes.yaml"

PLAYERS_MODEL_PATH = MODELS_DIR / "players_yolov8l.pt"
KEYPOINTS_MODEL_PATH = MODELS_DIR / "field_keypoints_yolov8m.pt"

VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "m4v", "asf")

# Class ids of the players model
PLAYER_CLASS_ID = 0
REFEREE_CLASS_ID = 1
BALL_CLASS_ID = 2


# --------------------------------------------------------------------------- #
# Hyperparameters (defaults match the original app's sliders)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionParams:
    player_conf: float = 0.6
    keypoint_conf: float = 0.7
    keypoint_displacement_tol: float = 7.0  # px; homography is recomputed above this
    num_palette_colors: int = 3  # dominant colours per player used for team voting
    frame_stride: int = 1  # 1 = process every frame (same as the original)


@dataclass(frozen=True)
class BallTrackParams:
    reset_after_frames: int = 30  # frames without a ball before the track is cleared
    max_jump_px: int = 100  # max distance between consecutive detections in one track
    max_length: int = 35  # max detections kept in the track history


@dataclass(frozen=True)
class PlotOptions:
    show_keypoints: bool = False
    show_palette: bool = True
    show_ball_track: bool = True
    show_players: bool = True


# --------------------------------------------------------------------------- #
# Demo presets (team names and colours from the original app)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoPreset:
    video_path: Path
    team1_name: str
    team2_name: str
    team1_player_color: str
    team1_gk_color: str
    team2_player_color: str
    team2_gk_color: str


DEMO_PRESETS: dict[str, DemoPreset] = {
    "Demo 1": DemoPreset(
        video_path=DEMO_DIR / "demo_vid_1.mp4",
        team1_name="France",
        team2_name="Switzerland",
        team1_player_color="#1E2530",
        team1_gk_color="#F5FD15",
        team2_player_color="#FBFCFA",
        team2_gk_color="#B1FCC4",
    ),
    "Demo 2": DemoPreset(
        video_path=DEMO_DIR / "demo_vid_2.mp4",
        team1_name="Chelsea",
        team2_name="Manchester City",
        team1_player_color="#29478A",
        team1_gk_color="#DC6258",
        team2_player_color="#90C8FF",
        team2_gk_color="#BCC703",
    ),
}


# --------------------------------------------------------------------------- #
# Asset loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_keypoint_positions() -> dict[str, tuple[int, int]]:
    """Pitch keypoint name -> (x, y) pixel position on the tactical map image."""
    with KEYPOINTS_POSITIONS_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {name: (int(x), int(y)) for name, (x, y) in raw.items()}


@lru_cache(maxsize=1)
def _load_classes() -> dict[str, dict[int, str]]:
    with CLASSES_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {
        section: {int(k): str(v) for k, v in names.items()} for section, names in raw.items()
    }


def load_pitch_keypoint_names() -> dict[int, str]:
    """Keypoints-model class id -> keypoint name (e.g. 0 -> 'TLC')."""
    return _load_classes()["pitch_keypoints"]


def load_player_class_names() -> dict[int, str]:
    """Players-model class id -> name (0 player, 1 referee, 2 ball)."""
    return _load_classes()["players"]
