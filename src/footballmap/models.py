"""Loading of the two YOLOv8 models (players/referees/ball and pitch keypoints)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from footballmap.config import KEYPOINTS_MODEL_PATH, PLAYERS_MODEL_PATH


@dataclass(frozen=True)
class Models:
    players: Any  # ultralytics.YOLO
    keypoints: Any  # ultralytics.YOLO


def missing_model_files() -> list[Path]:
    """Weight files that are not where the app expects them (see models/README.md)."""
    return [p for p in (PLAYERS_MODEL_PATH, KEYPOINTS_MODEL_PATH) if not p.is_file()]


@lru_cache(maxsize=1)
def load_models() -> Models:
    missing = missing_model_files()
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"Missing model weights: {names}. See models/README.md.")
    from ultralytics import YOLO  # imported lazily: it pulls in torch

    return Models(YOLO(str(PLAYERS_MODEL_PATH)), YOLO(str(KEYPOINTS_MODEL_PATH)))
