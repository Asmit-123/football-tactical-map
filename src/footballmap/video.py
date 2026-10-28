"""Video input/output helpers (no Streamlit)."""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from itertools import chain, count
from pathlib import Path

import cv2
import numpy as np

from footballmap.config import OUTPUTS_DIR

UPLOAD_DIR = Path(tempfile.gettempdir()) / "footballmap_uploads"


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


def probe(path: Path | str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        return VideoInfo(
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=fps if fps and fps > 0 else 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_frame(path: Path | str, frame_number: int) -> np.ndarray | None:
    """Frame ``frame_number`` (1-based) as a BGR image, or None if it can't be read."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def save_upload(data: bytes, suffix: str = ".mp4", directory: Path | None = None) -> Path:
    """Store uploaded bytes under a content-hash name; the same upload is stored only once."""
    directory = Path(directory) if directory else UPLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hashlib.sha1(data).hexdigest()[:16]}{suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path


def output_path(name: str | None = None, directory: Path | None = None) -> Path:
    """A free ``.mp4`` path: the sanitised name if given, else detect_1, detect_2, ..."""
    directory = Path(directory) if directory else OUTPUTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    name = (name or "").strip()
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    stem = re.sub(r"[^\w\- ]", "_", name).strip(" ._")

    if stem:
        candidates = chain([stem], (f"{stem}_{i}" for i in count(1)))
    else:
        candidates = (f"detect_{i}" for i in count(1))
    for candidate in candidates:
        path = directory / f"{candidate}.mp4"
        if not path.exists():
            return path
    raise RuntimeError("unreachable")


def _fourcc(code: str) -> int:
    try:
        return cv2.VideoWriter_fourcc(*code)
    except AttributeError:  # newer OpenCV releases expose it on the class only
        return cv2.VideoWriter.fourcc(*code)


class VideoSink:
    """Writes frames to an mp4 file; the size is taken from the first frame written."""

    def __init__(self, path: Path | str, fps: float) -> None:
        self.path = Path(path)
        self.fps = fps
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), _fourcc("mp4v"), self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise OSError(f"Could not open {self.path} for writing")
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
