"""Team assignment from jersey colours.

For every detected player we crop the torso area, reduce it to the 216-colour web palette,
take the most frequent colours and let each of them vote for the closest reference colour
(team player kit / team goalkeeper kit) in L*a*b* space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageColor
from skimage.color import rgb2lab

RGB = tuple[int, int, int]

COLORS_PER_TEAM = 2  # outfield kit + goalkeeper kit


def hex_to_rgb(color: str) -> RGB:
    """'#1E2530' -> (30, 37, 48)."""
    return ImageColor.getcolor(color, "RGB")


@dataclass(frozen=True)
class TeamSpec:
    name: str
    player_color: RGB
    gk_color: RGB

    @classmethod
    def from_hex(cls, name: str, player_hex: str, gk_hex: str) -> TeamSpec:
        return cls(name, hex_to_rgb(player_hex), hex_to_rgb(gk_hex))


def dominant_colors(crop_rgb: np.ndarray, num_colors: int = 3) -> list[RGB]:
    """Most frequent web-palette colours in the torso area of a player crop (RGB image).

    Returns an empty list when the crop is too small to contain a torso area.
    """
    height, width = crop_rgb.shape[:2]
    x1 = max(width // 2 - width // 5, 1)
    x2 = width // 2 + width // 5
    y1 = max(height // 3 - height // 5, 1)
    y2 = height // 3 + height // 5
    center = crop_rgb[y1:y2, x1:x2]
    if center.size == 0:
        return []

    reduced = Image.fromarray(np.uint8(center)).convert("P", palette=Image.Palette.WEB)
    flat_palette = reduced.getpalette()
    counts = sorted(reduced.getcolors(), key=lambda item: -item[0])  # (count, palette index)
    return [tuple(flat_palette[3 * idx : 3 * idx + 3]) for _, idx in counts[:num_colors]]


class TeamClassifier:
    """Assigns a player crop to team 0 or team 1."""

    def __init__(self, team1: TeamSpec, team2: TeamSpec, num_palette_colors: int = 3) -> None:
        self.teams = (team1, team2)
        self.num_palette_colors = num_palette_colors
        references = [team1.player_color, team1.gk_color, team2.player_color, team2.gk_color]
        self._reference_lab = _to_lab(references)

    def team_name(self, team_idx: int) -> str:
        return self.teams[team_idx].name

    def team_color(self, team_idx: int) -> RGB:
        return self.teams[team_idx].player_color

    def classify_palette(self, palette: Sequence[RGB]) -> int | None:
        """Majority vote of the palette colours; None if the palette is empty."""
        if len(palette) == 0:
            return None
        lab = _to_lab(palette)
        # Euclidean distance in L*a*b* (equivalent to skimage's deltaE_cie76)
        distances = np.linalg.norm(lab[:, None, :] - self._reference_lab[None, :, :], axis=-1)
        votes = [int(np.argmin(row)) // COLORS_PER_TEAM for row in distances]
        return max(votes, key=votes.count)

    def classify_crop(self, crop_rgb: np.ndarray) -> tuple[int | None, list[RGB]]:
        """Returns (team index or None, the palette used for the decision)."""
        palette = dominant_colors(crop_rgb, self.num_palette_colors)
        return self.classify_palette(palette), palette


def _to_lab(colors: Sequence[RGB]) -> np.ndarray:
    rgb = np.array(colors, dtype=float).reshape(1, -1, 3) / 255.0
    return rgb2lab(rgb).reshape(-1, 3)
