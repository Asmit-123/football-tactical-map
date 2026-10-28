import re

import pytest
from PIL import Image

from footballmap import config


@pytest.mark.parametrize(
    "path",
    [config.TACTICAL_MAP_PATH, config.KEYPOINTS_POSITIONS_PATH, config.CLASSES_PATH],
)
def test_bundled_assets_exist(path):
    assert path.is_file(), f"missing asset: {path}"


def test_keypoint_names_match_positions():
    names = set(config.load_pitch_keypoint_names().values())
    positions = set(config.load_keypoint_positions())
    assert len(names) == 28
    assert names == positions


def test_keypoints_fall_inside_tactical_map():
    width, height = Image.open(config.TACTICAL_MAP_PATH).size
    for name, (x, y) in config.load_keypoint_positions().items():
        assert 0 <= x < width and 0 <= y < height, name


def test_player_class_ids():
    names = config.load_player_class_names()
    assert names[config.PLAYER_CLASS_ID] == "player"
    assert names[config.REFEREE_CLASS_ID] == "referee"
    assert names[config.BALL_CLASS_ID] == "ball"


def test_demo_presets_have_valid_hex_colors():
    hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for preset in config.DEMO_PRESETS.values():
        for color in (
            preset.team1_player_color,
            preset.team1_gk_color,
            preset.team2_player_color,
            preset.team2_gk_color,
        ):
            assert hex_re.match(color)


def test_default_hyperparameters_match_original_app():
    d = config.DetectionParams()
    assert (d.player_conf, d.keypoint_conf, d.keypoint_displacement_tol) == (0.6, 0.7, 7.0)
    b = config.BallTrackParams()
    assert (b.reset_after_frames, b.max_jump_px, b.max_length) == (30, 100, 35)
