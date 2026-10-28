import numpy as np

from footballmap.teams import TeamClassifier, TeamSpec, dominant_colors, hex_to_rgb

RED, GREEN = (200, 30, 30), (30, 200, 30)
BLUE, YELLOW = (30, 30, 200), (230, 230, 30)


def make_classifier():
    return TeamClassifier(TeamSpec("Red FC", RED, GREEN), TeamSpec("Blue FC", BLUE, YELLOW))


def solid_crop(color, height=200, width=100):
    return np.full((height, width, 3), color, dtype=np.uint8)


def test_hex_to_rgb():
    assert hex_to_rgb("#1E2530") == (30, 37, 48)


def test_outfield_kits_are_assigned_to_the_right_team():
    classifier = make_classifier()
    assert classifier.classify_crop(solid_crop(RED))[0] == 0
    assert classifier.classify_crop(solid_crop(BLUE))[0] == 1


def test_goalkeeper_kits_are_assigned_to_the_right_team():
    classifier = make_classifier()
    assert classifier.classify_crop(solid_crop(GREEN))[0] == 0
    assert classifier.classify_crop(solid_crop(YELLOW))[0] == 1


def test_dominant_colors_respects_requested_count():
    colors = dominant_colors(solid_crop(RED), num_colors=3)
    assert 1 <= len(colors) <= 3
    assert all(len(c) == 3 for c in colors)


def test_tiny_crop_gives_no_team():
    team, palette = make_classifier().classify_crop(solid_crop(RED, height=2, width=2))
    assert team is None and palette == []


def test_empty_palette_gives_no_team():
    assert make_classifier().classify_palette([]) is None


def test_majority_vote_decides_the_team():
    assert make_classifier().classify_palette([RED, RED, BLUE]) == 0
    assert make_classifier().classify_palette([BLUE, YELLOW, RED]) == 1


def test_team_name_and_color_lookup():
    classifier = make_classifier()
    assert classifier.team_name(1) == "Blue FC"
    assert classifier.team_color(0) == RED
