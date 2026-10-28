import numpy as np

from footballmap import colorpick


def thumb(value):
    return np.full((80, 60, 3), value, dtype=np.uint8)


def test_rgb_to_hex():
    assert colorpick.rgb_to_hex((30, 37, 48)) == "#1e2530"


def test_grid_is_none_without_players():
    assert colorpick.build_thumbnail_grid([]) is None


def test_grid_has_two_equally_wide_rows():
    for n in (1, 2, 3, 4, 5, 8):
        grid = colorpick.build_thumbnail_grid([thumb(10)] * n)
        assert grid.shape == (160, 60 * ((n + 1) // 2), 3), n


def test_odd_count_is_padded_with_white():
    grid = colorpick.build_thumbnail_grid([thumb(0)] * 3)
    assert tuple(grid[0, -1]) == (255, 255, 255)  # padding at the end of the first row


def test_pick_color_and_bounds():
    grid = colorpick.build_thumbnail_grid([thumb(0), thumb(255)])
    assert colorpick.pick_color(grid, 10, 10) == "#000000"
    assert colorpick.pick_color(grid, 10, 100) == "#ffffff"
    assert colorpick.pick_color(grid, 999, 10) is None
    assert colorpick.pick_color(grid, 10, -1) is None


def test_player_thumbnails_only_keep_players_inside_the_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 90], [60, 10, 100, 90], [150, 50, 260, 90], [5, 5, 5, 5]], float)
    classes = np.array([0, 2, 0, 0])  # player, ball, player (partly outside), empty box
    thumbs = colorpick.player_thumbnails(frame, boxes, classes)
    assert len(thumbs) == 2 and all(t.shape == (80, 60, 3) for t in thumbs)
