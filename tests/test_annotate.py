import numpy as np

from footballmap import annotate


def blank(height=200, width=200, value=0):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_rgb_to_bgr():
    assert annotate.rgb_to_bgr((1, 2, 3)) == (3, 2, 1)


def test_labeled_box_is_drawn_in_the_given_color():
    frame = blank()
    annotate.draw_labeled_box(frame, (50, 60, 100, 150), "player", (0, 255, 0))
    assert tuple(frame[60, 75]) == (0, 255, 0)  # top edge of the box


def test_palette_squares_use_the_palette_colors():
    frame = blank()
    annotate.draw_palette(frame, (50, 50, 100, 150), [(255, 0, 0), (0, 0, 255)])
    assert tuple(frame[55, 105]) == (0, 0, 255)  # first square: red given as RGB -> BGR
    assert tuple(frame[65, 105]) == (255, 0, 0)  # second square: blue given as RGB -> BGR


def test_tactical_player_marker():
    tac = blank(value=255)
    annotate.draw_tactical_player(tac, (100.7, 100.2), (0, 128, 0))
    assert tuple(tac[100, 100]) == (0, 128, 0)
    assert (tac == 0).all(axis=2).any()  # black outline


def test_tactical_ball_marker():
    tac = blank(value=255)
    annotate.draw_tactical_ball(tac, (100, 100))
    assert (tac == np.array([0, 0, 255])).all(axis=2).any()


def test_ball_track_drawing():
    tac = blank(value=255)
    annotate.draw_ball_track(tac, np.empty((0, 1, 2), dtype=np.int32))
    assert (tac == 255).all()  # empty track draws nothing
    annotate.draw_ball_track(tac, np.array([[[10, 10]], [[100, 100]]], dtype=np.int32))
    assert not (tac == 255).all()


def test_keypoint_boxes_are_drawn_in_black():
    frame = blank(value=255)
    annotate.draw_keypoint_boxes(frame, np.array([[20.0, 20.0, 60.0, 60.0]]))
    assert tuple(frame[20, 40]) == (0, 0, 0)


def test_compose_output_size():
    frame = blank(720, 1280)
    tac = blank(546, 314)
    out = annotate.compose_output(frame, tac, fps=25)
    # frame + borders (40+10 tall, 10+10 wide) next to the map + borders (10+10 wide)
    assert out.shape == (770, 1300 + 334, 3)
    assert annotate.compose_output(frame, tac).shape == out.shape  # fps is optional


def test_compose_output_works_for_small_frames():
    out = annotate.compose_output(blank(360, 640), blank(546, 314))
    assert out.shape[0] == 410
