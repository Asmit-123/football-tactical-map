import numpy as np

from footballmap.ball import BallTracker
from footballmap.config import BallTrackParams


def make_tracker(max_jump=100, reset_after=3, max_length=5):
    return BallTracker(BallTrackParams(reset_after, max_jump, max_length))


def see(tracker, x, y=0):
    tracker.update(np.array([x, y], dtype=float), np.array([x / 2, y / 2]))


def test_records_points_in_order():
    tracker = make_tracker()
    for x in (0, 10, 20):
        see(tracker, x)
    assert tracker.src == [(0, 0), (10, 0), (20, 0)]
    assert tracker.dst == [(0, 0), (5, 0), (10, 0)]


def test_large_jump_starts_a_new_track():
    tracker = make_tracker(max_jump=100)
    see(tracker, 0)
    see(tracker, 500)
    assert tracker.src == [(500, 0)]


def test_track_length_is_capped():
    tracker = make_tracker(max_length=5)
    for x in range(0, 100, 10):
        see(tracker, x)
    assert len(tracker.src) == len(tracker.dst) == 5
    assert tracker.src[-1] == (90, 0)


def test_track_is_cleared_after_too_many_frames_without_ball():
    tracker = make_tracker(reset_after=3)
    see(tracker, 0)
    for _ in range(4):
        tracker.update(None, None)
    assert tracker.src  # 4 missed frames: the reset happens on the next call
    tracker.update(None, None)
    assert tracker.src == [] and tracker.dst == []


def test_seeing_the_ball_resets_the_missing_counter():
    tracker = make_tracker(reset_after=3)
    see(tracker, 0)
    for _ in range(3):
        tracker.update(None, None)
    see(tracker, 10)
    for _ in range(3):
        tracker.update(None, None)
    assert tracker.src == [(0, 0), (10, 0)]


def test_record_false_does_not_store_points():
    tracker = make_tracker()
    tracker.update(np.array([1.0, 1.0]), np.array([1.0, 1.0]), record=False)
    assert tracker.src == []


def test_no_homography_means_nothing_is_stored():
    tracker = make_tracker()
    tracker.update(np.array([1.0, 1.0]), None)
    assert tracker.src == []


def test_dst_points_shape_for_polylines():
    tracker = make_tracker()
    see(tracker, 0)
    see(tracker, 10)
    assert tracker.dst_points.shape == (2, 1, 2)
    assert tracker.dst_points.dtype == np.int32
    assert make_tracker().dst_points.shape == (0, 1, 2)
