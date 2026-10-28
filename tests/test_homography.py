import numpy as np

from footballmap.homography import (
    HomographyTracker,
    mean_squared_displacement,
    transform_points,
)

# A made-up "true" homography from frame pixels to tactical-map pixels.
H_TRUE = np.array([[0.5, 0.05, 20.0], [0.02, 0.6, 10.0], [1e-4, 2e-4, 1.0]])
MAP_POS = {"A": (0, 0), "B": (300, 0), "C": (300, 500), "D": (0, 500), "E": (150, 250)}


def frame_points(labels):
    """Where the keypoints would appear on the frame if H_TRUE were correct."""
    dst = np.array([MAP_POS[label] for label in labels], dtype=float)
    return transform_points(np.linalg.inv(H_TRUE), dst)


def maps_onto_tactical_map(matrix, labels, src):
    dst = np.array([MAP_POS[label] for label in labels], dtype=float)
    return np.allclose(transform_points(matrix, src), dst, atol=1e-3)


def test_transform_roundtrip():
    pts = np.array([[10.0, 20.0], [300.0, 400.0]])
    back = transform_points(np.linalg.inv(H_TRUE), transform_points(H_TRUE, pts))
    assert np.allclose(back, pts)


def test_transform_empty_input():
    assert transform_points(H_TRUE, np.empty((0, 2))).shape == (0, 2)


def test_mean_squared_displacement():
    assert mean_squared_displacement(np.array([[0, 0]]), np.array([[3, 4]])) == 12.5


def test_needs_four_keypoints():
    labels = ["A", "B", "C"]
    tracker = HomographyTracker()
    assert tracker.update(labels, frame_points(labels), MAP_POS) is None


def test_computes_matrix_on_first_frame():
    labels = ["A", "B", "C", "D"]
    src = frame_points(labels)
    matrix = HomographyTracker().update(labels, src, MAP_POS)
    assert matrix is not None and maps_onto_tactical_map(matrix, labels, src)


def test_keeps_matrix_while_keypoints_are_stable():
    labels = ["A", "B", "C", "D", "E"]
    src = frame_points(labels)
    tracker = HomographyTracker(displacement_tol=7.0)
    first = tracker.update(labels, src, MAP_POS)
    second = tracker.update(labels, src + 0.5, MAP_POS)  # mean squared shift 0.25 < 7
    assert second is first


def test_recomputes_when_keypoints_move():
    labels = ["A", "B", "C", "D", "E"]
    src = frame_points(labels)
    tracker = HomographyTracker(displacement_tol=7.0)
    first = tracker.update(labels, src, MAP_POS)
    moved = src + 20.0  # mean squared shift 400 > 7
    second = tracker.update(labels, moved, MAP_POS)
    assert second is not first and maps_onto_tactical_map(second, labels, moved)


def test_keeps_last_matrix_when_too_few_keypoints_are_seen():
    labels = ["A", "B", "C", "D"]
    tracker = HomographyTracker()
    first = tracker.update(labels, frame_points(labels), MAP_POS)
    assert tracker.update(["A", "B"], frame_points(["A", "B"]), MAP_POS) is first


def test_first_frame_with_few_keypoints_then_enough():
    # Crashed in the old code: previous-frame data was missing on the second frame.
    tracker = HomographyTracker()
    assert tracker.update(["A", "B"], frame_points(["A", "B"]), MAP_POS) is None
    labels = ["A", "B", "C", "D"]
    assert tracker.update(labels, frame_points(labels), MAP_POS) is not None
