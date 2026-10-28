import cv2
import numpy as np

from footballmap.config import PlotOptions, load_keypoint_positions, load_pitch_keypoint_names
from footballmap.homography import transform_points
from footballmap.pipeline import FrameProcessor, PipelineSettings, iter_processed_frames
from footballmap.teams import TeamClassifier, TeamSpec
from footballmap.video import _fourcc

RED, GREEN, BLUE, YELLOW = (200, 30, 30), (30, 200, 30), (30, 30, 200), (230, 230, 30)

# Made-up "true" homography from frame pixels to tactical-map pixels
H_TRUE = np.array([[0.5, 0.05, 20.0], [0.02, 0.6, 10.0], [1e-4, 2e-4, 1.0]])
FRAME_SHAPE = (720, 1280, 3)


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeModel:
    """Stands in for a YOLO model: returns fixed boxes whatever the frame."""

    def __init__(self, xyxy=(), xywh=(), cls=(), conf=()):
        boxes = type("Boxes", (), {})()
        boxes.xyxy = FakeTensor(np.reshape(xyxy, (-1, 4)))
        boxes.xywh = FakeTensor(np.reshape(xywh, (-1, 4)))
        boxes.cls = FakeTensor(cls)
        boxes.conf = FakeTensor(conf)
        self._result = type("Result", (), {"boxes": boxes})()

    def __call__(self, frame, conf=0.5, verbose=False):
        return [self._result]


def keypoint_model():
    """Eight pitch keypoints placed where H_TRUE says they appear on the frame."""
    names = load_pitch_keypoint_names()
    positions = load_keypoint_positions()
    ids = list(range(8))
    map_pts = np.array([positions[names[i]] for i in ids], dtype=float)
    src = transform_points(np.linalg.inv(H_TRUE), map_pts)
    xywh = np.column_stack([src, np.full(len(ids), 20.0), np.full(len(ids), 20.0)])
    xyxy = np.column_stack([src - 10, src + 10])
    return FakeModel(xyxy, xywh, ids, [0.9] * len(ids))


def player_and_ball_model(player_map_pos, ball_map_pos):
    foot = transform_points(np.linalg.inv(H_TRUE), np.array([player_map_pos], float))[0]
    ball = transform_points(np.linalg.inv(H_TRUE), np.array([ball_map_pos], float))[0]
    w, h = 40.0, 100.0
    p_xywh = [foot[0], foot[1] - h / 2, w, h]
    p_xyxy = [foot[0] - w / 2, foot[1] - h, foot[0] + w / 2, foot[1]]
    b_xywh = [ball[0], ball[1], 10.0, 10.0]
    b_xyxy = [ball[0] - 5, ball[1] - 5, ball[0] + 5, ball[1] + 5]
    model = FakeModel([p_xyxy, b_xyxy], [p_xywh, b_xywh], [0, 2], [0.9, 0.8])
    return model, p_xyxy


def red_player_frame(p_xyxy):
    frame = np.full(FRAME_SHAPE, 90, dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in p_xyxy)
    frame[y1:y2, x1:x2] = RED[::-1]  # BGR
    return frame


def make_processor(players_model, settings=None):
    classifier = TeamClassifier(TeamSpec("Red FC", RED, GREEN), TeamSpec("Blue FC", BLUE, YELLOW))
    return FrameProcessor(players_model, keypoint_model(), classifier, settings)


def test_player_is_placed_on_the_tactical_map_and_assigned_a_team():
    model, p_xyxy = player_and_ball_model((150, 250), (100, 100))
    analysis = make_processor(model).analyze(red_player_frame(p_xyxy))
    (player,) = analysis.players
    assert player.team_idx == 0
    assert np.allclose(player.map_pos, (150, 250), atol=1.0)
    assert np.allclose(analysis.ball_map_pos, (100, 100), atol=1.0)
    assert [o.label for o in analysis.others] == ["ball"]


def test_homography_is_reused_while_the_pitch_is_stable():
    model, p_xyxy = player_and_ball_model((150, 250), (100, 100))
    processor = make_processor(model)
    frame = red_player_frame(p_xyxy)
    first = processor.analyze(frame).homography
    assert first is not None and processor.analyze(frame).homography is first


def test_no_keypoints_means_no_tactical_positions_but_still_renders():
    model, p_xyxy = player_and_ball_model((150, 250), (100, 100))
    classifier = TeamClassifier(TeamSpec("Red FC", RED, GREEN), TeamSpec("Blue FC", BLUE, YELLOW))
    processor = FrameProcessor(model, FakeModel(), classifier)
    frame = red_player_frame(p_xyxy)
    analysis = processor.analyze(frame)
    assert analysis.homography is None
    assert analysis.players[0].map_pos is None and analysis.ball_map_pos is None
    assert processor.render(frame, analysis).shape == (770, 1634, 3)


def test_render_shape_and_input_frame_is_left_untouched():
    model, p_xyxy = player_and_ball_model((150, 250), (100, 100))
    processor = make_processor(model)
    frame = red_player_frame(p_xyxy)
    before = frame.copy()
    out = processor.process(frame, fps=25)
    assert out.shape == (770, 1634, 3)
    assert np.array_equal(frame, before)


def test_ball_track_follows_the_ball_and_can_be_switched_off():
    model, p_xyxy = player_and_ball_model((150, 250), (100, 100))
    frame = red_player_frame(p_xyxy)
    processor = make_processor(model)
    processor.process(frame)
    processor.process(frame)
    assert len(processor.ball.dst) == 2

    off = PipelineSettings(plot=PlotOptions(show_ball_track=False))
    processor = make_processor(model, off)
    processor.process(frame)
    assert processor.ball.dst == []


def test_iter_processed_frames_respects_the_stride(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), _fourcc("mp4v"), 10.0, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()

    classifier = TeamClassifier(TeamSpec("A", RED, GREEN), TeamSpec("B", BLUE, YELLOW))
    processor = FrameProcessor(FakeModel(), FakeModel(), classifier)
    every_frame = list(iter_processed_frames(path, processor))
    every_third = list(iter_processed_frames(path, processor, frame_stride=3))
    assert [f.index for f in every_frame] == list(range(1, 11))
    assert [f.index for f in every_third] == [1, 4, 7, 10]
    assert every_frame[0].total == 10 and every_frame[0].image.ndim == 3
