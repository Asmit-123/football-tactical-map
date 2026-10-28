import cv2
import numpy as np

from footballmap import video


def make_video(path, frames=10, width=64, height=48):
    writer = cv2.VideoWriter(str(path), video._fourcc("mp4v"), 10.0, (width, height))
    for i in range(frames):
        writer.write(np.full((height, width, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_probe_reports_size_and_frame_count(tmp_path):
    info = video.probe(make_video(tmp_path / "clip.mp4"))
    assert (info.frame_count, info.width, info.height) == (10, 64, 48)
    assert info.fps > 0


def test_probe_rejects_missing_file(tmp_path):
    try:
        video.probe(tmp_path / "nope.mp4")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_read_frame_is_one_based(tmp_path):
    path = make_video(tmp_path / "clip.mp4")
    first, third = video.read_frame(path, 1), video.read_frame(path, 3)
    assert first.shape == (48, 64, 3)
    assert third.mean() > first.mean()  # later frames are brighter in the test clip
    assert video.read_frame(path, 500) is None


def test_save_upload_is_idempotent(tmp_path):
    first = video.save_upload(b"abc", ".MP4", directory=tmp_path)
    second = video.save_upload(b"abc", ".MP4", directory=tmp_path)
    other = video.save_upload(b"abd", ".mp4", directory=tmp_path)
    assert first == second and first != other
    assert first.suffix == ".mp4" and first.read_bytes() == b"abc"


def test_output_path_default_names_count_up(tmp_path):
    assert video.output_path(None, tmp_path).name == "detect_1.mp4"
    (tmp_path / "detect_1.mp4").write_bytes(b"x")
    assert video.output_path("", tmp_path).name == "detect_2.mp4"


def test_output_path_sanitises_and_avoids_overwriting(tmp_path):
    assert video.output_path("my match/final.mp4", tmp_path).name == "my match_final.mp4"
    (tmp_path / "game.mp4").write_bytes(b"x")
    assert video.output_path("game", tmp_path).name == "game_1.mp4"


def test_video_sink_writes_a_readable_file(tmp_path):
    path = tmp_path / "out.mp4"
    with video.VideoSink(path, fps=10.0) as sink:
        for i in range(5):
            sink.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
    assert sink.frames_written == 5
    assert video.probe(path).frame_count == 5
