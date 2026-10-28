"""Runs the detection pipeline over the selected video and streams frames to the page."""

from __future__ import annotations

import streamlit as st

from footballmap.models import Models
from footballmap.pipeline import FrameProcessor, iter_processed_frames
from footballmap.teams import TeamClassifier, TeamSpec
from footballmap.ui.detection_tab import DetectionRequest
from footballmap.ui.sidebar import SidebarState
from footballmap.video import VideoSink, output_path, probe


def run_detection(
    state: SidebarState,
    teams: tuple[TeamSpec, TeamSpec],
    request: DetectionRequest,
    models: Models,
    frame_slot,
) -> None:
    settings = request.settings
    stride = settings.detection.frame_stride
    classifier = TeamClassifier(*teams, num_palette_colors=settings.detection.num_palette_colors)
    processor = FrameProcessor(models.players, models.keypoints, classifier, settings)

    sink = None
    saved_to = None
    if request.save_output:
        saved_to = output_path(request.output_name)
        sink = VideoSink(saved_to, fps=probe(state.video_path).fps / stride)

    progress = st.progress(0, text="Detection starting.")
    st.toast("Detection Started!")
    try:
        for item in iter_processed_frames(state.video_path, processor, stride):
            percent = min(int(item.index / item.total * 100), 100) if item.total else 0
            progress.progress(percent, text=f"Detection in progress ({percent}%)")
            frame_slot.image(item.image, channels="BGR")
            if sink is not None:
                sink.write(item.image)
    finally:  # also runs when the user presses Stop (Streamlit interrupts the script)
        progress.empty()
        if sink is not None:
            sink.close()

    st.toast("Detection Completed!")
    if saved_to is not None:
        st.success(f"Saved to {saved_to}")
