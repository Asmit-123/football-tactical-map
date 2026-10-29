"""'Model Hyperparameters & Detection' tab."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from footballmap.config import BallTrackParams, DetectionParams, PlotOptions
from footballmap.pipeline import PipelineSettings
from footballmap.ui.sidebar import SidebarState


@dataclass(frozen=True)
class DetectionRequest:
    settings: PipelineSettings
    save_output: bool
    output_name: str | None
    start: bool
    stop: bool


def render_detection_tab(state: SidebarState) -> DetectionRequest:
    defaults = DetectionParams()
    ball_defaults = BallTrackParams()

    left, right = st.columns(2)
    with left:
        player_conf = st.slider(
            "Players Detection Confidence Threshold", 0.0, 1.0, defaults.player_conf
        )
        keypoint_conf = st.slider(
            "Field Keypoints Detection Confidence Threshold", 0.0, 1.0, defaults.keypoint_conf
        )
        displacement_tol = st.slider(
            "Keypoints Displacement Tolerance (mean squared pixels)",
            min_value=0,
            max_value=100,
            value=int(defaults.keypoint_displacement_tol),
            help="Maximum allowed average displacement of the field keypoints between the "
            "current and previous detections. Used to decide whether to update the homography "
            "matrix.",
        )
    with right:
        num_palette_colors = st.slider(
            "Number of palette colors",
            min_value=1,
            max_value=5,
            step=1,
            value=defaults.num_palette_colors,
            help="How many colors to extract from each detected player's bounding box. "
            "Used for team prediction.",
        )
        frame_stride = st.slider(
            "Process every Nth frame",
            min_value=1,
            max_value=10,
            value=defaults.frame_stride,
            help="1 processes every frame. Higher values are faster (useful on a CPU) but "
            "the output looks choppier.",
        )
        st.markdown("---")
        save_output = st.checkbox(
            "Save output (adds a download button when detection ends)", value=True
        )
        output_name = None
        if save_output:
            output_name = st.text_input(
                "File Name (Optional)", placeholder="Enter output video file name."
            )
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        reset_after = st.number_input(
            "Ball track reset threshold (frames)",
            min_value=1,
            max_value=10000,
            value=ball_defaults.reset_after_frames,
            help="After how many frames without a ball detection should the track be reset?",
        )
        max_jump = st.number_input(
            "Ball track distance threshold (pixels)",
            min_value=1,
            max_value=1280,
            value=ball_defaults.max_jump_px,
            help="Maximum allowed distance between two consecutive ball detections to keep "
            "the current track.",
        )
        max_length = st.number_input(
            "Maximum ball track length (Nbr. detections)",
            min_value=1,
            max_value=1000,
            value=ball_defaults.max_length,
            help="Maximum number of ball detections kept in the tracking history.",
        )
    with right:
        st.write("Annotation options:")
        col_a, col_b = st.columns(2)
        with col_a:
            show_keypoints = st.toggle("Show Keypoints Detections", value=False)
            show_players = st.toggle("Show Players Detections", value=True)
        with col_b:
            show_palette = st.toggle("Show Color Palettes", value=True)
            show_ball_track = st.toggle("Show Ball Tracks", value=True)
        st.markdown("---")
        _, start_col, stop_col, _ = st.columns([1.5, 1, 1, 1])
        with start_col:
            names_missing = not state.team1_name or not state.team2_name
            start = st.button("Start Detection", disabled=names_missing)
        with stop_col:
            stop = st.button("Stop Detection", disabled=not start)

    settings = PipelineSettings(
        detection=DetectionParams(
            player_conf=player_conf,
            keypoint_conf=keypoint_conf,
            keypoint_displacement_tol=float(displacement_tol),
            num_palette_colors=num_palette_colors,
            frame_stride=frame_stride,
        ),
        ball=BallTrackParams(int(reset_after), int(max_jump), int(max_length)),
        plot=PlotOptions(show_keypoints, show_palette, show_ball_track, show_players),
    )
    return DetectionRequest(settings, save_output, output_name, start, stop)
