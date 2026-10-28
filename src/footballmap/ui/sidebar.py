"""Sidebar: demo selection, video upload and team names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from footballmap.config import DEMO_PRESETS, VIDEO_EXTENSIONS, DemoPreset
from footballmap.ui.common import COLOR_KEYS
from footballmap.video import save_upload


@dataclass(frozen=True)
class SidebarState:
    video_path: Path
    preset: DemoPreset
    team1_name: str
    team2_name: str


def _apply_preset(preset_name: str) -> None:
    """Reset team names and kit colours to the selected demo's values."""
    preset = DEMO_PRESETS[preset_name]
    st.session_state["team1_name"] = preset.team1_name
    st.session_state["team2_name"] = preset.team2_name
    colors = (
        preset.team1_player_color,
        preset.team1_gk_color,
        preset.team2_player_color,
        preset.team2_gk_color,
    )
    for key, color in zip(COLOR_KEYS, colors, strict=True):
        st.session_state[key] = color
    st.session_state["_active_preset"] = preset_name


def _stored_upload(uploaded) -> Path:
    """Write the upload to disk once and reuse the path on later reruns."""
    cache = st.session_state.setdefault("_upload_paths", {})
    key = (uploaded.name, uploaded.size)
    path = cache.get(key)
    if path is None or not Path(path).is_file():
        suffix = Path(uploaded.name).suffix or ".mp4"
        path = save_upload(uploaded.getvalue(), suffix)
        cache[key] = path
    return Path(path)


def render_sidebar() -> SidebarState | None:
    """Draws the sidebar; returns None if there is no usable video."""
    st.sidebar.title("Main Settings")
    preset_name = st.sidebar.radio("Select Demo Video", list(DEMO_PRESETS), horizontal=True)
    if st.session_state.get("_active_preset") != preset_name:
        _apply_preset(preset_name)
    preset = DEMO_PRESETS[preset_name]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Video Upload")
    uploaded = st.sidebar.file_uploader("Upload a video file", type=list(VIDEO_EXTENSIONS))

    if uploaded is not None:
        video_path, caption = _stored_upload(uploaded), "Input video"
    else:
        video_path, caption = preset.video_path, "Demo video"
        if not video_path.is_file():
            st.sidebar.error(f"Demo video not found: {video_path.name} (see assets/demo/README.md)")
            return None
    st.sidebar.text(caption)
    st.sidebar.video(str(video_path))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Team Names")
    team1_name = st.sidebar.text_input("First Team Name", key="team1_name")
    team2_name = st.sidebar.text_input("Second Team Name", key="team2_name")
    st.sidebar.markdown("---")
    return SidebarState(video_path, preset, team1_name, team2_name)
