"""'Team Colors' tab: pick each team's player and goalkeeper kit colours from a frame."""

from __future__ import annotations

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from footballmap.colorpick import build_thumbnail_grid, pick_color, player_thumbnails
from footballmap.models import Models
from footballmap.teams import TeamSpec
from footballmap.ui.common import COLOR_KEYS, show_image
from footballmap.ui.sidebar import SidebarState
from footballmap.video import probe, read_frame

DETECTION_CONF = 0.7  # confidence used when looking for players to pick colours from


def _thumbnail_grid(state: SidebarState, models: Models, frame, frame_number: int):
    """Detect players in the chosen frame (cached until the frame or video changes)."""
    cache_key = (str(state.video_path), frame_number)
    if st.session_state.get("_thumb_key") != cache_key:
        with st.spinner("Detecting players in selected frame.."):
            boxes = models.players(frame, conf=DETECTION_CONF, verbose=False)[0].boxes
            thumbs = player_thumbnails(frame, boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy())
            st.session_state["_thumb_grid"] = build_thumbnail_grid(thumbs)
        st.session_state["_thumb_key"] = cache_key
    return st.session_state["_thumb_grid"]


def render_team_colors_tab(state: SidebarState, models: Models) -> tuple[TeamSpec, TeamSpec]:
    """Draws the tab; returns the two teams with their currently selected colours."""
    info = probe(state.video_path)
    names = (state.team1_name, state.team2_name)
    labels = [
        f"{names[0]} P color",
        f"{names[0]} GK color",
        f"{names[1]} P color",
        f"{names[1]} GK color",
    ]

    left, right = st.columns(2)
    with left:
        frame_number = st.slider(
            "Select frame",
            min_value=1,
            max_value=max(info.frame_count, 2),
            step=1,
            help="Select frame to pick team colors from",
        )
        frame = read_frame(state.video_path, frame_number)
        if frame is None:
            st.error("Could not read this frame from the video.")
            grid = None
        else:
            grid = _thumbnail_grid(state, models, frame, frame_number)

        st.write("Detected players")
        click = None
        if grid is None:
            st.info("No players detected in this frame. Try another frame.")
        else:
            click = streamlit_image_coordinates(grid, key="player_thumbnails")

        st.markdown("---")
        active = st.radio(
            "Select which team color to pick from the image above",
            options=list(range(4)),
            format_func=labels.__getitem__,
            horizontal=True,
            help="Choose the team color you want to pick, then click a player above. "
            "Colors are shown in the boxes below.",
        )

        # Apply a *new* click to the selected slot (before the colour pickers are created)
        if click is not None and click != st.session_state.get("_last_click"):
            picked = pick_color(grid, click["x"], click["y"])
            if picked:
                st.session_state[COLOR_KEYS[active]] = picked
            st.session_state["_last_click"] = click

        st.write("Boxes below can be used to manually adjust selected colors.")
        for column, key, label in zip(st.columns(4), COLOR_KEYS, labels, strict=True):
            with column:
                st.color_picker(label, key=key)
    st.markdown("---")

    with right:
        if frame is not None:
            show_image(st, frame, channels="BGR")

    colors = [st.session_state[key] for key in COLOR_KEYS]
    return (
        TeamSpec.from_hex(names[0], colors[0], colors[1]),
        TeamSpec.from_hex(names[1], colors[2], colors[3]),
    )
