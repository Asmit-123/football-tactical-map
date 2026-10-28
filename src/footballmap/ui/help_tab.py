"""'How to use?' tab."""

from __future__ import annotations

import streamlit as st

from footballmap import __version__


def render_help_tab() -> None:
    st.header(":blue[Welcome!]")
    st.subheader("Main Application Functionalities:", divider="blue")
    st.markdown(
        """
        1. Football players, referee, and ball detection.
        2. Players team prediction.
        3. Estimation of players and ball positions on a tactical map.
        4. Ball Tracking.
        """
    )
    st.subheader("How to use?", divider="blue")
    st.markdown(
        """
        **Two demo videos are loaded automatically, with recommended settings and hyperparameters.**
        1. Upload a video to analyse using the sidebar "Browse files" button.
        2. Enter the team names that correspond to the uploaded video in the sidebar.
        3. Open the "Team Colors" tab.
        4. Select a frame where players and goalkeepers from both teams can be detected.
        5. Follow the instructions on the page to pick each team's colors.
        6. Go to the "Model Hyperparameters & Detection" tab, adjust the hyperparameters and
           select the annotation options (the defaults are recommended).
        7. Run Detection!
        8. If "Save output" was selected, the saved video can be found in the `outputs` folder.
        """
    )
    st.write(f"Version {__version__}")
