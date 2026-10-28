"""Streamlit entry point:  streamlit run app.py"""

import sys
from pathlib import Path

# Lets `streamlit run app.py` work without `pip install -e .` (e.g. on a hosting service)
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from footballmap.models import load_models, missing_model_files  # noqa: E402
from footballmap.ui.detection_tab import render_detection_tab  # noqa: E402
from footballmap.ui.help_tab import render_help_tab  # noqa: E402
from footballmap.ui.runner import run_detection  # noqa: E402
from footballmap.ui.sidebar import render_sidebar  # noqa: E402
from footballmap.ui.team_colors import render_team_colors_tab  # noqa: E402


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return load_models()


def main() -> None:
    st.set_page_config(
        page_title="AI Powered Web Application for Football Tactical Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Football Players Detection With Team Prediction & Tactical Map")
    st.subheader(":red[Works only with Tactical Camera footage]")

    state = render_sidebar()
    if state is None:
        st.stop()

    missing = missing_model_files()
    if missing:
        st.error(
            "Model weights not found: "
            + ", ".join(p.name for p in missing)
            + ". Put them in the `models/` folder (see models/README.md)."
        )
        st.stop()
    models = get_models()

    help_tab, colors_tab, detection_tab = st.tabs(
        ["How to use?", "Team Colors", "Model Hyperparameters & Detection"]
    )
    with help_tab:
        render_help_tab()
    with colors_tab:
        teams = render_team_colors_tab(state, models)
    with detection_tab:
        request = render_detection_tab(state)

    frame_slot = st.empty()
    if request.start and not request.stop:
        run_detection(state, teams, request, models, frame_slot)


if __name__ == "__main__":
    main()
