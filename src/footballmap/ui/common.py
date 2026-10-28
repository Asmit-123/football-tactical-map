"""Small Streamlit helpers shared by the UI modules."""

from __future__ import annotations

from typing import Any

# Session-state keys holding the four picked kit colours (hex strings)
COLOR_KEYS = ("color_t1_player", "color_t1_gk", "color_t2_player", "color_t2_gk")


def show_image(target: Any, image: Any, **kwargs: Any) -> None:
    """Show an image stretched to the container width, on old and new Streamlit versions."""
    try:
        target.image(image, width="stretch", **kwargs)
    except Exception:  # older Streamlit: 'width' only accepts pixels
        target.image(image, use_container_width=True, **kwargs)
