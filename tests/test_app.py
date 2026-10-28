from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_app_starts_without_exceptions():
    """Runs the whole Streamlit script once (needs weights + demo video, so it is slow)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    app.run()
    assert not app.exception
