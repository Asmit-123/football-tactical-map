# Football Match Analysis with Tactical Map

A web app that analyses football footage from a tactical camera and builds a live tactical map:

1. Detects players, referees and the ball (YOLOv8).
2. Predicts each player's team from the colours of the jersey.
3. Maps players and the ball onto a 2D pitch using a homography estimated from detected pitch keypoints.
4. Tracks the ball and draws its recent path on the map.

> Works only with tactical camera footage (wide view of the whole pitch).

This is a rebuild of the original project with a cleaner structure, a Streamlit-free processing
core, tests and a deployable setup. The prototype behaves the same as before.

## Quick start

Requires Python 3.14 (any version your PyTorch build supports should work).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
pip install -e .
streamlit run app.py
```

Before the first run, place the model weights in `models/` and the demo videos in
`assets/demo/` (see the README in each folder). `git lfs install` before cloning or adding them.

## Project layout

```
app.py                      Streamlit entry point
src/footballmap/
  config.py                 paths, default hyperparameters, demo presets
  models.py                 loads the two YOLO models
  homography.py             frame -> tactical map transformation
  teams.py                  team prediction from jersey colours
  ball.py                   ball track history
  annotate.py               drawing helpers
  pipeline.py               analyse + render a frame, run a whole video (no Streamlit)
  video.py  colorpick.py    video I/O and colour-picking helpers
  ui/                       Streamlit pages (sidebar, tabs, runner)
assets/                     tactical map image, keypoint positions, class names, demo videos
models/                     YOLO weights (Git LFS)
tests/                      pytest suite
docs/manual-test-checklist.md
```

## Tests

```bash
pytest -m "not slow"     # fast unit tests, no models needed
pytest                   # also starts the whole app once (needs weights + demo video)
ruff check .             # lint
```

Manual checks: [docs/manual-test-checklist.md](docs/manual-test-checklist.md).

## Deploying

The app needs a long-running Python server and PyTorch, so serverless platforms such as Vercel
are not suitable. It is deployed on **Streamlit Community Cloud** straight from this repository:
`requirements.txt` (CPU-only PyTorch), `packages.txt` (system libraries for OpenCV) and
`.streamlit/config.toml` are the only deployment files. GitHub also runs CI
(`.github/workflows/ci.yml`).

Tips for the hosted demo: the free tier is a small shared CPU, so set *Process every Nth frame*
to 3 or more, and use short clips. The app goes to sleep after 12 hours without visitors and
takes a minute to wake up.
