# Hugging Face Docker Space (CPU). Build is untested until the first push: if a package has no
# wheel for this Python version, change the base image to python:3.12-slim and rebuild.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Hugging Face runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# CPU-only torch first (the default Linux wheel pulls gigabytes of CUDA libraries)
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt \
    # ultralytics installs the GUI build of OpenCV, which needs system libraries this image lacks
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --user opencv-python-headless

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
