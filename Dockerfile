# Reverse Turing Test — container image
#
# A plain container: `docker run -p 7860:7860 <image>`. Runs on any host that
# gives it ~512 MB.
#
# Three choices keep it small enough for a free tier:
#   * int8 ONNX graphs (71 MB) instead of fp32 PyTorch weights (293 MB)
#   * onnxruntime instead of torch — no CUDA libraries, no ~800 MB runtime
#   * prebuilt artifacts, so no dataset ships in the image
#
# Run `python src/quantize.py` and `python src/build_artifacts.py` before
# building.

FROM python:3.11-slim

# libgomp1 is onnxruntime's OpenMP dependency; opencv's headless wheel needs
# nothing further.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Many hosts (including HF Spaces) run containers as uid 1000; matching it
# keeps the caches writable.
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

# Dependencies first so layer caching survives source edits.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Only the artifacts the app actually loads. The fp32 PyTorch weights in
# models/bert_classifier/ and models/cnn_classifier.pt are training outputs and
# are deliberately excluded — copying them would quadruple the image.
COPY --chown=appuser:appuser models/onnx/ ./models/onnx/
COPY --chown=appuser:appuser models/stylometric_classifier.pkl \
     models/meta_classifier_int8.pkl ./models/
COPY --chown=appuser:appuser artifacts/ ./artifacts/
COPY --chown=appuser:appuser results/evaluation_raid_int8.json ./results/
COPY --chown=appuser:appuser src/features.py src/model_defs.py ./src/
COPY --chown=appuser:appuser app/ ./app/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Shell form so $PORT expands: hosts inject their own port (Render, Cloud Run,
# Fly), and 7860 is only the local default.
#
# One worker with threads: a second worker would duplicate the loaded graphs in
# memory. onnxruntime releases the GIL during inference, so threads serve
# concurrent requests fine. --preload loads everything before forking.
CMD gunicorn app.app:app \
      --bind "0.0.0.0:${PORT:-7860}" \
      --workers 1 \
      --threads 4 \
      --timeout 120 \
      --preload \
      --access-logfile -
