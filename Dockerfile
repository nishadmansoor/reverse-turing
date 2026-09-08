# Reverse Turing Test — container image
#
# Targets Hugging Face Spaces (Docker SDK), but is a plain container and runs
# anywhere: `docker run -p 7860:7860 <image>`.
#
# Two deliberate choices keep this deployable:
#   * CPU-only torch. The default wheel pulls CUDA libraries that add well over
#     a gigabyte and are dead weight on a CPU Space.
#   * The app loads prebuilt artifacts and never touches the raw dataset, so no
#     CSVs ship in the image. Run src/build_artifacts.py before building.

FROM python:3.11-slim

# libgomp1 is required by opencv/torch; the rest of the usual OpenCV system
# deps are unnecessary with the headless wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000; matching it keeps the caches writable.
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Install dependencies first so layer caching survives source edits.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Model weights and artifacts change less often than templates.
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser artifacts/ ./artifacts/
COPY --chown=appuser:appuser src/features.py src/model_defs.py ./src/
COPY --chown=appuser:appuser app/ ./app/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# One worker with threads: each worker would otherwise hold its own ~270 MB
# copy of the transformer. Inference releases the GIL, so threads serve
# concurrent requests fine. --preload loads the models before forking.
CMD ["gunicorn", "app.app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--preload", \
     "--access-logfile", "-"]
