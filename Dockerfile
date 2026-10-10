# Overtake: API + pit-wall dashboard in one image (Design.md Section 8).
#
# Bakes in the processed race tables (data/processed, ~16 MB) and the trained
# models (data/models, ~5 MB), so the image runs with no network and no
# FastF1 calls. The 12 GB FastF1 cache is NOT baked in; it is only needed by
# the Driver/corner view and is mounted by docker-compose.yml when present.
#
#   docker build -t overtake .
#   docker run -p 8000:8000 overtake        # http://localhost:8000

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp: OpenMP runtime XGBoost needs on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the default PyPI wheel pulls in CUDA, ~2 GB more),
# then everything else at the pinned versions.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY backend ./backend
COPY frontend ./frontend
COPY data/processed ./data/processed
COPY data/models ./data/models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
