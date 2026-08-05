FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Dependencies first so code changes do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Build the dataset and train the model at image build time.
#
# Why not ship a pre-trained .joblib? Model files are gitignored, so a cloud
# build cloning the repo would produce an image with no model in it and the API
# would boot degraded. Training here takes seconds and makes the image
# self-contained and reproducible: the same Dockerfile always yields the same
# model. If you later train something heavy (the transformer), push the
# artefact to object storage and download it here instead.
#
# IMPORTANT: with no data/raw/reviews.csv in the repo this trains on the
# synthetic bootstrap sample. Fine for a live demo, meaningless as a result.
# To deploy a real model, commit a licensed dataset to data/raw/ (uncomment
# the COPY below) or fetch it in this step.
# COPY data/raw ./data/raw
RUN python -m src.data.build_dataset && \
    python -m src.training.train_baseline && \
    test -f models/baseline.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\",8000)}/health')"

# Render injects $PORT and expects the process to bind to it. Shell form so the
# variable is expanded at runtime rather than baked in.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
