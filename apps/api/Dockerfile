# Build context = repo root (see render.yaml). This preserves the repo layout inside
# the image so file-relative paths (brand docs, data dir) resolve as they do locally.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# ffmpeg powers the storyboard -> slideshow video.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY apps/api/requirements.txt ./apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

COPY apps/api/app ./apps/api/app
COPY data/brand ./data/brand

EXPOSE 8000
# Render injects $PORT. Brand docs are auto-ingested on startup (app/main.py).
CMD ["sh", "-c", "uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port ${PORT:-8000}"]
