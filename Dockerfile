FROM python:3.11-slim

# System-level audio + media dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade yt-dlp

# Copy the full project
COPY . .

# HF Spaces uses 7860; Render injects $PORT at runtime
ENV PORT=7860
# Without this Python block-buffers stdout when it is not a TTY, so every
# [Download]/[Beat]/[Cookies] line is swallowed and the platform logs show
# nothing but uvicorn's own output.
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "app.py"]
