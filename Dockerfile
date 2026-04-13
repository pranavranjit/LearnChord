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
EXPOSE 7860

CMD ["python", "app.py"]
