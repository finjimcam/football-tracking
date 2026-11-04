# CPU image for the analysis pipeline (tools/ + workflows/). GPU/iGPU
# acceleration is host-specific and not configured here.
#   docker build -t football-tracking .
#   docker run --rm -v "$PWD:/app" football-tracking tools/track_identities.py --match-id <id>
# Mount the repo (videos, config/, .tmp/) at /app; credentials stay in your
# local .env, which .dockerignore keeps out of the image.
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
ENTRYPOINT ["python"]
CMD ["tools/track_identities.py", "--help"]
