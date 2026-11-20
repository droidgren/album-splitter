FROM python:3.12-slim

WORKDIR /app
# exiftool writes captions and dates into exported photos.
RUN apt-get update && apt-get install -y --no-install-recommends libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY albumsplit/ albumsplit/

# Mount scans read-only and a writable work dir, e.g.
#   docker run --rm -v /share/scans:/scans:ro -v /share/work:/work albumsplit detect /scans --out /work
# For --llm, also pass the API key: -e ANTHROPIC_API_KEY
# Review UI (then open http://<nas>:8000/):
#   docker run --rm -p 8000:8000 -v /share/scans:/scans:ro -v /share/work:/work albumsplit #     review /work --input /scans --host 0.0.0.0
# Export approved pages:
#   docker run --rm -v /share/scans:/scans:ro -v /share/work:/work -v /share/photos:/export \
#     albumsplit export /work --input /scans --out /export
EXPOSE 8000
ENTRYPOINT ["python", "-m", "albumsplit"]
