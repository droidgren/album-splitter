FROM python:3.12-slim

WORKDIR /app
# exiftool writes captions and dates into exported photos.
RUN apt-get update && apt-get install -y --no-install-recommends libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY albumsplit/ albumsplit/

# The commit this image was built from, shown by --version and in the web UI (no .git here).
# The GitHub workflow passes them; a local "docker build" leaves them empty (plain version).
ARG ALBUMSPLIT_COMMIT=""
ARG ALBUMSPLIT_COMMIT_DATE=""
ENV ALBUMSPLIT_COMMIT=$ALBUMSPLIT_COMMIT ALBUMSPLIT_COMMIT_DATE=$ALBUMSPLIT_COMMIT_DATE

# The start page keeps its recent-projects list in $HOME/.albumsplit; mount /config to keep it.
ENV HOME=/config PYTHONUNBUFFERED=1
RUN mkdir -p /config /scans /work /export

# QNAP Container Station: see docker-compose.yml. Plain docker, e.g.
#   docker run --rm -v /share/scans:/scans:ro -v /share/work:/work albumsplit detect /scans --out /work
# For --llm, also pass the API key: -e ANTHROPIC_API_KEY
# Review UI with the start page (then open http://<nas>:8000/):
#   docker run -d -p 8000:8000 -v /share/scans:/scans:ro -v /share/work:/work -v /share/photos:/export albumsplit
# Export approved pages:
#   docker run --rm -v /share/scans:/scans:ro -v /share/work:/work -v /share/photos:/export \
#     albumsplit export /work --input /scans --out /export
EXPOSE 8000
HEALTHCHECK --interval=1m --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/recent', timeout=4)" || exit 1
ENTRYPOINT ["python", "-m", "albumsplit"]
CMD ["review", "--host", "0.0.0.0", "--port", "8000"]
