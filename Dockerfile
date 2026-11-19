FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY albumsplit/ albumsplit/

# Mount scans read-only and a writable work dir, e.g.
#   docker run --rm -v /share/scans:/scans:ro -v /share/work:/work albumsplit detect /scans --out /work
ENTRYPOINT ["python", "-m", "albumsplit"]
