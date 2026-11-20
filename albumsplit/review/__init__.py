"""Local web UI to review and fix detected pages: python -m albumsplit review <work_dir>."""

from .server import create_app

__all__ = ["create_app"]
