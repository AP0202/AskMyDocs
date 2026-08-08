#!/usr/bin/env python
"""
Ask My Docs - single entrypoint.

Run this file to start the whole application (API + static frontend) on
one process/port, e.g.:

    python main.py

Then open http://localhost:8000 in your browser.

This mirrors the "one main file calls everything" structure of the
reference fitness-assistant project, while keeping frontend/ and backend/
as clearly separated folders.
"""

import os
import sys
from pathlib import Path

# Make `backend/` importable as the `app` package root, so backend/app/*
# can use clean absolute imports like `from app.config import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import uvicorn  # noqa: E402

from app.config import APP_HOST, APP_PORT  # noqa: E402


def main():
    reload = os.getenv("APP_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=reload,
        app_dir=str(Path(__file__).resolve().parent / "backend"),
    )


if __name__ == "__main__":
    main()
