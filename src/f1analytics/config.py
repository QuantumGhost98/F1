"""
Package-level configuration constants for f1analytics.

All paths are derived from __file__ so the package works 
regardless of where it is installed.
"""
import logging
from pathlib import Path

# Repo root: src/f1analytics/config.py → parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
LOGO_PATH = REPO_ROOT / "logo-square.png"
ATTRIBUTION = "Provided by: Pietro Paolo Melella"

# Package-level logger (all modules should use this instead of print())
logger = logging.getLogger("f1analytics")
