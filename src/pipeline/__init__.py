"""
F1 Telemetry Pipeline
=====================
Three-stage pipeline: capture → decode → build → analyze.

Quick start:
    from pipeline import load_pipeline
    session = load_pipeline("2026/Bahrein/test_day2.txt")
"""

from pipeline.load_pipeline import load_pipeline
from pipeline.load_session import load_session

__all__ = ["load_pipeline", "load_session"]
