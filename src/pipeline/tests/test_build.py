"""
Unit tests for the build stage (pipeline.build_df.build_dataframes).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import pytest

from pipeline.build_df.build_dataframes import (
    parse_ts,
    load_json,
    build_driver_map,
    add_distance_to_telemetry,
)


# ── parse_ts ───────────────────────────────────────────────────────────────

class TestParseTs:
    def test_standard_precision(self):
        ts = parse_ts("2026-02-11T10:30:45.123Z")
        assert ts.year == 2026
        assert ts.month == 2
        assert ts.day == 11
        assert ts.hour == 10
        assert ts.minute == 30
        assert ts.second == 45

    def test_high_precision(self):
        """F1 sometimes sends >6 fractional digits — should truncate cleanly."""
        ts = parse_ts("2026-02-11T10:30:45.1234567Z")
        assert ts.second == 45

    def test_no_fractional_seconds(self):
        ts = parse_ts("2026-02-11T10:30:45Z")
        assert ts.second == 45

    def test_timezone_aware(self):
        ts = parse_ts("2026-02-11T10:30:45.000Z")
        assert ts.tzinfo is not None


# ── load_json ──────────────────────────────────────────────────────────────

class TestLoadJson:
    def test_missing_file_returns_default(self, tmp_path):
        result = load_json(tmp_path / "nonexistent.json")
        assert result == []

    def test_missing_file_custom_default(self, tmp_path):
        result = load_json(tmp_path / "nonexistent.json", default={})
        assert result == {}

    def test_valid_json(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('[{"key": "value"}]')
        result = load_json(f)
        assert result == [{"key": "value"}]


# ── build_driver_map ──────────────────────────────────────────────────────

class TestBuildDriverMap:
    def test_basic_mapping(self):
        raw = [{"data": {
            "16": {"Tla": "LEC", "FullName": "Charles Leclerc",
                   "TeamName": "Ferrari", "TeamColour": "E80020"},
            "1":  {"Tla": "VER", "FullName": "Max Verstappen",
                   "TeamName": "Red Bull Racing", "TeamColour": "3671C6"},
        }}]
        drivers = build_driver_map(raw)
        assert "16" in drivers
        assert drivers["16"]["tla"] == "LEC"
        assert drivers["1"]["team"] == "Red Bull Racing"


# ── add_distance_to_telemetry ─────────────────────────────────────────────

class TestAddDistance:
    def test_empty_df(self):
        df = pd.DataFrame()
        result = add_distance_to_telemetry(df)
        assert "Distance" in result.columns
        assert len(result) == 0

    def test_distance_increases(self):
        """If a car travels at constant 360 km/h = 100 m/s for 1 second,
        distance should be ~100m."""
        base = datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "Date": pd.to_datetime([
                base,
                base + pd.Timedelta(seconds=1),
                base + pd.Timedelta(seconds=2),
            ]),
            "DriverNumber": ["1", "1", "1"],
            "Speed": [360.0, 360.0, 360.0],
        })
        result = add_distance_to_telemetry(df)
        assert result["Distance"].iloc[0] == 0.0
        assert abs(result["Distance"].iloc[1] - 100.0) < 1.0
        assert abs(result["Distance"].iloc[2] - 200.0) < 1.0
