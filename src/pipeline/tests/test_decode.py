"""
Unit tests for the decode stage (pipeline.decode.decode_telemetry).
"""

import base64
import json
import zlib

import pytest

from pipeline.decode.decode_telemetry import (
    decode_z_data,
    parse_line,
    process_entry,
)


# ── decode_z_data ──────────────────────────────────────────────────────────

class TestDecodeZData:
    """Round-trip and edge-case tests for base64+zlib decoding."""

    def test_round_trip_dict(self):
        """Encode a dict, then decode it — values must survive the round trip."""
        original = {"Speed": 320, "RPM": 12500, "Gear": 8}
        payload = json.dumps(original).encode()
        compressed = zlib.compress(payload)
        encoded = base64.b64encode(compressed).decode()

        result = decode_z_data(encoded)
        assert result == original

    def test_round_trip_list(self):
        """Encode a list payload."""
        original = [1, 2, 3]
        payload = json.dumps(original).encode()
        compressed = zlib.compress(payload)
        encoded = base64.b64encode(compressed).decode()

        result = decode_z_data(encoded)
        assert result == original

    def test_invalid_base64_returns_error_string(self):
        result = decode_z_data("!!!not_base64!!!")
        assert isinstance(result, str)
        assert "DECODE_ERROR" in result

    def test_empty_string(self):
        result = decode_z_data("")
        assert isinstance(result, str)
        assert "DECODE_ERROR" in result


# ── parse_line ─────────────────────────────────────────────────────────────

class TestParseLine:
    def test_valid_list(self):
        line = "['TimingData', {'Lines': {}}, '2026-02-11T10:00:00.000Z']"
        result = parse_line(line)
        assert isinstance(result, list)
        assert result[0] == "TimingData"

    def test_empty_line(self):
        assert parse_line("") is None
        assert parse_line("   ") is None

    def test_invalid_syntax(self):
        assert parse_line("not a python literal") is None


# ── process_entry ──────────────────────────────────────────────────────────

class TestProcessEntry:
    def test_non_compressed_dict(self):
        entry = ["WeatherData", {"AirTemp": "28"}, "2026-02-11T10:00:00Z"]
        topic, result = process_entry(entry)
        assert topic == "WeatherData"
        assert result["data"]["AirTemp"] == "28"
        assert result["timestamp"] == "2026-02-11T10:00:00Z"

    def test_compressed_z_topic(self):
        original = {"Entries": [{"Utc": "2026-02-11T10:00:00Z"}]}
        payload = json.dumps(original).encode()
        compressed = zlib.compress(payload)
        encoded = base64.b64encode(compressed).decode()

        entry = ["CarData.z", encoded, "2026-02-11T10:00:00Z"]
        topic, result = process_entry(entry)
        assert topic == "CarData"  # .z stripped
        assert result["data"] == original

    def test_short_entry(self):
        topic, result = process_entry(["only_one"])
        assert topic is None
        assert result is None

    def test_none_entry(self):
        topic, result = process_entry(None)
        assert topic is None
