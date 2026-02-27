#!/usr/bin/env python3
"""
Monitor the live telemetry recording status.

Usage:
    python src/pipeline/01_parse/monitor_recording.py <recording_file>
    python src/pipeline/01_parse/monitor_recording.py   # uses default path
"""

import argparse
import os
import time
from datetime import datetime

DEFAULT_RECORDING_FILE = "2026/Bahrein/test_day1.txt"


def get_file_size(filepath):
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except FileNotFoundError:
        return 0


def monitor(filepath, interval=5):
    """Monitor a growing recording file, tracking new lines incrementally."""
    print("🏎️  F1 Live Telemetry Recording Monitor")
    print("=" * 60)
    print(f"Monitoring: {filepath}")
    print("Press Ctrl+C to stop monitoring\n")

    last_pos = 0
    total_lines = 0
    last_timestamp = None

    try:
        while True:
            try:
                with open(filepath, 'r') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
            except FileNotFoundError:
                print(f"\r⏳ Waiting for file: {filepath}", end='', flush=True)
                time.sleep(interval)
                continue

            new_count = len(new_lines)
            total_lines += new_count

            # Extract timestamp from the last new line
            if new_lines:
                last_line = new_lines[-1].strip()
                if last_line.endswith("']"):
                    parts = last_line.split("'")
                    if len(parts) >= 4:
                        last_timestamp = parts[-2]

            size_mb = get_file_size(filepath)
            rate_indicator = "📈" if new_count > 0 else "⏸️"

            print(f"\r{rate_indicator} Messages: {total_lines:,} | Size: {size_mb:.2f} MB | "
                  f"Latest: {last_timestamp or 'N/A'} | +{new_count}/{interval}s",
                  end='', flush=True)

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n✅ Monitoring stopped")
        print(f"Final count: {total_lines:,} messages ({get_file_size(filepath):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor F1 live telemetry recording status"
    )
    parser.add_argument('file', nargs='?', default=DEFAULT_RECORDING_FILE,
                        help=f'Recording file to monitor (default: {DEFAULT_RECORDING_FILE})')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='Polling interval in seconds (default: 5)')
    args = parser.parse_args()

    monitor(args.file, args.interval)


if __name__ == "__main__":
    main()
