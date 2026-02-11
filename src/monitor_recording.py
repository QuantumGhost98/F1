#!/usr/bin/env python3
"""
Monitor the live telemetry recording status.

Usage:
    python src/monitor_recording.py
"""

import os
import time
import sys
from datetime import datetime

RECORDING_FILE = "/Users/PietroPaolo/Desktop/GitHub/F1/2026/Bahrein/test_day1.txt"

def get_file_size(filepath):
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except FileNotFoundError:
        return 0

def get_line_count(filepath):
    """Get number of lines in file."""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def get_latest_timestamp(filepath):
    """Get the timestamp from the last line."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                # Extract timestamp from: ['Category', {...}, '2026-02-11T08:54:42.221Z']
                if last_line.endswith("']"):
                    parts = last_line.split("'")
                    if len(parts) >= 4:
                        return parts[-2]
        return None
    except Exception:
        return None

def main():
    print("🏎️  F1 Live Telemetry Recording Monitor")
    print("=" * 60)
    print(f"Monitoring: {RECORDING_FILE}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_count = 0
    
    try:
        while True:
            count = get_line_count(RECORDING_FILE)
            size_mb = get_file_size(RECORDING_FILE)
            timestamp = get_latest_timestamp(RECORDING_FILE)
            
            new_messages = count - last_count
            rate_indicator = "📈" if new_messages > 0 else "⏸️"
            
            print(f"\r{rate_indicator} Messages: {count:,} | Size: {size_mb:.2f} MB | "
                  f"Latest: {timestamp or 'N/A'} | +{new_messages}/5s", end='', flush=True)
            
            last_count = count
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        print(f"Final count: {count:,} messages ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
