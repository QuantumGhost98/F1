#!/usr/bin/env python3
"""
F1 Telemetry Pipeline — End-to-End
====================================
Runs the full pipeline in one go:
  1. Decode raw capture file (base64+zlib → JSON)
  2. Build DataFrames (JSON → pandas)

Usage:
    # Process a recorded file
    python src/pipeline/run_pipeline.py 2026/Bahrein/test_day1_FULL.txt

    # Specify output directory
    python src/pipeline/run_pipeline.py 2026/Bahrein/test_day1_FULL.txt --output 2026/Bahrein/

    # Watch mode: continuously process as file grows (during live session)
    python src/pipeline/run_pipeline.py 2026/Bahrein/test_day1_FULL.txt --watch

    # Watch mode with custom refresh interval (seconds)
    python src/pipeline/run_pipeline.py 2026/Bahrein/test_day1_FULL.txt --watch --interval 30
"""

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

from pipeline.log import logger
from pipeline.decode.decode_telemetry import process_file
from pipeline.build_df.build_dataframes import (
    load_json, build_driver_map, build_laps_df,
    build_telemetry_df, build_position_df, add_distance_to_telemetry
)


def run_decode(input_file: str, output_dir: str) -> str:
    """Stage 1: Decode raw capture → JSON files."""
    print("=" * 70)
    print("  STAGE 1 — DECODE (base64 + zlib → JSON)")
    print("=" * 70)
    process_file(input_file, output_dir)
    return output_dir


def run_build_df(decoded_dir: str, df_output_dir: str):
    """Stage 2: Build DataFrames from decoded JSON."""
    print()
    print("=" * 70)
    print("  STAGE 2 — BUILD DATAFRAMES (JSON → pandas)")
    print("=" * 70)

    decoded_path = Path(decoded_dir)
    df_path = Path(df_output_dir)
    df_path.mkdir(parents=True, exist_ok=True)

    # Load driver info
    driver_list = load_json(decoded_path / 'DriverList.json')
    drivers = build_driver_map(driver_list)
    print(f"\n✓ Loaded {len(drivers)} drivers")

    # Build DataFrames
    print("\nBuilding DataFrames...")

    laps = build_laps_df(decoded_path, drivers)
    print(f"✓ Laps:      {len(laps)} rows, {len(laps['Driver'].unique()) if len(laps) > 0 else 0} drivers")

    telemetry = build_telemetry_df(decoded_path, drivers)
    print(f"✓ Telemetry: {len(telemetry)} rows, {len(telemetry['Driver'].unique()) if len(telemetry) > 0 else 0} drivers")

    positions = build_position_df(decoded_path, drivers)
    print(f"✓ Positions: {len(positions)} rows, {len(positions['Driver'].unique()) if len(positions) > 0 else 0} drivers")

    # Add distance
    print("\nComputing distance...")
    telemetry = add_distance_to_telemetry(telemetry)
    print("✓ Distance added")

    # Save
    laps.to_pickle(df_path / 'laps.pkl')
    telemetry.to_pickle(df_path / 'telemetry.pkl')
    positions.to_pickle(df_path / 'positions.pkl')

    print(f"\n✓ Saved to: {df_path}")
    print(f"  - laps.pkl      ({len(laps)} rows)")
    print(f"  - telemetry.pkl ({len(telemetry)} rows)")
    print(f"  - positions.pkl ({len(positions)} rows)")

    return laps, telemetry, positions


def run_full_pipeline(input_file: str, output_base: str = None):
    """Run the entire pipeline once."""
    input_path = Path(input_file)

    if output_base:
        base_dir = Path(output_base)
    else:
        base_dir = input_path.parent

    decoded_dir = str(base_dir / 'decoded')
    df_dir = str(base_dir / 'dataframes')

    start = time.time()
    print(f"\n🏎  F1 Telemetry Pipeline")
    print(f"   Input:  {input_path}")
    print(f"   Output: {base_dir}")
    print()

    # Stage 1: Decode
    run_decode(str(input_path), decoded_dir)

    # Stage 2: Build DataFrames
    laps, telemetry, positions = run_build_df(decoded_dir, df_dir)

    elapsed = time.time() - start

    # Summary
    print()
    print("=" * 70)
    print(f"  ✅ PIPELINE COMPLETE — {elapsed:.1f}s")
    print("=" * 70)
    print(f"  Laps:      {len(laps)} rows")
    print(f"  Telemetry: {len(telemetry)} rows")
    print(f"  Positions: {len(positions)} rows")
    print(f"  Output:    {base_dir}")
    print()
    print(f"  Quick start:")
    print(f"    import pandas as pd")
    print(f"    laps = pd.read_pickle('{df_dir}/laps.pkl')")
    print(f"    telemetry = pd.read_pickle('{df_dir}/telemetry.pkl')")
    print("=" * 70)

    return laps, telemetry, positions


def watch_mode(input_file: str, output_base: str = None, interval: int = 30):
    """
    Watch mode: continuously re-process the file as it grows.
    Useful during a live session while recording.
    """
    input_path = Path(input_file)
    last_size = 0
    run_count = 0

    print(f"\n👁  WATCH MODE — Monitoring {input_path}")
    print(f"   Refresh interval: {interval}s")
    print(f"   Press Ctrl+C to stop\n")

    try:
        while True:
            # Check if file has grown
            if not input_path.exists():
                print(f"   ⏳ Waiting for file to appear: {input_path}")
                time.sleep(interval)
                continue

            current_size = input_path.stat().st_size

            if current_size == last_size and run_count > 0:
                now = datetime.now().strftime('%H:%M:%S')
                print(f"   [{now}] No new data (file size: {current_size / 1024:.0f} KB). Waiting...")
                time.sleep(interval)
                continue

            # File has grown — re-process
            run_count += 1
            now = datetime.now().strftime('%H:%M:%S')
            delta = current_size - last_size
            print(f"\n{'#' * 70}")
            print(f"  [{now}] Run #{run_count} — +{delta / 1024:.0f} KB new data")
            print(f"{'#' * 70}")

            try:
                run_full_pipeline(str(input_path), output_base)
            except Exception as e:
                print(f"\n   ⚠️  Pipeline error: {e}")
                print(f"   Will retry on next interval...")

            last_size = current_size
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n   🛑 Watch mode stopped after {run_count} runs.")


def main():
    parser = argparse.ArgumentParser(
        description='F1 Telemetry Pipeline — End-to-End',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-shot processing
  python run_pipeline.py 2026/Bahrein/test_day1_FULL.txt

  # Watch mode (during live session)
  python run_pipeline.py 2026/Bahrein/test_day1_FULL.txt --watch

  # Watch mode with 60s interval
  python run_pipeline.py 2026/Bahrein/test_day1_FULL.txt --watch --interval 60
        """
    )

    parser.add_argument('input', help='Path to raw capture .txt file')
    parser.add_argument('--output', '-o', help='Output base directory (default: same as input)')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch mode: re-process as file grows')
    parser.add_argument('--interval', '-i', type=int, default=30,
                        help='Watch mode refresh interval in seconds (default: 30)')

    args = parser.parse_args()

    if not Path(args.input).exists() and not args.watch:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    if args.watch:
        watch_mode(args.input, args.output, args.interval)
    else:
        run_full_pipeline(args.input, args.output)


if __name__ == '__main__':
    main()
