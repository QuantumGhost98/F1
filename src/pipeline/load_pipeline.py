#!/usr/bin/env python3
"""
Load Pipeline Data — One-Shot Convenience Loader
==================================================
Goes from raw .txt capture (or pre-processed directory) to an f1analytics-ready
FastF1 Session in a single call.

Usage:
    from pipeline.load_pipeline import load_pipeline

    # From raw capture file (runs decode + build if needed)
    session = load_pipeline("2026/Bahrein/test_day2.txt")

    # From a directory that already has dataframes/
    session = load_pipeline("2026/Bahrein/")

    # Then use f1analytics directly:
    from f1analytics.telemetry import Telemetry
    t = Telemetry("Pre-Season Test", 2026, "Day 2", session=session)
    t.compare_laps({'LEC': 'fastest', 'VER': 'fastest'})
"""

from pathlib import Path

from pipeline.log import logger
from pipeline.decode.decode_telemetry import process_file
from pipeline.build_df.build_dataframes import (
    load_json, build_driver_map, build_laps_df,
    build_telemetry_df, build_position_df, add_distance_to_telemetry
)
from pipeline.load_session import load_session


def load_pipeline(path, session_name="Testing", year=2026,
                  event_name="Pre-Season Test", force_rebuild=False):
    """Load pipeline data and return a FastF1-compatible Session.

    Accepts either:
    - A raw .txt capture file → runs decode + build_df first
    - A directory containing decoded/ and/or dataframes/

    Args:
        path: Path to .txt file or session directory
        session_name: Session name for the FastF1 Session object
        year: Season year
        event_name: Event name (e.g. "Pre-Season Test", "Bahrain Grand Prix")
        force_rebuild: If True, re-run decode + build even if outputs exist

    Returns:
        fastf1.core.Session with laps, car_data, pos_data, weather, circuit_info
    """
    path = Path(path)

    # Determine base directory and whether we need to process
    if path.is_file() and path.suffix == '.txt':
        base_dir = path.parent
        input_file = path
    elif path.is_dir():
        base_dir = path
        # Look for a .txt file in the directory
        txt_files = list(path.glob('*.txt'))
        input_file = txt_files[0] if txt_files else None
    else:
        raise FileNotFoundError(f"Path not found or not supported: {path}")

    decoded_dir = base_dir / 'decoded'
    df_dir = base_dir / 'dataframes'

    # Run pipeline stages if needed
    needs_decode = force_rebuild or not decoded_dir.exists()
    needs_build = force_rebuild or not df_dir.exists()

    if needs_decode and input_file:
        logger.info("Stage 1: Decoding %s ...", input_file.name)
        process_file(str(input_file), str(decoded_dir))

    if needs_build and decoded_dir.exists():
        logger.info("Stage 2: Building DataFrames ...")
        df_dir.mkdir(parents=True, exist_ok=True)

        driver_list = load_json(decoded_dir / 'DriverList.json')
        drivers = build_driver_map(driver_list)

        laps = build_laps_df(decoded_dir, drivers)
        telemetry = build_telemetry_df(decoded_dir, drivers)
        positions = build_position_df(decoded_dir, drivers)
        telemetry = add_distance_to_telemetry(telemetry)

        laps.to_pickle(df_dir / 'laps.pkl')
        telemetry.to_pickle(df_dir / 'telemetry.pkl')
        positions.to_pickle(df_dir / 'positions.pkl')
        logger.info("Saved %d laps, %d telemetry, %d position rows",
                     len(laps), len(telemetry), len(positions))

    # Load as FastF1 Session
    if not df_dir.exists():
        raise FileNotFoundError(
            f"No dataframes/ directory found at {df_dir}. "
            f"Provide a raw .txt capture file to process first."
        )

    session = load_session(
        str(df_dir),
        session_name=session_name,
        year=year,
        event_name=event_name,
    )

    return session


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Please provide a path (raw .txt file or session directory).")
        sys.exit(1)

    session = load_pipeline(sys.argv[1])

    # Quick test
    print("\n--- Quick test ---")
    fastest = session.laps.pick_fastest()
    print(f"Fastest lap: {fastest['Driver']} — {fastest['LapTime']}")

    car = fastest.get_car_data()
    print(f"Car data: {len(car)} points, columns: {list(car.columns)}")

    try:
        car_dist = car.add_distance()
        print(f"add_distance(): ✅ ({len(car_dist)} points, max dist: {car_dist['Distance'].max():.0f}m)")
    except Exception as e:
        print(f"add_distance(): ❌ {e}")

    try:
        tel = fastest.get_telemetry()
        print(f"get_telemetry(): ✅ ({len(tel)} points)")
    except Exception as e:
        print(f"get_telemetry(): ❌ {e}")
