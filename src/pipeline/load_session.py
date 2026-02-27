#!/usr/bin/env python3
"""
Load Pipeline Data as FastF1 Session
=====================================
Constructs a FastF1-compatible Session object from the pipeline's
DataFrames (laps.pkl, telemetry.pkl, positions.pkl).

This allows using FastF1's analysis API (pick_fastest, get_telemetry,
get_car_data, get_pos_data, etc.) on live-captured pipeline data.

FastF1 Compatibility
--------------------
Tested with: FastF1 >=3.4, <=3.8
This module accesses the following FastF1 private attributes:
  - core.Session.__new__()      — bypass normal __init__
  - session._laps               — Laps DataFrame
  - session._car_data           — dict[driver_number] -> Telemetry
  - session._pos_data           — dict[driver_number] -> Telemetry
  - session._t0_date            — session start timestamp
  - session._loaded             — set of loaded data categories
  - session._weather_data       — weather DataFrame
  - session._circuit_info       — CircuitInfo object
  - session._results            — SessionResults
  - session._session_info       — dict from SessionInfo.json
  - session._session_status, _race_control_messages

If FastF1 breaks after an upgrade, check these attributes first.

Usage:
    from pipeline.load_session import load_session

    session = load_session("2026/day_2/dataframes")

    # Now use FastF1 API as normal
    lec = session.laps.pick_driver("LEC")
    fastest = lec.pick_fastest()
    tel = fastest.get_car_data()
    tel['Speed'].plot()
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

import fastf1.core as core
from fastf1.mvapi import get_circuit_info


def _parse_laptime_to_td(val):
    """Convert lap time string like '1:34.442' to pd.Timedelta."""
    if pd.isna(val) or val is None or val == '':
        return pd.NaT
    val = str(val)
    try:
        if ':' in val:
            parts = val.split(':')
            return pd.Timedelta(minutes=int(parts[0]), seconds=float(parts[1]))
        return pd.Timedelta(seconds=float(val))
    except (ValueError, TypeError):
        return pd.NaT


def _parse_sector_to_td(val):
    """Convert sector time string like '29.743' to pd.Timedelta."""
    if pd.isna(val) or val is None or val == '':
        return pd.NaT
    try:
        return pd.Timedelta(seconds=float(val))
    except (ValueError, TypeError):
        return pd.NaT


class _Event(pd.Series):
    """Minimal Event that behaves like FastF1's Event (has .year attribute)."""
    @property
    def year(self):
        ed = self.get('EventDate')
        if hasattr(ed, 'year'):
            return ed.year
        return self.get('_year', 2026)


def load_session(dataframes_dir, session_name="Testing", year=2026,
                 event_name="Pre-Season Test"):
    """Load pipeline DataFrames and construct a FastF1 Session object.

    Args:
        dataframes_dir: Path to directory containing laps.pkl, telemetry.pkl,
                        positions.pkl
        session_name: Session name (e.g. "Testing", "Race", "Qualifying")
        year: Season year
        event_name: Event name (e.g. "Pre-Season Test", "Bahrain Grand Prix")

    Returns:
        fastf1.core.Session with laps, car_data, and pos_data populated
    """
    dataframes_dir = Path(dataframes_dir)
    decoded_dir = dataframes_dir.parent / 'decoded'

    # Load raw pipeline DataFrames
    raw_laps = pd.read_pickle(dataframes_dir / 'laps.pkl')
    raw_tele = pd.read_pickle(dataframes_dir / 'telemetry.pkl')
    raw_pos = pd.read_pickle(dataframes_dir / 'positions.pkl')

    # ── Load session metadata from decoded directory (once) ─────────────
    si_loaded = None
    circuit_key = 63  # Bahrain default
    if (decoded_dir / 'SessionInfo.json').exists():
        with open(decoded_dir / 'SessionInfo.json') as f:
            si_loaded = json.load(f)
        if si_loaded:
            info = si_loaded[0].get('data', si_loaded[0])
            meeting = info.get('Meeting', {})
            event_name = meeting.get('OfficialName', event_name)
            circuit_key = meeting.get('Circuit', {}).get('Key', circuit_key)

    # ── Create a minimal Event object ──────────────────────────────────
    event = _Event({
        'RoundNumber': 0,
        'Country': 'Bahrain',
        'Location': 'Sakhir',
        'OfficialEventName': event_name,
        'EventDate': pd.Timestamp(year=year, month=2, day=1),
        'EventName': event_name,
        'EventFormat': 'testing',
        '_year': year,
    })

    # ── Create Session ─────────────────────────────────────────────────
    session = core.Session.__new__(core.Session)
    session.event = event
    session.name = session_name
    session.f1_api_support = False
    session._loaded = set()
    session._t0_date = None
    session._session_status = None
    session._race_control_messages = None
    session._session_info = {}
    if si_loaded:
        session._session_info = si_loaded[0].get('data', si_loaded[0])
    session._results = None

    # ── Load weather data ──────────────────────────────────────────────
    if (decoded_dir / 'WeatherData.json').exists():
        with open(decoded_dir / 'WeatherData.json') as f:
            weather_raw = json.load(f)
        weather_rows = []
        for entry in weather_raw:
            row = entry.get('data', entry)
            ts = entry.get('timestamp', '')
            if ts:
                row['Time'] = pd.Timestamp(ts)
            weather_rows.append(row)
        weather_df = pd.DataFrame(weather_rows)
        for col in ['AirTemp', 'Humidity', 'Pressure', 'Rainfall',
                    'TrackTemp', 'WindDirection', 'WindSpeed']:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
        session._weather_data = weather_df
    else:
        session._weather_data = pd.DataFrame(columns=[
            'AirTemp', 'Humidity', 'Pressure', 'Rainfall',
            'TrackTemp', 'WindDirection', 'WindSpeed'
        ])

    # ── Load circuit info (marker distances added after laps are built) ─
    try:
        _ci = get_circuit_info(year=year, circuit_key=circuit_key)
        session._circuit_info = _ci
    except Exception:
        session._circuit_info = None

    # Override get_circuit_info() to return the pre-loaded one
    import types
    session.get_circuit_info = types.MethodType(
        lambda self_: self_._circuit_info, session
    )

    # ── Determine t0_date (session start) ──────────────────────────────
    first_date = raw_tele['Date'].min()
    if pd.isna(first_date):
        first_date = raw_pos['Date'].min()
    session._t0_date = first_date

    # ── Build Laps DataFrame ──────────────────────────────────────────
    laps_data = raw_laps.copy()

    # Convert time columns to timedelta
    laps_data['LapTime'] = laps_data['LapTime'].apply(_parse_laptime_to_td)
    for col in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        laps_data[col] = laps_data[col].apply(_parse_sector_to_td)

    # Convert speeds to float
    for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
        laps_data[col] = pd.to_numeric(laps_data[col], errors='coerce')

    # LapNumber as float (FastF1 convention)
    laps_data['LapNumber'] = laps_data['LapNumber'].astype(float)

    # DriverNumber as string
    laps_data['DriverNumber'] = laps_data['DriverNumber'].astype(str)

    # Compute LapEndDate from LapEndTime string (timezone-naive for FastF1)
    laps_data['LapEndDate'] = pd.to_datetime(
        laps_data['LapEndTime'], utc=True, errors='coerce'
    ).dt.tz_localize(None)

    # Time = LapEndDate - t0_date (session-relative timedelta)
    t0_naive = session._t0_date.tz_localize(None) if session._t0_date.tzinfo else session._t0_date
    laps_data['Time'] = laps_data['LapEndDate'] - t0_naive

    # LapStartTime = Time - LapTime
    laps_data['LapStartTime'] = laps_data['Time'] - laps_data['LapTime']

    # LapStartDate = LapEndDate - LapTime
    laps_data['LapStartDate'] = laps_data['LapEndDate'] - laps_data['LapTime']

    # Sector session times (approximate)
    laps_data['Sector3SessionTime'] = laps_data['Time']
    laps_data['Sector2SessionTime'] = (
        laps_data['Time'] - laps_data['Sector3Time']
    )
    laps_data['Sector1SessionTime'] = (
        laps_data['Sector2SessionTime'] - laps_data['Sector2Time']
    )

    # Fill missing columns with defaults
    defaults = {
        'Stint': 1.0,
        'PitOutTime': pd.NaT,
        'PitInTime': pd.NaT,
        'Compound': 'UNKNOWN',
        'TyreLife': np.nan,
        'FreshTyre': False,
        'TrackStatus': '1',
        'Position': np.nan,
        'Deleted': None,
        'DeletedReason': '',
        'FastF1Generated': False,
        'IsAccurate': True,
    }
    for col, default in defaults.items():
        if col not in laps_data.columns:
            laps_data[col] = default

    # Create Laps object
    laps_obj = core.Laps(laps_data, _force_default_cols=True)
    laps_obj.session = session
    session._laps = laps_obj

    # ── Add marker distances to circuit info (now that laps exist) ─────
    if session._circuit_info is not None:
        try:
            session._circuit_info.add_marker_distance(
                reference_lap=laps_obj.pick_fastest()
            )
        except Exception:
            pass  # corners still have X/Y but no Distance — that's OK

    # ── Build car_data (telemetry per driver) ──────────────────────────
    drivers = sorted(raw_tele['DriverNumber'].unique().astype(str))
    session._car_data = {}
    session._pos_data = {}

    for drv in drivers:
        # Car telemetry
        drv_tele = raw_tele[raw_tele['DriverNumber'].astype(str) == drv].copy()
        drv_tele = drv_tele.sort_values('Date').reset_index(drop=True)

        # Ensure Date is timezone-aware
        if drv_tele['Date'].dt.tz is None:
            drv_tele['Date'] = drv_tele['Date'].dt.tz_localize('UTC')

        drv_tele['Time'] = drv_tele['Date'] - session._t0_date
        drv_tele['SessionTime'] = drv_tele['Time']

        # DRS column (not in our data, default 0)
        if 'DRS' not in drv_tele.columns:
            drv_tele['DRS'] = 0
        if 'Source' not in drv_tele.columns:
            drv_tele['Source'] = 'pipeline'

        car_tel = core.Telemetry(
            drv_tele[['Date', 'Time', 'SessionTime',
                       'Speed', 'RPM', 'Throttle', 'Brake',
                       'nGear', 'DRS', 'Source']],
            session=session,
            driver=drv,
        )
        session._car_data[drv] = car_tel

        # Position data
        drv_pos = raw_pos[raw_pos['DriverNumber'].astype(str) == drv].copy()
        drv_pos = drv_pos.sort_values('Date').reset_index(drop=True)

        if drv_pos['Date'].dt.tz is None:
            drv_pos['Date'] = drv_pos['Date'].dt.tz_localize('UTC')

        drv_pos['Time'] = drv_pos['Date'] - session._t0_date
        drv_pos['SessionTime'] = drv_pos['Time']
        if 'Source' not in drv_pos.columns:
            drv_pos['Source'] = 'pipeline'

        pos_tel = core.Telemetry(
            drv_pos[['Date', 'Time', 'SessionTime',
                      'X', 'Y', 'Z', 'Status', 'Source']],
            session=session,
            driver=drv,
        )
        session._pos_data[drv] = pos_tel

    # ── Build minimal results (driver list) ────────────────────────────
    from fastf1.core import SessionResults
    results_data = raw_laps[['DriverNumber', 'Driver', 'Team']].drop_duplicates()
    results_data = results_data.rename(columns={
        'Driver': 'Abbreviation',
        'Team': 'TeamName'
    })
    results_data['DriverNumber'] = results_data['DriverNumber'].astype(str)
    results_data = results_data.set_index(results_data['DriverNumber'])

    try:
        session._results = SessionResults(results_data,
                                          _force_default_cols=True)
    except Exception:
        session._results = results_data

    # Mark as loaded
    session._loaded = {'laps', 'telemetry'}

    # Provide drivers property
    session._drivers = drivers

    print(f"✅ FastF1 Session loaded from pipeline data")
    print(f"   {len(laps_obj)} laps, {len(drivers)} drivers")
    print(f"   Car data: {sum(len(v) for v in session._car_data.values()):,} points")
    print(f"   Pos data: {sum(len(v) for v in session._pos_data.values()):,} points")

    return session


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '2026/day_2/dataframes'
    session = load_session(path)

    # Quick test
    print("\n--- Quick test ---")
    fastest = session.laps.pick_fastest()
    print(f"Fastest lap: {fastest['Driver']} — {fastest['LapTime']}")

    tel = fastest.get_car_data()
    print(f"Telemetry: {len(tel)} points")
    print(f"  Max speed: {tel['Speed'].max()} km/h")
