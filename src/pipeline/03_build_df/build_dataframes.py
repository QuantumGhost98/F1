#!/usr/bin/env python3
"""
Build F1 DataFrames from Decoded Live Data
===========================================
Reads the decoded JSON files and builds pandas DataFrames:
- laps: one row per lap per driver
- telemetry: all telemetry data with distance
- positions: all position data

Usage:
    python build_dataframes.py <decoded_dir>
    
Example:
    python build_dataframes.py 2026/Bahrein/decoded/
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys


def parse_ts(utc_str):
    """Parse F1 timestamp handling any fractional second precision."""
    s = utc_str.rstrip('Z')
    if '.' in s:
        base, frac = s.split('.', 1)
        frac = frac[:6].ljust(6, '0')
        s = f"{base}.{frac}"
    return datetime.fromisoformat(s + '+00:00')


def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def build_driver_map(driver_list_data):
    """Build driver number -> info mapping."""
    drivers = {}
    dl = driver_list_data[0]['data']
    for num, info in dl.items():
        if isinstance(info, dict) and 'Tla' in info:
            drivers[num] = {
                'tla': info['Tla'],
                'name': info.get('FullName', ''),
                'team': info.get('TeamName', ''),
                'color': info.get('TeamColour', ''),
            }
    return drivers


def build_laps_df(decoded_dir, drivers):
    """Build the main laps DataFrame."""
    timing_data = load_json(Path(decoded_dir) / 'TimingData.json')
    app_data = load_json(Path(decoded_dir) / 'TimingAppData.json')
    
    # Get keyframe (first entry has full state)
    kf = timing_data[0]['data']
    kf_app = app_data[0]['data']
    
    laps_records = []
    
    # Track lap completions from TimingData updates
    lap_updates = {}   # driver_num -> {lap_num -> {time, sectors, speeds, ...}}
    current_lap = {}   # driver_num -> current NumberOfLaps (tracks lap in progress)
    pending = {}       # driver_num -> {sector/speed data for the lap being driven}
    
    for entry in timing_data:
        data = entry.get('data', {})
        ts = entry.get('timestamp', '')
        lines = data.get('Lines', {})
        
        for num, d in lines.items():
            if not isinstance(d, dict):
                continue
            
            if num not in lap_updates:
                lap_updates[num] = {}
            if num not in pending:
                pending[num] = {}
            
            # Update current lap number when F1 sends it
            nl = d.get('NumberOfLaps')
            if nl is not None:
                current_lap[num] = nl
            
            # Sectors — accumulate into pending BEFORE lap completion check
            # (Sector 3 arrives in the same message as LastLapTime/NumberOfLaps)
            sectors = d.get('Sectors', {})
            if isinstance(sectors, list):
                for i, sv in enumerate(sectors):
                    if isinstance(sv, dict) and sv.get('Value', '') != '':
                        pending[num][f'sector_{i+1}'] = sv['Value']
            elif isinstance(sectors, dict):
                for sk, sv in sectors.items():
                    if isinstance(sv, dict) and sv.get('Value', '') != '':
                        pending[num][f'sector_{int(sk)+1}'] = sv['Value']
            
            # Speeds — accumulate into pending BEFORE lap completion check
            speeds = d.get('Speeds', {})
            if isinstance(speeds, dict):
                for trap, sv in speeds.items():
                    if isinstance(sv, dict) and sv.get('Value', '') != '':
                        pending[num][f'speed_{trap}'] = sv['Value']
            
            # Lap completion — merge pending data into this lap
            lt = d.get('LastLapTime', {})
            if isinstance(lt, dict) and lt.get('Value') and nl is not None:
                if nl not in lap_updates[num]:
                    lap_updates[num][nl] = {}
                
                lap_updates[num][nl].update({
                    'lap_time': lt['Value'],
                    'timestamp': ts,
                    'personal_best': lt.get('PersonalFastest', False),
                    'overall_best': lt.get('OverallFastest', False),
                })
                
                # Merge all pending sector/speed data into this lap
                for k, v in pending[num].items():
                    if k not in lap_updates[num][nl]:
                        lap_updates[num][nl][k] = v
                
                # Clear pending for next lap
                pending[num] = {}
    
    # Final pass: merge any remaining pending data into the last known lap
    for num, pdata in pending.items():
        if pdata and num in current_lap:
            nl = current_lap[num]
            if nl in lap_updates.get(num, {}):
                for k, v in pdata.items():
                    if k not in lap_updates[num][nl]:
                        lap_updates[num][nl][k] = v
    
    # Build laps records
    for num, laps in lap_updates.items():
        driver_info = drivers.get(num, {'tla': num, 'name': '', 'team': ''})
        
        for lap_num, lap_data in sorted(laps.items()):
            if not lap_data.get('lap_time'):
                continue
            
            laps_records.append({
                'DriverNumber': num,
                'Driver': driver_info['tla'],
                'Team': driver_info['team'],
                'LapNumber': lap_num,
                'LapTime': lap_data.get('lap_time'),
                'LapEndTime': lap_data.get('timestamp'),
                'Sector1Time': lap_data.get('sector_1'),
                'Sector2Time': lap_data.get('sector_2'),
                'Sector3Time': lap_data.get('sector_3'),
                'SpeedI1': lap_data.get('speed_I1'),
                'SpeedI2': lap_data.get('speed_I2'),
                'SpeedFL': lap_data.get('speed_FL'),
                'SpeedST': lap_data.get('speed_ST'),
                'IsPersonalBest': lap_data.get('personal_best', False),
            })
    
    df = pd.DataFrame(laps_records)
    
    # Sort by driver and lap
    if not df.empty:
        df = df.sort_values(['DriverNumber', 'LapNumber']).reset_index(drop=True)
    
    return df


def build_telemetry_df(decoded_dir, drivers):
    """Build the telemetry DataFrame with all car data."""
    car_data = load_json(Path(decoded_dir) / 'CarData.json')
    
    records = []
    
    for msg in car_data:
        for entry in msg['data'].get('Entries', []):
            utc = entry.get('Utc', '')
            if not utc:
                continue
            ts = parse_ts(utc)
            
            for num, car in entry.get('Cars', {}).items():
                ch = car.get('Channels', {})
                records.append({
                    'Date': ts,
                    'DriverNumber': num,
                    'Driver': drivers.get(num, {}).get('tla', num),
                    'RPM': ch.get('0', 0),
                    'Speed': ch.get('2', 0),
                    'nGear': ch.get('3', 0),
                    'Throttle': ch.get('4', 0),
                    'Brake': ch.get('5', 0),
                })
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        df = df.sort_values(['DriverNumber', 'Date']).reset_index(drop=True)
        
        # Add SessionTime (relative to first timestamp)
        if len(df) > 0:
            first_ts = df['Date'].min()
            df['SessionTime'] = (df['Date'] - first_ts).dt.total_seconds()
    
    return df


def build_position_df(decoded_dir, drivers):
    """Build the position DataFrame."""
    pos_data = load_json(Path(decoded_dir) / 'Position.json')
    
    records = []
    
    for msg in pos_data:
        for entry in msg['data'].get('Position', []):
            utc = entry.get('Timestamp', '')
            if not utc:
                continue
            ts = parse_ts(utc)
            
            for num, pos in entry.get('Entries', {}).items():
                records.append({
                    'Date': ts,
                    'DriverNumber': num,
                    'Driver': drivers.get(num, {}).get('tla', num),
                    'X': pos.get('X', 0),
                    'Y': pos.get('Y', 0),
                    'Z': pos.get('Z', 0),
                    'Status': pos.get('Status', ''),
                })
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        df = df.sort_values(['DriverNumber', 'Date']).reset_index(drop=True)
    
    return df


def add_distance_to_telemetry(telemetry_df):
    """Add Distance column to telemetry DataFrame."""
    if telemetry_df.empty:
        telemetry_df['Distance'] = []
        return telemetry_df
    
    # Group by driver
    def compute_distance(group):
        group = group.sort_values('Date').copy()
        
        # Speed in m/s
        speeds_ms = group['Speed'].values / 3.6
        
        # Time deltas
        dates = group['Date'].values
        dt = np.zeros(len(group))
        for i in range(1, len(group)):
            dt[i] = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])).total_seconds()
        
        # Cumulative distance
        distance = np.cumsum(speeds_ms * dt)
        group['Distance'] = distance
        
        return group
    
    telemetry_df = telemetry_df.groupby('DriverNumber', group_keys=False).apply(compute_distance)
    
    return telemetry_df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Please provide decoded directory path.")
        sys.exit(1)
    
    decoded_dir = Path(sys.argv[1])
    
    if not decoded_dir.exists():
        print(f"Error: Directory not found: {decoded_dir}")
        sys.exit(1)
    
    print(f"Loading data from: {decoded_dir}")
    print()
    
    # Load driver info
    driver_list = load_json(decoded_dir / 'DriverList.json')
    drivers = build_driver_map(driver_list)
    print(f"✓ Loaded {len(drivers)} drivers")
    
    # Build DataFrames
    print("\nBuilding DataFrames...")
    
    laps = build_laps_df(decoded_dir, drivers)
    print(f"✓ Laps:      {len(laps)} rows, {len(laps['Driver'].unique())} drivers")
    
    telemetry = build_telemetry_df(decoded_dir, drivers)
    print(f"✓ Telemetry: {len(telemetry)} rows, {len(telemetry['Driver'].unique())} drivers")
    
    positions = build_position_df(decoded_dir, drivers)
    print(f"✓ Positions: {len(positions)} rows, {len(positions['Driver'].unique())} drivers")
    
    # Add distance
    print("\nComputing distance...")
    telemetry = add_distance_to_telemetry(telemetry)
    print(f"✓ Distance added to telemetry")
    
    # Save to pickle
    output_dir = decoded_dir.parent / 'dataframes'
    output_dir.mkdir(exist_ok=True)
    
    laps.to_pickle(output_dir / 'laps.pkl')
    telemetry.to_pickle(output_dir / 'telemetry.pkl')
    positions.to_pickle(output_dir / 'positions.pkl')
    
    print(f"\n✓ Saved DataFrames to: {output_dir}")
    print(f"  - laps.pkl")
    print(f"  - telemetry.pkl")
    print(f"  - positions.pkl")
    
    # Show sample
    print("\n" + "="*80)
    print("LAPS SAMPLE (first 10 rows)")
    print("="*80)
    print(laps.head(10).to_string())
    
    print("\n" + "="*80)
    print("TELEMETRY SAMPLE (LEC, first 10 rows)")
    print("="*80)
    lec_tel = telemetry[telemetry.Driver == 'LEC'].head(10)
    print(lec_tel.to_string())
    
    print("\n" + "="*80)
    print("Quick access example:")
    print("="*80)
    print("""
import pandas as pd

# Load DataFrames
laps = pd.read_pickle('2026/Bahrein/dataframes/laps.pkl')
telemetry = pd.read_pickle('2026/Bahrein/dataframes/telemetry.pkl')
positions = pd.read_pickle('2026/Bahrein/dataframes/positions.pkl')

# Get Leclerc's laps
lec_laps = laps[laps.Driver == 'LEC']

# Get telemetry for a specific time window (e.g., lap 26)
# Lap 26: 13:43:25 -> 13:45:05
lec_tel = telemetry[
    (telemetry.Driver == 'LEC') & 
    (telemetry.Date >= '2026-02-11T13:43:25') &
    (telemetry.Date <= '2026-02-11T13:45:05')
]

# Plot
lec_tel.plot(x='Distance', y='Speed')
""")


if __name__ == '__main__':
    main()
