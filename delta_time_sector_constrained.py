import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter
from typing import Tuple, Optional
import fastf1
from fastf1.core import Lap
import warnings


def delta_time_sector_constrained(reference_lap: Lap, comparison_lap: Lap, debug: bool = False) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Calculate delta time with sector time constraints and speed-informed interpolation.
    
    Key principle: Delta MUST match sector time differences at sector boundaries,
    with smooth, speed-informed interpolation between sectors.
    
    This ensures:
    1. Sector deltas are exactly correct (ground truth)
    2. Micro-sector variations respect speed differences
    3. Smooth transitions between sectors
    """
    
    # Get telemetry and sector times
    try:
        ref_tel = reference_lap.get_telemetry()
        comp_tel = comparison_lap.get_telemetry()
        
        # Get sector times (ground truth)
        ref_s1 = reference_lap['Sector1Time'].total_seconds() if pd.notna(reference_lap['Sector1Time']) else None
        ref_s2 = reference_lap['Sector2Time'].total_seconds() if pd.notna(reference_lap['Sector2Time']) else None
        ref_s3 = reference_lap['Sector3Time'].total_seconds() if pd.notna(reference_lap['Sector3Time']) else None
        
        comp_s1 = comparison_lap['Sector1Time'].total_seconds() if pd.notna(comparison_lap['Sector1Time']) else None
        comp_s2 = comparison_lap['Sector2Time'].total_seconds() if pd.notna(comparison_lap['Sector2Time']) else None
        comp_s3 = comparison_lap['Sector3Time'].total_seconds() if pd.notna(comparison_lap['Sector3Time']) else None
        
    except Exception as e:
        raise ValueError(f"Could not load telemetry/sector data: {e}")
    
    # Calculate sector deltas (ground truth constraints)
    sector_deltas = {
        'S1': comp_s1 - ref_s1 if (comp_s1 and ref_s1) else None,
        'S2': comp_s2 - ref_s2 if (comp_s2 and ref_s2) else None,
        'S3': comp_s3 - ref_s3 if (comp_s3 and ref_s3) else None
    }
    
    # Synchronize telemetry
    ref_sync, comp_sync = _synchronize_telemetry_basic(ref_tel, comp_tel, debug)
    
    # Find sector boundaries in telemetry
    sector_boundaries = _find_sector_boundaries(ref_sync, reference_lap, debug)
    
    # Calculate constrained delta
    delta_series = _calculate_constrained_delta(
        ref_sync, comp_sync, sector_deltas, sector_boundaries, debug
    )
    
    return delta_series, ref_sync, comp_sync


def _synchronize_telemetry_basic(ref_tel: pd.DataFrame, comp_tel: pd.DataFrame, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Basic telemetry synchronization to common distance grid."""
    
    # Clean data
    ref_tel = ref_tel.dropna(subset=['Distance', 'Time', 'Speed']).sort_values('Distance').reset_index(drop=True)
    comp_tel = comp_tel.dropna(subset=['Distance', 'Time', 'Speed']).sort_values('Distance').reset_index(drop=True)
    
    ref_tel = ref_tel.drop_duplicates(subset=['Distance'], keep='first')
    comp_tel = comp_tel.drop_duplicates(subset=['Distance'], keep='first')
    
    # Common distance range
    min_dist = max(ref_tel['Distance'].min(), comp_tel['Distance'].min())
    max_dist = min(ref_tel['Distance'].max(), comp_tel['Distance'].max())
    
    # High-resolution grid (1m resolution)
    distance_grid = np.arange(min_dist, max_dist, 1.0)
    
    # Interpolate to common grid
    ref_sync = _interpolate_telemetry(ref_tel, distance_grid)
    comp_sync = _interpolate_telemetry(comp_tel, distance_grid)
    
    return ref_sync, comp_sync


def _interpolate_telemetry(telemetry: pd.DataFrame, distance_grid: np.ndarray) -> pd.DataFrame:
    """Interpolate telemetry to distance grid."""
    
    sync_tel = pd.DataFrame({'Distance': distance_grid})
    
    # Time interpolation
    time_seconds = telemetry['Time'].dt.total_seconds()
    sync_tel['Time'] = pd.to_timedelta(
        np.interp(distance_grid, telemetry['Distance'], time_seconds), unit='s'
    )
    
    # Speed and other channels
    for channel in ['Speed', 'Throttle', 'Brake', 'RPM']:
        if channel in telemetry.columns:
            sync_tel[channel] = np.interp(distance_grid, telemetry['Distance'], telemetry[channel])
    
    return sync_tel


def _find_sector_boundaries(ref_tel: pd.DataFrame, reference_lap: Lap, debug: bool = False) -> dict:
    """Find sector boundary positions in telemetry."""
    
    # Get sector times
    ref_s1 = reference_lap['Sector1Time'].total_seconds() if pd.notna(reference_lap['Sector1Time']) else None
    ref_s2 = reference_lap['Sector2Time'].total_seconds() if pd.notna(reference_lap['Sector2Time']) else None
    
    boundaries = {'S1_end': None, 'S2_end': None}
    
    if ref_s1:
        # Find S1 end: time closest to sector 1 time
        ref_time = ref_tel['Time'].dt.total_seconds()
        lap_start_time = ref_time.iloc[0]
        s1_end_time = lap_start_time + ref_s1
        
        s1_end_idx = np.argmin(np.abs(ref_time - s1_end_time))
        boundaries['S1_end'] = s1_end_idx
    
    if ref_s2 and ref_s1:
        # Find S2 end: time closest to sector 1 + sector 2 time
        s2_end_time = lap_start_time + ref_s1 + ref_s2
        s2_end_idx = np.argmin(np.abs(ref_time - s2_end_time))
        boundaries['S2_end'] = s2_end_idx
    
    return boundaries


def _calculate_constrained_delta(ref_tel: pd.DataFrame, comp_tel: pd.DataFrame, 
                               sector_deltas: dict, boundaries: dict, debug: bool = False) -> np.ndarray:
    """
    Calculate delta time with sector constraints and speed-informed interpolation.
    
    Algorithm:
    1. Set delta = 0 at lap start
    2. Set delta = sector_delta at each sector boundary (constraints)
    3. Interpolate between boundaries using speed-informed method
    4. Ensure smooth transitions
    """
    
    n_points = len(ref_tel)
    delta_series = np.zeros(n_points)
    
    # Get basic time delta for reference
    ref_time = ref_tel['Time'].dt.total_seconds().values
    comp_time = comp_tel['Time'].dt.total_seconds().values
    ref_time_norm = ref_time - ref_time[0]
    comp_time_norm = comp_time - comp_time[0]
    raw_delta = comp_time_norm - ref_time_norm
    
    # Define constraint points
    constraint_points = [(0, 0.0)]  # Start at 0
    
    if boundaries['S1_end'] is not None and sector_deltas['S1'] is not None:
        constraint_points.append((boundaries['S1_end'], sector_deltas['S1']))
    
    if boundaries['S2_end'] is not None and sector_deltas['S2'] is not None:
        s1_s2_cumulative = sector_deltas['S1'] + sector_deltas['S2']
        constraint_points.append((boundaries['S2_end'], s1_s2_cumulative))
    
    # Final point: total lap delta
    if sector_deltas['S3'] is not None:
        total_delta = sum(d for d in sector_deltas.values() if d is not None)
        constraint_points.append((n_points - 1, total_delta))
    
    # Interpolate between constraint points with speed influence
    for i in range(len(constraint_points) - 1):
        start_idx, start_delta = constraint_points[i]
        end_idx, end_delta = constraint_points[i + 1]
        
        if start_idx == end_idx:
            continue
        
        # Get speed data for this segment
        segment_ref_speed = ref_tel['Speed'].iloc[start_idx:end_idx + 1].values
        segment_comp_speed = comp_tel['Speed'].iloc[start_idx:end_idx + 1].values
        segment_raw_delta = raw_delta[start_idx:end_idx + 1]
        
        # Create speed-informed interpolation
        segment_delta = _speed_informed_interpolation(
            start_delta, end_delta, segment_ref_speed, segment_comp_speed, 
            segment_raw_delta, debug and i == 0  # Only debug first segment
        )
        
        delta_series[start_idx:end_idx + 1] = segment_delta
    
    # Apply smoothing while preserving constraints
    delta_smoothed = _constrained_smoothing(delta_series, constraint_points, debug)
    
    return delta_smoothed


def _speed_informed_interpolation(start_delta: float, end_delta: float, 
                                ref_speed: np.ndarray, comp_speed: np.ndarray,
                                raw_delta: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Interpolate between constraint points using speed information.
    
    The interpolation respects:
    1. Fixed start and end deltas (constraints)
    2. Speed differences (faster = gaining time)
    3. Smooth progression
    """
    
    n_points = len(ref_speed)
    if n_points <= 1:
        return np.array([start_delta])
    
    # Linear baseline interpolation
    linear_interp = np.linspace(start_delta, end_delta, n_points)
    
    # Speed-based modulation
    speed_diff = comp_speed - ref_speed  # Positive = comp faster
    
    # Normalize speed differences to create smooth modulation
    if np.std(speed_diff) > 0:
        speed_diff_norm = (speed_diff - np.mean(speed_diff)) / np.std(speed_diff)
        speed_modulation = speed_diff_norm * 0.02  # Small modulation (±0.02s max)
    else:
        speed_modulation = np.zeros_like(speed_diff)
    
    # Combine linear interpolation with speed modulation
    result = linear_interp + speed_modulation
    
    # Ensure constraints are preserved
    result[0] = start_delta
    result[-1] = end_delta
    
    return result


def _constrained_smoothing(delta: np.ndarray, constraint_points: list, debug: bool = False) -> np.ndarray:
    """
    Apply smoothing while preserving constraint points exactly.
    """
    
    try:
        # Apply Savitzky-Golay smoothing
        smoothed = savgol_filter(delta, window_length=21, polyorder=3)
        
        # Restore constraint points exactly
        for idx, value in constraint_points:
            if 0 <= idx < len(smoothed):
                smoothed[idx] = value
        
        # Smooth transitions around constraint points
        for idx, value in constraint_points:
            if 0 < idx < len(smoothed) - 1:
                # Gentle blending around constraint points
                for offset in [-2, -1, 1, 2]:
                    blend_idx = idx + offset
                    if 0 <= blend_idx < len(smoothed):
                        # Weighted average: 70% smoothed, 30% constraint influence
                        weight = 0.3 * (1 - abs(offset) / 3)  # Decreasing weight with distance
                        smoothed[blend_idx] = (1 - weight) * smoothed[blend_idx] + weight * value
        
        return smoothed
        
    except Exception:
        return delta


# Main function
def delta_time(reference_lap: Lap, comparison_lap: Lap) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Sector-constrained delta time calculation."""
    return delta_time_sector_constrained(reference_lap, comparison_lap, debug=False)
