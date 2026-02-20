import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import warnings

def interpolate_dataframe(df, target_points=None, time_based=True, time_column='Time', distance_column='Distance'):
    """
    Improved interpolation function for FastF1 telemetry data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to interpolate (FastF1 telemetry format)
    target_points : int, optional
        Number of target points. If None, uses adaptive sizing based on data density
    time_based : bool, default True
        If True, interpolates based on time column (FastF1 standard)
        If False, uses uniform index-based interpolation
    time_column : str, default 'Time'
        Name of the time column to use for time-based interpolation (FastF1 uses 'Time')
    distance_column : str, default 'Distance'
        Name of the distance column if available (fallback option)
    
    Returns:
    --------
    pd.DataFrame
        Interpolated DataFrame
    """
    # Input validation
    if df.empty:
        warnings.warn("Empty DataFrame provided to interpolate_dataframe")
        return df.copy()
    
    n = len(df)
    if n < 2:
        warnings.warn("DataFrame has fewer than 2 rows, returning original")
        return df.copy()
    
    # Adaptive target points if not specified
    if target_points is None:
        if time_based and time_column in df.columns:
            # FastF1 primary path: resample to ~15 Hz (typical raw data is ~3.7 Hz)
            time_span = _get_time_span(df[time_column])
            target_points = max(1000, min(8000, int(time_span * 15))) if time_span > 0 else n * 3
        elif distance_column in df.columns:
            track_length = df[distance_column].max() - df[distance_column].min()
            target_points = max(1000, min(8000, int(track_length * 2.5)))
        else:
            target_points = min(5000, n * 3)
    
    new_df = pd.DataFrame()
    
    # Identify constant columns (e.g., 'Source') — no need to interpolate
    constant_cols = {c for c in df.columns if df[c].nunique() <= 1}
    
    # Choose interpolation basis (FastF1 priority: Time > Distance > Index)
    interpolation_method = 'index'  # default fallback
    
    if time_based and time_column in df.columns:
        # Time-based interpolation (FastF1 standard)
        try:
            original_basis = _convert_time_to_numeric(df[time_column])
            
            # Ensure monotonic time (required for interpolation)
            if len(original_basis) > 1 and np.all(np.diff(original_basis) >= 0):
                time_min, time_max = original_basis[0], original_basis[-1]
                new_basis = np.linspace(time_min, time_max, target_points)
                interpolation_method = 'time'
            else:
                warnings.warn(f"Time column '{time_column}' is not monotonic or invalid, trying distance-based")
        except Exception as e:
            warnings.warn(f"Error processing time column '{time_column}': {e}. Trying distance-based.")
    
    if interpolation_method == 'index' and distance_column in df.columns:
        # Distance-based fallback
        try:
            original_basis = df[distance_column].values
            
            if len(original_basis) > 1 and np.all(np.diff(original_basis) >= 0):
                dist_min, dist_max = original_basis[0], original_basis[-1]
                new_basis = np.linspace(dist_min, dist_max, target_points)
                interpolation_method = 'distance'
            else:
                warnings.warn(f"Distance column '{distance_column}' is not monotonic, falling back to index-based")
        except Exception as e:
            warnings.warn(f"Error processing distance column '{distance_column}': {e}. Using index-based.")
    
    if interpolation_method == 'index':
        # Index-based interpolation (final fallback)
        original_basis = np.linspace(0, 1, n)
        new_basis = np.linspace(0, 1, target_points)

    for column in df.columns:
        # Skip constant columns — just fill with the single value
        if column in constant_cols:
            new_df[column] = [df[column].iloc[0]] * target_points
            continue
        
        original_data = df[column].to_numpy()
        dtype = original_data.dtype
        
        try:
            # Handle continuous numerical values
            if np.issubdtype(dtype, np.floating):
                # Check for NaN values and handle them
                if np.any(np.isnan(original_data)):
                    # Use linear interpolation for NaN handling, then PCHIP
                    mask = ~np.isnan(original_data)
                    if np.sum(mask) < 2:
                        # Not enough valid points, use nearest neighbor
                        new_data = _nearest_neighbor_interpolation(original_basis, original_data, new_basis)
                    else:
                        # Interpolate over NaN values first
                        clean_basis = original_basis[mask]
                        clean_data = original_data[mask]
                        interpolator = PchipInterpolator(clean_basis, clean_data, extrapolate=False)
                        new_data = interpolator(new_basis)
                else:
                    interpolator = PchipInterpolator(original_basis, original_data, extrapolate=False)
                    new_data = interpolator(new_basis)

            # Handle datetime64
            elif np.issubdtype(dtype, np.datetime64):
                time_int = original_data.astype('datetime64[ns]').astype('int64')
                interpolator = PchipInterpolator(original_basis, time_int, extrapolate=False)
                new_data = pd.to_datetime(interpolator(new_basis))

            # Handle timedelta64
            elif np.issubdtype(dtype, np.timedelta64):
                time_int = original_data.astype('timedelta64[ns]').astype('int64')
                interpolator = PchipInterpolator(original_basis, time_int, extrapolate=False)
                new_data = pd.to_timedelta(interpolator(new_basis))

            # Handle integer-like data (e.g., gears) – round PCHIP
            # For discrete-code columns (DRS), use nearest-neighbor instead
            elif np.issubdtype(dtype, np.integer):
                if column == 'DRS':
                    new_data = _nearest_neighbor_interpolation(original_basis, original_data, new_basis)
                else:
                    interpolator = PchipInterpolator(original_basis, original_data.astype(float), extrapolate=False)
                    new_data = np.round(interpolator(new_basis)).astype(original_data.dtype)

            # Handle booleans – use nearest (forward-fill style)
            elif np.issubdtype(dtype, np.bool_):
                new_data = _nearest_neighbor_interpolation(original_basis, original_data, new_basis)

            # Handle strings/objects – use nearest
            else:
                new_data = _nearest_neighbor_interpolation(original_basis, original_data, new_basis)

            new_df[column] = new_data
            
        except Exception as e:
            warnings.warn(f"Error interpolating column '{column}': {e}. Using nearest neighbor fallback.")
            new_data = _nearest_neighbor_interpolation(original_basis, original_data, new_basis)
            new_df[column] = new_data

    return new_df


def _nearest_neighbor_interpolation(original_basis, original_data, new_basis):
    """
    Helper function for nearest neighbor interpolation (vectorized).
    """
    # Find nearest indices
    indices = np.searchsorted(original_basis, new_basis, side='left')
    
    # Clip to valid range for comparison (need at least index 1 to look left)
    indices = np.clip(indices, 1, len(original_data) - 1)
    
    # Choose the closer of left or right neighbor (fully vectorized)
    left_dist = np.abs(new_basis - original_basis[indices - 1])
    right_dist = np.abs(new_basis - original_basis[indices])
    indices[left_dist < right_dist] -= 1
    
    return original_data[indices]


def _get_time_span(time_series):
    """
    Helper function to get time span in seconds from FastF1 time data.
    Handles FastF1's specific Timedelta format.
    """
    try:
        # Convert to numeric first using our robust converter
        numeric_times = _convert_time_to_numeric(time_series)
        if len(numeric_times) > 1:
            return float(numeric_times[-1] - numeric_times[0])
        else:
            return 0
    except Exception:
        return 0


def _convert_time_to_numeric(time_series):
    """
    Convert FastF1 time data to numeric seconds for interpolation.
    FastF1 always provides pd.Timedelta series; defensive fallbacks are kept
    for non-standard inputs.
    """
    try:
        # FastF1 standard: pandas Timedelta series (always the case in practice)
        if hasattr(time_series, 'dt') and hasattr(time_series.dt, 'total_seconds'):
            return time_series.dt.total_seconds().values
        # Defensive fallbacks for non-standard inputs
        if np.issubdtype(time_series.dtype, np.timedelta64):
            return time_series.astype('timedelta64[ns]').astype('int64') / 1e9
        # Assume already numeric
        return time_series.values
    except Exception as e:
        warnings.warn(f"Error converting time data: {e}. Falling back to index-based interpolation.")
        return np.arange(len(time_series), dtype=float)


# Backward compatibility function
def interpolate_dataframe_legacy(df):
    """
    Legacy version of interpolate_dataframe for backward compatibility.
    Uses the original fixed 5000 points, index-based approach.
    """
    return interpolate_dataframe(df, target_points=5000, time_based=False)