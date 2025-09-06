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
        # Use adaptive sizing based on available data
        if time_based and time_column in df.columns:
            # For FastF1 time-based data: aim for ~10-20 Hz sampling rate
            time_span = _get_time_span(df[time_column])
            if time_span > 0:
                target_points = max(1000, min(8000, int(time_span * 15)))  # 15 Hz target
            else:
                target_points = min(5000, max(1000, n * 3))
        elif distance_column in df.columns:
            # Distance-based fallback
            track_length = df[distance_column].max() - df[distance_column].min()
            target_points = max(1000, min(8000, int(track_length * 2.5)))
        else:
            target_points = min(5000, max(1000, n * 3))  # 3x original density
    
    new_df = pd.DataFrame()
    
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
                # Debug info
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
                # Debug info
            else:
                warnings.warn(f"Distance column '{distance_column}' is not monotonic, falling back to index-based")
        except Exception as e:
            warnings.warn(f"Error processing distance column '{distance_column}': {e}. Using index-based.")
    
    if interpolation_method == 'index':
        # Index-based interpolation (final fallback)
        original_basis = np.linspace(0, 1, n)
        new_basis = np.linspace(0, 1, target_points)

    for column in df.columns:
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
                        interpolator = PchipInterpolator(clean_basis, clean_data, extrapolate=True)
                        new_data = interpolator(new_basis)
                else:
                    interpolator = PchipInterpolator(original_basis, original_data, extrapolate=True)
                    new_data = interpolator(new_basis)

            # Handle datetime64
            elif np.issubdtype(dtype, np.datetime64):
                time_int = original_data.astype('datetime64[ns]').astype('int64')
                interpolator = PchipInterpolator(original_basis, time_int, extrapolate=True)
                new_data = pd.to_datetime(interpolator(new_basis))

            # Handle timedelta64
            elif np.issubdtype(dtype, np.timedelta64):
                time_int = original_data.astype('timedelta64[ns]').astype('int64')
                interpolator = PchipInterpolator(original_basis, time_int, extrapolate=True)
                new_data = pd.to_timedelta(interpolator(new_basis))

            # Handle integer-like data (e.g., gears) – round PCHIP
            elif np.issubdtype(dtype, np.integer):
                interpolator = PchipInterpolator(original_basis, original_data.astype(float), extrapolate=True)
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
    Helper function for nearest neighbor interpolation.
    """
    # Find nearest indices
    indices = np.searchsorted(original_basis, new_basis, side='left')
    
    # Handle boundary conditions
    indices = np.clip(indices, 0, len(original_data) - 1)
    
    # For points exactly between two samples, choose the closer one
    for i in range(len(indices)):
        if indices[i] > 0 and indices[i] < len(original_data):
            left_dist = abs(new_basis[i] - original_basis[indices[i] - 1])
            right_dist = abs(new_basis[i] - original_basis[indices[i]])
            if left_dist < right_dist:
                indices[i] -= 1
    
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
    Helper function to convert FastF1 time data to numeric values for interpolation.
    Handles the specific FastF1 Timedelta format: "0 days 00:00:00.150000"
    """
    try:
        # Check if it's pandas Timedelta (FastF1 standard format)
        if hasattr(time_series, 'dt') and hasattr(time_series.dt, 'total_seconds'):
            # pandas Timedelta series - use .dt accessor
            return time_series.dt.total_seconds().values
        elif hasattr(time_series.iloc[0], 'total_seconds'):
            # individual timedelta objects - convert to seconds
            return np.array([t.total_seconds() for t in time_series])
        elif str(time_series.dtype).startswith('timedelta'):
            # numpy timedelta64 or pandas timedelta - convert to seconds
            return pd.to_timedelta(time_series).dt.total_seconds().values
        elif np.issubdtype(time_series.dtype, np.timedelta64):
            # numpy timedelta64 - convert to seconds
            return time_series.astype('timedelta64[ns]').astype('int64') / 1e9
        else:
            # assume already numeric
            return time_series.values
    except Exception as e:
        warnings.warn(f"Error converting time data: {e}. Falling back to index-based interpolation.")
        # fallback to index-based
        return np.arange(len(time_series), dtype=float)


# Backward compatibility function
def interpolate_dataframe_legacy(df):
    """
    Legacy version of interpolate_dataframe for backward compatibility.
    Uses the original fixed 5000 points, index-based approach.
    """
    return interpolate_dataframe(df, target_points=5000, time_based=False)