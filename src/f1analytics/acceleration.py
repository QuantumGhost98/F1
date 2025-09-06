import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings


def compute_acceleration(df, method='gradient', smooth=True, smooth_params=None, clip_range=(-6, 6), output_units='g'):
    """
    Compute longitudinal acceleration from telemetry data with multiple calculation methods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry DataFrame with columns: 'Speed' [km/h], 'Distance' [m], 'Time' [timedelta]
        Expected to have ~1000 data points after interpolation
    method : str, default 'gradient'
        Acceleration calculation method:
        - 'gradient': dv/dx * v (spatial derivative method)
        - 'time_derivative': dv/dt (time derivative method)  
        - 'combined': weighted average of both methods
    smooth : bool, default True
        Whether to apply smoothing filter to reduce noise
    smooth_params : dict, optional
        Smoothing parameters: {'window_length': int, 'polyorder': int}
        If None, uses adaptive parameters based on data size
    clip_range : tuple, default (-0.8, 0.8)
        Range to clip acceleration values [g] for realistic F1 bounds
        F1 cars typically: +0.6g acceleration, -0.8g braking
    output_units : str, default 'g'
        Output units: 'g' for g-force, 'ms2' for m/s²
        
    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame with added 'Acceleration' column [g] (longitudinal)
    """
    if df.empty:
        warnings.warn("Empty DataFrame provided")
        return df.copy()
    
    if len(df) < 3:
        warnings.warn("DataFrame has fewer than 3 rows, cannot compute acceleration")
        result = df.copy()
        result['Acceleration'] = 0.0
        return result
    
    # Required columns check
    required_cols = ['Speed', 'Distance']
    if method in ['time_derivative', 'combined']:
        required_cols.append('Time')
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    result = df.copy()
    
    # Convert speed from km/h to m/s
    speed_ms = result['Speed'] / 3.6
    
    # Ensure distance is monotonic and interpolated
    result['Distance'] = result['Distance'].interpolate()
    
    if method == 'gradient':
        # Spatial derivative method: dv/dx * v
        acceleration = _compute_spatial_acceleration(speed_ms, result['Distance'])
        
    elif method == 'time_derivative':
        # Time derivative method: dv/dt
        acceleration = _compute_time_acceleration(speed_ms, result['Time'])
        
    elif method == 'combined':
        # Combined method: weighted average
        acc_spatial = _compute_spatial_acceleration(speed_ms, result['Distance'])
        acc_time = _compute_time_acceleration(speed_ms, result['Time'])
        
        # Weight based on data quality (prefer method with less noise)
        weight_spatial = 0.7  # Spatial method typically more stable
        weight_time = 0.3
        
        acceleration = weight_spatial * acc_spatial + weight_time * acc_time
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gradient', 'time_derivative', or 'combined'")
    
    # Apply smoothing if requested
    if smooth:
        acceleration = _apply_smoothing(acceleration, smooth_params)
    
    # Convert to requested units
    if output_units == 'g':
        # Convert from m/s² to g-force (divide by 9.81 m/s²)
        acceleration = acceleration / 9.81
    elif output_units != 'ms2':
        warnings.warn(f"Unknown output_units '{output_units}', using m/s²")
    
    # Clip to realistic range
    if clip_range is not None:
        acceleration = np.clip(acceleration, clip_range[0], clip_range[1])
    
    result['Acceleration'] = acceleration
    return result


def compute_total_acceleration(df, smooth=True, smooth_params=None, clip_range=(0, 6.0), output_units='g'):
    """
    Compute total acceleration magnitude from telemetry data.
    
    This computes the vector magnitude: |a_total| = √(a_longitudinal² + a_lateral²)
    This represents the actual g-force magnitude experienced by the driver.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry DataFrame with columns: 'Speed' [km/h], 'Distance' [m], 'Time' [timedelta]
    smooth : bool, default True
        Whether to apply smoothing filter to reduce noise
    smooth_params : dict, optional
        Smoothing parameters: {'window_length': int, 'polyorder': int}
    clip_range : tuple, default (0, 6.0)
        Range to clip total acceleration [g] for realistic F1 bounds
    output_units : str, default 'g'
        Output units: 'g' for g-force, 'ms2' for m/s²
        
    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame with added 'Total_Acceleration' column [g]
    """
    if df.empty:
        warnings.warn("Empty DataFrame provided")
        return df.copy()
    
    if len(df) < 3:
        warnings.warn("DataFrame has fewer than 3 rows, cannot compute acceleration")
        result = df.copy()
        result['Total_Acceleration'] = 1.0  # 1g baseline (gravity)
        return result
    
    result = df.copy()
    speed_ms = result['Speed'] / 3.6
    
    # Step 1: Compute longitudinal acceleration (improved method)
    longitudinal_acc = _compute_total_longitudinal_acceleration(result, speed_ms)
    
    # Step 2: Compute lateral acceleration (simplified method)
    lateral_acc = _compute_total_lateral_acceleration(result, speed_ms)
    
    # Step 3: Compute total acceleration magnitude
    # |a_total| = √(a_long² + a_lat²)
    total_acc_ms2 = np.sqrt(longitudinal_acc**2 + lateral_acc**2)
    
    # Step 4: Apply smoothing if requested
    if smooth:
        total_acc_ms2 = _apply_total_smoothing(total_acc_ms2, smooth_params)
    
    # Step 5: Convert to requested units
    if output_units == 'g':
        total_acc = total_acc_ms2 / 9.81  # Convert to g-force
    else:
        total_acc = total_acc_ms2
    
    # Step 6: Apply realistic clipping
    if clip_range is not None:
        total_acc = np.clip(total_acc, clip_range[0], clip_range[1])
    
    result['Total_Acceleration'] = total_acc
    return result


def _compute_total_longitudinal_acceleration(df, speed_ms):
    """
    Compute longitudinal acceleration for total acceleration calculation.
    Uses time-based method with smoothing.
    """
    if 'Time' not in df.columns:
        # Fallback to spatial method if no time data
        distance = df['Distance'].interpolate().values if 'Distance' in df.columns else np.arange(len(speed_ms))
        return _compute_spatial_acceleration(speed_ms, distance)
    
    # Use time-based method (more accurate)
    time_series = df['Time']
    
    # Convert time to seconds
    if hasattr(time_series, 'dt') and hasattr(time_series.dt, 'total_seconds'):
        time_seconds = time_series.dt.total_seconds().values
    elif hasattr(time_series.iloc[0], 'total_seconds'):
        time_seconds = np.array([t.total_seconds() for t in time_series])
    else:
        time_seconds = time_series.values
    
    # Ensure monotonic time
    if len(time_seconds) > 1:
        dt = np.diff(time_seconds)
        if np.any(dt <= 0):
            time_seconds = np.linspace(time_seconds[0], time_seconds[-1], len(time_seconds))
    
    # Light smoothing to preserve braking peaks
    from scipy.signal import savgol_filter
    window_length = min(11, len(speed_ms))  # Reduced smoothing for sharper braking
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        speed_smooth = savgol_filter(speed_ms, window_length=window_length, polyorder=2)
    else:
        speed_smooth = speed_ms.copy()
    
    # Compute dv/dt with higher resolution
    longitudinal_acc = np.gradient(speed_smooth, time_seconds)
    
    # Apply realistic F1 constraints (preserve heavy braking)
    longitudinal_acc = np.clip(longitudinal_acc, -70.0, 15.0)  # -7g braking to +1.5g acceleration
    
    return longitudinal_acc


def _compute_total_lateral_acceleration(df, speed_ms):
    """
    Compute lateral acceleration for total acceleration calculation.
    Uses the existing lateral_acceleration module for better accuracy.
    """
    # Import and use the existing lateral acceleration computation
    from f1analytics.lateral_acceleration import compute_lateral_acceleration
    
    # Create a temporary dataframe with the required columns
    temp_df = df.copy()
    temp_df['Speed'] = speed_ms * 3.6  # Convert back to km/h for the function
    
    # Use the existing lateral acceleration function with centripetal method
    result_df = compute_lateral_acceleration(temp_df, method='centripetal', smooth=True)
    
    # Extract lateral acceleration and convert to m/s²
    if 'Lateral_Acceleration' in result_df.columns:
        lateral_acc_g = result_df['Lateral_Acceleration'].values
        lateral_acc_ms2 = lateral_acc_g * 9.81  # Convert from g to m/s²
    else:
        # Fallback: simple speed-based estimation
        lateral_acc_ms2 = np.abs(speed_ms) * 0.1  # Very basic fallback
    
    return lateral_acc_ms2


def _apply_total_smoothing(total_acc, smooth_params=None):
    """
    Apply smoothing to total acceleration.
    """
    if smooth_params is None:
        data_length = len(total_acc)
        if data_length >= 1000:
            window_length = min(31, data_length)
            polyorder = 3
        elif data_length >= 500:
            window_length = min(21, data_length)
            polyorder = 2
        else:
            window_length = min(15, data_length)
            polyorder = 2
            
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 1)
    else:
        window_length = smooth_params.get('window_length', 31)
        polyorder = smooth_params.get('polyorder', 3)
        window_length = min(window_length, len(total_acc))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 1)
    
    try:
        if window_length >= 3:
            from scipy.signal import savgol_filter
            smoothed = savgol_filter(total_acc, window_length=window_length, polyorder=polyorder)
            return smoothed
        else:
            return total_acc
    except Exception as e:
        warnings.warn(f"Total acceleration smoothing failed: {e}")
        return total_acc


def _compute_spatial_acceleration(speed_ms, distance):
    """
    Compute acceleration using improved spatial derivative method.
    Uses smoothed speed data and better gradient calculation to avoid artifacts.
    """
    # Step 1: Smooth speed data to reduce noise-induced artifacts
    from scipy.signal import savgol_filter
    
    # Apply light smoothing to speed before gradient calculation
    window_length = min(21, len(speed_ms))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        speed_smooth = savgol_filter(speed_ms, window_length=window_length, polyorder=2)
    else:
        speed_smooth = speed_ms.copy()
    
    # Step 2: Ensure distance is properly monotonic
    distance_smooth = np.copy(distance)
    
    # Step 3: Compute dv/dx using smoothed data
    dv_dx = np.gradient(speed_smooth, distance_smooth)
    
    # Step 4: Apply the chain rule: a = dv/dt = dv/dx * dx/dt = dv/dx * v
    # But use original speed for the multiplication to preserve magnitude
    acceleration = dv_dx * speed_ms
    
    # Step 5: Apply physical constraints
    # In corners (low speed + high dv/dx), limit positive acceleration
    speed_kmh = speed_ms * 3.6
    
    # Detect potential corner sections (low speed + high speed gradient)
    abs_dv_dx = np.abs(dv_dx)
    corner_threshold = np.percentile(abs_dv_dx, 75)  # Top 25% of speed changes
    
    corner_mask = (speed_kmh < 150) & (abs_dv_dx > corner_threshold)
    
    # In corners, cap positive acceleration to realistic values
    # F1 cars can't accelerate hard while cornering at low speeds
    acceleration[corner_mask & (acceleration > 2.0)] = 2.0  # Cap at 2 m/s² in corners
    
    return acceleration


def _compute_time_acceleration(speed_ms, time_series):
    """
    Compute acceleration using improved time derivative: dv/dt
    Handles FastF1 timedelta format with better noise handling.
    """
    # Convert time to seconds
    if hasattr(time_series, 'dt') and hasattr(time_series.dt, 'total_seconds'):
        time_seconds = time_series.dt.total_seconds().values
    elif hasattr(time_series.iloc[0], 'total_seconds'):
        time_seconds = np.array([t.total_seconds() for t in time_series])
    else:
        # Assume already numeric
        time_seconds = time_series.values
    
    # Step 1: Ensure time is monotonic
    if len(time_seconds) > 1:
        dt = np.diff(time_seconds)
        if np.any(dt <= 0):
            # Fix non-monotonic time
            time_seconds = np.linspace(time_seconds[0], time_seconds[-1], len(time_seconds))
    
    # Step 2: Apply light smoothing to speed for gradient calculation
    from scipy.signal import savgol_filter
    
    window_length = min(15, len(speed_ms))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        speed_smooth = savgol_filter(speed_ms, window_length=window_length, polyorder=2)
    else:
        speed_smooth = speed_ms.copy()
    
    # Step 3: Compute dv/dt using smoothed data
    acceleration = np.gradient(speed_smooth, time_seconds)
    
    # Step 4: Apply physical realism constraints
    # F1 cars have physical limits for acceleration/deceleration
    
    # Extreme braking: up to -6g in emergency situations
    # Normal braking: -3g to -5g
    # Acceleration: typically +0.3g to +0.8g (limited by traction and power)
    
    # Cap unrealistic values
    acceleration = np.clip(acceleration, -60.0, 8.0)  # -6g to +0.8g approximately
    
    return acceleration


def _apply_smoothing(acceleration, smooth_params=None):
    """
    Apply Savitzky-Golay smoothing to acceleration data.
    """
    if smooth_params is None:
        # Adaptive smoothing parameters based on data size
        data_length = len(acceleration)
        
        if data_length >= 1000:
            # For ~1000 points, use moderate smoothing
            window_length = min(31, data_length)
            polyorder = 3
        elif data_length >= 500:
            window_length = min(21, data_length)
            polyorder = 2
        else:
            window_length = min(11, data_length)
            polyorder = 2
            
        # Ensure window_length is odd and >= polyorder + 1
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 1)
        
    else:
        window_length = smooth_params.get('window_length', 31)
        polyorder = smooth_params.get('polyorder', 3)
        
        # Validate parameters
        window_length = min(window_length, len(acceleration))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 1)
    
    try:
        if window_length >= 3:
            smoothed = savgol_filter(acceleration, window_length=window_length, polyorder=polyorder)
            return smoothed
        else:
            warnings.warn("Window length too small for smoothing, returning original data")
            return acceleration
    except Exception as e:
        warnings.warn(f"Smoothing failed: {e}. Returning original data")
        return acceleration


def compute_acceleration_metrics(df_with_acc):
    """
    Compute useful acceleration metrics from telemetry data with acceleration.
    
    Parameters:
    -----------
    df_with_acc : pd.DataFrame
        DataFrame with 'Acceleration' [g], 'Speed', and 'Distance' columns
        
    Returns:
    --------
    dict
        Dictionary containing acceleration metrics (in g-force units)
    """
    if 'Acceleration' not in df_with_acc.columns:
        raise ValueError("DataFrame must contain 'Acceleration' column")
    
    acc = df_with_acc['Acceleration']  # Already in g-force
    speed = df_with_acc['Speed'] / 3.6  # Convert to m/s
    
    metrics = {
        'max_acceleration_g': acc.max(),
        'max_deceleration_g': acc.min(),
        'avg_acceleration_g': acc[acc > 0].mean() if any(acc > 0) else 0,
        'avg_deceleration_g': acc[acc < 0].mean() if any(acc < 0) else 0,
        'acceleration_std_g': acc.std(),
        'max_acc_at_speed': speed[acc.idxmax()] * 3.6 if not acc.empty else 0,  # km/h
        'max_dec_at_speed': speed[acc.idxmin()] * 3.6 if not acc.empty else 0,  # km/h
    }
    
    # Find acceleration zones (thresholds in g-force)
    acc_threshold = 0.05  # 0.05g ≈ 0.5 m/s²
    dec_threshold = -0.05  # -0.05g ≈ -0.5 m/s²
    
    accelerating = acc > acc_threshold
    braking = acc < dec_threshold
    
    metrics.update({
        'time_accelerating_pct': (accelerating.sum() / len(acc)) * 100,
        'time_braking_pct': (braking.sum() / len(acc)) * 100,
        'time_coasting_pct': ((~accelerating & ~braking).sum() / len(acc)) * 100,
    })
    
    return metrics


def compare_acceleration_profiles(df1, df2, driver1_name="Driver 1", driver2_name="Driver 2"):
    """
    Compare acceleration profiles between two drivers/laps.
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        DataFrames with 'Acceleration' and 'Distance' columns
    driver1_name, driver2_name : str
        Names for the comparison
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with distance-aligned acceleration comparison
    """
    if 'Acceleration' not in df1.columns or 'Acceleration' not in df2.columns:
        raise ValueError("Both DataFrames must contain 'Acceleration' column")
    
    # Align by distance (assuming both have similar distance ranges)
    min_dist = max(df1['Distance'].min(), df2['Distance'].min())
    max_dist = min(df1['Distance'].max(), df2['Distance'].max())
    
    # Create common distance grid
    distance_grid = np.linspace(min_dist, max_dist, 500)
    
    # Interpolate both acceleration profiles to common grid
    acc1_interp = np.interp(distance_grid, df1['Distance'], df1['Acceleration'])
    acc2_interp = np.interp(distance_grid, df2['Distance'], df2['Acceleration'])
    
    comparison_df = pd.DataFrame({
        'Distance': distance_grid,
        f'{driver1_name}_Acceleration': acc1_interp,
        f'{driver2_name}_Acceleration': acc2_interp,
        'Acceleration_Delta': acc1_interp - acc2_interp
    })
    
    return comparison_df


# Example usage and testing
if __name__ == "__main__":
    # Create sample telemetry data for testing
    print("Creating sample telemetry data for testing...")
    
    # Simulate ~1000 data points of telemetry
    n_points = 1000
    distance = np.linspace(0, 5000, n_points)  # 5km track
    
    # Simulate realistic F1 speed profile
    # Start slow, accelerate, maintain high speed, then brake for corners
    speed_profile = []
    for d in distance:
        if d < 1000:  # Acceleration zone
            speed = 80 + (d / 1000) * 240  # 80-320 km/h
        elif d < 3000:  # High speed section
            speed = 320 - 20 * np.sin((d - 1000) / 500)  # 300-340 km/h with variation
        elif d < 4000:  # Braking zone
            speed = 320 - (d - 3000) / 1000 * 200  # 320-120 km/h
        else:  # Corner section
            speed = 120 + 30 * np.sin((d - 4000) / 200)  # 90-150 km/h
        speed_profile.append(max(50, speed))  # Minimum 50 km/h
    
    # Create time series (approximate)
    time_seconds = np.cumsum(np.diff(np.concatenate([[0], distance])) / (np.array(speed_profile) / 3.6))
    time_deltas = pd.to_timedelta(time_seconds, unit='s')
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'Distance': distance,
        'Speed': speed_profile,
        'Time': time_deltas
    })
    
    print(f"Sample data created with {len(sample_df)} points")
    print(f"Speed range: {sample_df['Speed'].min():.1f} - {sample_df['Speed'].max():.1f} km/h")
    print(f"Distance range: {sample_df['Distance'].min():.1f} - {sample_df['Distance'].max():.1f} m")
    
    # Test different acceleration methods
    print("\nTesting acceleration computation methods...")
    
    # Method 1: Gradient (spatial derivative)
    df_gradient = compute_acceleration(sample_df, method='gradient')
    print(f"Gradient method - Acceleration range: {df_gradient['Acceleration'].min():.2f} to {df_gradient['Acceleration'].max():.2f} g")
    
    # Method 2: Time derivative
    df_time = compute_acceleration(sample_df, method='time_derivative')
    print(f"Time derivative method - Acceleration range: {df_time['Acceleration'].min():.2f} to {df_time['Acceleration'].max():.2f} g")
    
    # Method 3: Combined
    df_combined = compute_acceleration(sample_df, method='combined')
    print(f"Combined method - Acceleration range: {df_combined['Acceleration'].min():.2f} to {df_combined['Acceleration'].max():.2f} g")
    
    # Compute metrics
    print("\nAcceleration metrics (gradient method):")
    metrics = compute_acceleration_metrics(df_gradient)
    for key, value in metrics.items():
        if 'pct' in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value:.2f}")
    
    print("\nAcceleration computation module ready!")
    print("Use compute_acceleration(df) with your interpolated telemetry data.")
