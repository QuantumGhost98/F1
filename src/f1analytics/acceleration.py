import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings


def compute_acceleration(df, method='time_derivative', smooth_window=None, clip_range=(-6, 2), output_units='g'):
    """
    Compute longitudinal acceleration from telemetry data.

    Uses Savitzky-Golay derivative (deriv=1) for single-pass smoothed differentiation.
    This simultaneously denoises and differentiates, preserving braking peaks.

    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry DataFrame with columns: 'Speed' [km/h], 'Distance' [m], 'Time' [timedelta]
    method : str, default 'time_derivative'
        Acceleration calculation method:
        - 'time_derivative': dv/dt (recommended — direct, cleanest)
        - 'gradient': dv/dx * v (spatial derivative, noisier at low speed)
        - 'combined': weighted average of both methods
    smooth_window : int, optional
        Savitzky-Golay window length (must be odd). If None, auto-selects based on
        data size and sample rate. Larger = smoother but may attenuate sharp peaks.
    clip_range : tuple, default (-6, 2)
        Range to clip acceleration values [g] for realistic F1 bounds.
        F1 cars typically: -5.5g braking, +1.5g acceleration (traction-limited).
    output_units : str, default 'g'
        Output units: 'g' for g-force, 'ms2' for m/s²

    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame with added 'Acceleration' column
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
    speed_ms = result['Speed'].values / 3.6

    if method == 'time_derivative':
        acceleration = _sg_time_derivative(speed_ms, result['Time'], smooth_window)

    elif method == 'gradient':
        distance = result['Distance'].interpolate().values
        acceleration = _sg_spatial_derivative(speed_ms, distance, smooth_window)

    elif method == 'combined':
        acc_time = _sg_time_derivative(speed_ms, result['Time'], smooth_window)
        distance = result['Distance'].interpolate().values
        acc_spatial = _sg_spatial_derivative(speed_ms, distance, smooth_window)
        # Weight time method higher (more stable)
        acceleration = 0.7 * acc_time + 0.3 * acc_spatial

    else:
        raise ValueError(f"Unknown method: {method}. Use 'time_derivative', 'gradient', or 'combined'")

    # Convert to requested units
    if output_units == 'g':
        acceleration = acceleration / 9.81
    elif output_units != 'ms2':
        warnings.warn(f"Unknown output_units '{output_units}', using m/s²")

    # Clip to realistic range
    if clip_range is not None:
        acceleration = np.clip(acceleration, clip_range[0], clip_range[1])

    result['Acceleration'] = acceleration
    return result



# ---------------------------------------------------------------------------
# Core derivative functions
# ---------------------------------------------------------------------------

def _sg_time_derivative(speed_ms, time_series, smooth_window=None):
    """
    Compute acceleration using Savitzky-Golay derivative: dv/dt in one pass.

    Uses savgol_filter with deriv=1 to simultaneously smooth and differentiate.
    This preserves sharp transitions (braking) while removing noise.
    """
    # Convert time to seconds
    time_seconds = _time_to_seconds(time_series)

    # Check monotonicity
    if len(time_seconds) > 1:
        dt = np.diff(time_seconds)
        if np.any(dt <= 0):
            warnings.warn(
                "Non-monotonic time detected — replacing with uniform spacing. "
                "This may distort acceleration values."
            )
            time_seconds = np.linspace(time_seconds[0], time_seconds[-1], len(time_seconds))
            dt = np.diff(time_seconds)

    # Compute median time step for SG delta parameter
    dt_median = np.median(dt) if len(dt) > 0 else 1.0

    # Auto-select window if not provided
    window = _auto_window(len(speed_ms), dt_median, smooth_window)

    # Single-pass smoothed derivative
    # deriv=1 computes the first derivative of the fitted polynomial
    # delta=dt_median scales the derivative correctly to physical units (m/s² when speed is m/s)
    acceleration = savgol_filter(
        speed_ms,
        window_length=window,
        polyorder=3,
        deriv=1,
        delta=dt_median
    )

    return acceleration


def _sg_spatial_derivative(speed_ms, distance, smooth_window=None):
    """
    Compute acceleration using spatial Savitzky-Golay derivative: dv/dx * v.

    Chain rule: a = dv/dt = (dv/dx)(dx/dt) = (dv/dx) * v
    """
    # Ensure distance is monotonic
    distance = np.copy(distance)

    dx = np.diff(distance)
    dx_median = np.median(dx[dx > 0]) if np.any(dx > 0) else 1.0

    # Auto-select window
    window = _auto_window(len(speed_ms), dx_median, smooth_window)

    # Single-pass smoothed derivative: dv/dx
    dv_dx = savgol_filter(
        speed_ms,
        window_length=window,
        polyorder=3,
        deriv=1,
        delta=dx_median
    )

    # Apply chain rule: a = dv/dx * v
    acceleration = dv_dx * speed_ms

    return acceleration


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _time_to_seconds(time_series):
    """Convert time series to float seconds array."""
    if hasattr(time_series, 'dt') and hasattr(time_series.dt, 'total_seconds'):
        return time_series.dt.total_seconds().values
    elif hasattr(time_series.iloc[0], 'total_seconds'):
        return np.array([t.total_seconds() for t in time_series])
    else:
        return time_series.values.astype(float)


def _auto_window(n_points, step_size, user_window=None):
    """
    Auto-select Savitzky-Golay window length.

    The window should cover ~1.5 seconds of data — long enough to smooth
    out interpolation noise (~15 Hz data), short enough to preserve
    F1 braking ramps (which last ~1-2 seconds).

    For ~15 Hz data: 1.5s ≈ 23 points
    For ~4 Hz data: 1.5s ≈ 7 points
    """
    if user_window is not None:
        window = user_window
    else:
        # Target ~1.5 seconds of data
        target_seconds = 1.5
        window = max(7, int(target_seconds / step_size))

    # SG filter constraints
    window = min(window, n_points)
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)  # minimum 5 for polyorder=3

    return window


# ---------------------------------------------------------------------------
# Metrics & comparison (unchanged logic, cleaned up)
# ---------------------------------------------------------------------------

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
    dec_threshold = -0.05

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

    # Align by distance
    min_dist = max(df1['Distance'].min(), df2['Distance'].min())
    max_dist = min(df1['Distance'].max(), df2['Distance'].max())

    distance_grid = np.linspace(min_dist, max_dist, 500)

    acc1_interp = np.interp(distance_grid, df1['Distance'], df1['Acceleration'])
    acc2_interp = np.interp(distance_grid, df2['Distance'], df2['Acceleration'])

    comparison_df = pd.DataFrame({
        'Distance': distance_grid,
        f'{driver1_name}_Acceleration': acc1_interp,
        f'{driver2_name}_Acceleration': acc2_interp,
        'Acceleration_Delta': acc1_interp - acc2_interp
    })

    return comparison_df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Creating sample telemetry data for testing...")

    n_points = 1000
    distance = np.linspace(0, 5000, n_points)  # 5km track

    # Simulate realistic F1 speed profile
    speed_profile = []
    for d in distance:
        if d < 1000:  # Acceleration zone
            speed = 80 + (d / 1000) * 240
        elif d < 3000:  # High speed section
            speed = 320 - 20 * np.sin((d - 1000) / 500)
        elif d < 4000:  # Braking zone
            speed = 320 - (d - 3000) / 1000 * 200
        else:  # Corner section
            speed = 120 + 30 * np.sin((d - 4000) / 200)
        speed_profile.append(max(50, speed))

    # Create time series
    time_seconds = np.cumsum(np.diff(np.concatenate([[0], distance])) / (np.array(speed_profile) / 3.6))
    time_deltas = pd.to_timedelta(time_seconds, unit='s')

    sample_df = pd.DataFrame({
        'Distance': distance,
        'Speed': speed_profile,
        'Time': time_deltas
    })

    print(f"Sample data: {len(sample_df)} points")
    print(f"Speed range: {sample_df['Speed'].min():.1f} - {sample_df['Speed'].max():.1f} km/h")

    # Test
    print("\nTesting acceleration (time_derivative method)...")
    df_result = compute_acceleration(sample_df, method='time_derivative')
    acc = df_result['Acceleration']
    print(f"  Range: {acc.min():.2f}g to {acc.max():.2f}g")
    print(f"  Braking peak: {acc.min():.2f}g (expected ~-4 to -6g)")
    print(f"  Accel peak: {acc.max():.2f}g (expected ~+0.5 to +1.5g)")

    print("\nTesting acceleration (gradient/spatial method)...")
    df_spatial = compute_acceleration(sample_df, method='gradient')
    acc_s = df_spatial['Acceleration']
    print(f"  Range: {acc_s.min():.2f}g to {acc_s.max():.2f}g")

    # Metrics
    print("\nAcceleration metrics:")
    metrics = compute_acceleration_metrics(df_result)
    for key, value in metrics.items():
        if 'pct' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value:.2f}")

    print("\n✅ Acceleration module ready.")
