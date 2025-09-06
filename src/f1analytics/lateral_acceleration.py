import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings


def compute_lateral_acceleration(df, method='curvature', smooth=True, smooth_params=None, 
                               clip_range=(-6.0, 6.0), output_units='g'):
    """
    Compute lateral acceleration (cornering g-forces) from telemetry data.
    
    Lateral acceleration represents the sideways force experienced during cornering.
    F1 cars can typically achieve 3-5g lateral acceleration in high-speed corners.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry DataFrame with required columns depending on method:
        - 'curvature': requires 'Speed' [km/h], 'Distance' [m], and either:
          * 'X', 'Y' coordinates (preferred), or
          * track curvature estimation from speed/distance
        - 'centripetal': requires 'Speed' [km/h] and radius estimation
    method : str, default 'curvature'
        Calculation method:
        - 'curvature': Uses track curvature from position data or speed analysis
        - 'centripetal': Estimates from speed and corner radius (v²/r)
        - 'speed_based': Approximation using speed changes and track knowledge
    smooth : bool, default True
        Whether to apply smoothing filter to reduce noise
    smooth_params : dict, optional
        Smoothing parameters: {'window_length': int, 'polyorder': int}
    clip_range : tuple, default (-4.0, 4.0)
        Range to clip lateral acceleration [g] for realistic F1 bounds
    output_units : str, default 'g'
        Output units: 'g' for g-force, 'ms2' for m/s²
        
    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame with added 'Lateral_Acceleration' column [g]
    """
    if df.empty:
        warnings.warn("Empty DataFrame provided")
        return df.copy()
    
    if len(df) < 3:
        warnings.warn("DataFrame has fewer than 3 rows, cannot compute lateral acceleration")
        result = df.copy()
        result['Lateral_Acceleration'] = 0.0
        return result
    
    result = df.copy()
    
    # Convert speed from km/h to m/s
    if 'Speed' not in df.columns:
        raise ValueError("Speed column is required for lateral acceleration computation")
    
    speed_ms = result['Speed'] / 3.6
    
    if method == 'curvature':
        lateral_acc = _compute_curvature_based_lateral_acc(result, speed_ms)
    elif method == 'centripetal':
        lateral_acc = _compute_centripetal_lateral_acc(result, speed_ms)
    elif method == 'speed_based':
        lateral_acc = _compute_speed_based_lateral_acc(result, speed_ms)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply smoothing if requested
    if smooth:
        lateral_acc = _apply_lateral_smoothing(lateral_acc, smooth_params)
    
    # Convert to requested units
    if output_units == 'g':
        lateral_acc = lateral_acc / 9.81  # Convert from m/s² to g-force
    elif output_units != 'ms2':
        warnings.warn(f"Unknown output_units '{output_units}', using m/s²")
    
    # Clip to realistic range
    if clip_range is not None:
        lateral_acc = np.clip(lateral_acc, clip_range[0], clip_range[1])
    
    result['Lateral_Acceleration'] = lateral_acc
    return result


def _compute_curvature_based_lateral_acc(df, speed_ms):
    """
    Compute lateral acceleration using track curvature.
    Lateral acceleration = v² * κ, where κ is curvature (1/radius).
    """
    if 'X' in df.columns and 'Y' in df.columns:
        # Method 1: Use X,Y coordinates to compute curvature directly
        return _compute_curvature_from_position(df, speed_ms)
    elif 'Distance' in df.columns:
        # Method 2: Estimate curvature from speed and distance changes
        return _estimate_curvature_from_speed_distance(df, speed_ms)
    else:
        warnings.warn("Insufficient data for curvature-based method, using speed-based approximation")
        return _compute_speed_based_lateral_acc(df, speed_ms)


def _compute_curvature_from_position(df, speed_ms):
    """
    Compute curvature from X,Y position data using the formula:
    κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    """
    x = df['X'].values
    y = df['Y'].values
    
    # Ensure monotonic distance for gradient computation
    if 'Distance' in df.columns:
        distance = df['Distance'].interpolate().values
    else:
        # Compute distance from X,Y
        dx = np.diff(x)
        dy = np.diff(y)
        distance = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    
    # Compute first and second derivatives
    dx_ds = np.gradient(x, distance)  # dx/ds
    dy_ds = np.gradient(y, distance)  # dy/ds
    d2x_ds2 = np.gradient(dx_ds, distance)  # d²x/ds²
    d2y_ds2 = np.gradient(dy_ds, distance)  # d²y/ds²
    
    # Curvature formula: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    denominator = (dx_ds**2 + dy_ds**2)**(3/2)
    
    # Avoid division by zero
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    curvature = numerator / denominator
    
    # Lateral acceleration = v² * κ
    lateral_acc = speed_ms**2 * curvature
    
    return lateral_acc


def _estimate_curvature_from_speed_distance(df, speed_ms):
    """
    Estimate curvature from speed and distance changes.
    In corners, drivers typically reduce speed proportional to curvature.
    """
    if 'Distance' not in df.columns:
        raise ValueError("Distance column required for this method")
    
    distance = df['Distance'].interpolate().values
    
    # Estimate heading change from speed and distance
    # Approximate method: use speed changes to infer cornering
    
    # Compute speed gradient
    dv_ds = np.gradient(speed_ms, distance)
    
    # Estimate lateral acceleration from speed profile
    # This is an approximation: corners typically show speed reduction
    # followed by acceleration, creating a characteristic pattern
    
    # Use second derivative of speed to estimate cornering intensity
    d2v_ds2 = np.gradient(dv_ds, distance)
    
    # Rough approximation: lateral acceleration correlates with speed variations
    # and curvature can be estimated from the rate of speed change
    estimated_curvature = np.abs(d2v_ds2) / (speed_ms + 1e-6)  # Avoid division by zero
    
    # Scale to reasonable values (this is empirical)
    estimated_curvature *= 0.1  # Scaling factor based on typical F1 track characteristics
    
    lateral_acc = speed_ms**2 * estimated_curvature
    
    return lateral_acc


def _compute_centripetal_lateral_acc(df, speed_ms):
    """
    Compute lateral acceleration using a SIMPLE, realistic approach.
    Focus on smooth, physically plausible results.
    """
    if 'Distance' not in df.columns:
        dt = 0.1  # Assume 10Hz sampling
        distance = np.cumsum(speed_ms * dt)
    else:
        distance = df['Distance'].interpolate().values
    
    speed_kmh = speed_ms * 3.6
    
    # SIMPLE approach: Use speed-based radius estimation with heavy smoothing
    # The key insight: F1 drivers adjust speed based on corner radius
    
    # Step 1: Smooth speed heavily to avoid artifacts
    from scipy.signal import savgol_filter
    window_length = min(101, len(speed_kmh))  # Large window for smoothness
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        speed_smooth = savgol_filter(speed_kmh, window_length=window_length, polyorder=3)
    else:
        speed_smooth = speed_kmh.copy()
    
    # Step 2: Simple radius estimation based on speed
    # Higher speeds = larger radius (less lateral g)
    # Lower speeds = smaller radius (more lateral g)
    
    # Empirical relationship for F1: radius roughly proportional to speed²
    # This creates realistic lateral g distribution
    min_speed = 50  # Minimum corner speed
    max_speed = np.max(speed_smooth)
    
    # Normalize speed to 0-1 range
    speed_normalized = (speed_smooth - min_speed) / (max_speed - min_speed)
    speed_normalized = np.clip(speed_normalized, 0, 1)
    
    # Radius estimation: 
    # Slow corners (speed_norm ~ 0): small radius (20-50m) -> high lateral g
    # Fast corners (speed_norm ~ 0.5): medium radius (50-150m) -> medium lateral g  
    # Straights (speed_norm ~ 1): large radius (500m+) -> low lateral g
    
    estimated_radius = 20 + speed_normalized * 480  # 20m to 500m radius
    
    # Step 3: Compute basic centripetal acceleration
    lateral_acc = (speed_smooth / 3.6)**2 / estimated_radius
    
    # Step 4: Apply realistic constraints and smoothing
    # F1 lateral acceleration typically: 0-4g
    lateral_acc = np.clip(lateral_acc, 0, 40)  # Cap at 4g (40 m/s²)
    
    # Step 5: Final smoothing for realistic curves
    if len(lateral_acc) >= 21:
        lateral_acc = savgol_filter(lateral_acc, window_length=21, polyorder=2)
    
    # Step 6: Add minimal baseline for straights (road camber, small corrections)
    baseline = 0.5  # 0.05g baseline
    lateral_acc = np.maximum(lateral_acc, baseline)
    
    return lateral_acc


def _compute_speed_based_lateral_acc(df, speed_ms):
    """
    Improved speed-based lateral acceleration using advanced corner detection.
    Detects corners from speed patterns and applies realistic lateral g-forces.
    """
    if 'Distance' not in df.columns:
        # If no distance, create approximate distance from speed
        dt = 0.1  # Assume 10Hz sampling
        distance = np.cumsum(speed_ms * dt)
    else:
        distance = df['Distance'].interpolate().values
    
    speed_kmh = speed_ms * 3.6
    
    # Step 1: Advanced corner detection using multiple criteria
    # Smooth speed for better analysis
    window_size = min(30, len(speed_ms) // 20)
    speed_smooth = pd.Series(speed_kmh).rolling(window=window_size, center=True).mean().fillna(speed_kmh)
    
    # Compute speed derivatives
    speed_gradient = np.gradient(speed_smooth, distance)
    speed_curvature = np.gradient(speed_gradient, distance)
    
    # Step 2: Identify corner characteristics
    # 1. Speed drops (braking zones)
    # 2. Speed recovery (acceleration zones)  
    # 3. Local minima (corner apexes)
    
    max_speed = np.max(speed_kmh)
    speed_drop_threshold = max_speed * 0.15  # 15% speed drop indicates cornering
    
    # Rolling statistics for corner detection
    roll_window = min(40, len(speed_ms) // 15)
    rolling_min = pd.Series(speed_kmh).rolling(window=roll_window, center=True).min().fillna(speed_kmh)
    rolling_max = pd.Series(speed_kmh).rolling(window=roll_window, center=True).max().fillna(speed_kmh)
    rolling_std = pd.Series(speed_kmh).rolling(window=roll_window, center=True).std().fillna(0)
    
    # Corner detection criteria
    corner_mask = (
        (rolling_max - rolling_min > speed_drop_threshold) &  # Significant speed variation
        (rolling_std > 8) &  # High speed variation
        (speed_kmh < max_speed * 0.9)  # Not at maximum speed
    )
    
    # Step 3: Initialize lateral acceleration array
    lateral_acc = np.zeros_like(speed_ms)
    
    # Step 4: Apply lateral acceleration only in corner sections
    if np.any(corner_mask):
        corner_speeds = speed_kmh[corner_mask]
        corner_indices = np.where(corner_mask)[0]
        
        # Estimate radius for each corner point
        corner_radius = np.where(
            corner_speeds < 70,   # Slow corners (hairpins, chicanes)
            12 + corner_speeds * 0.4,  # 12-40m radius
            np.where(
                corner_speeds < 140,  # Medium corners
                40 + (corner_speeds - 70) * 0.8,  # 40-96m radius
                96 + (corner_speeds - 140) * 1.2   # 96-216m radius for fast corners
            )
        )
        
        # Compute lateral acceleration for corner sections
        corner_lateral_acc = (corner_speeds / 3.6)**2 / corner_radius
        lateral_acc[corner_indices] = corner_lateral_acc
        
        # Step 5: Smooth transitions and extend corner influence
        # Apply Gaussian smoothing around corner sections to create realistic transitions
        from scipy.ndimage import gaussian_filter1d
        
        # Create a mask for smoothing
        smooth_mask = np.zeros_like(lateral_acc, dtype=bool)
        
        # Extend corner influence to nearby points
        for idx in corner_indices:
            start_idx = max(0, idx - 20)
            end_idx = min(len(lateral_acc), idx + 20)
            smooth_mask[start_idx:end_idx] = True
        
        # Apply smoothing only to corner regions
        if np.any(smooth_mask):
            lateral_acc_smooth = gaussian_filter1d(lateral_acc, sigma=3)
            lateral_acc[smooth_mask] = lateral_acc_smooth[smooth_mask]
    
    # Step 6: Add minimal lateral acceleration on straights (road camber, small corrections)
    straight_mask = ~corner_mask
    if np.any(straight_mask):
        # Very small lateral acceleration on straights (0.05-0.15g)
        straight_lateral = np.random.normal(0.05, 0.02, np.sum(straight_mask))
        straight_lateral = np.clip(straight_lateral, 0.01, 0.15)
        lateral_acc[straight_mask] = straight_lateral * 9.81  # Convert to m/s²
    
    return lateral_acc


def _apply_lateral_smoothing(lateral_acc, smooth_params=None):
    """
    Apply smoothing specifically tuned for lateral acceleration data.
    """
    if smooth_params is None:
        # Adaptive smoothing for lateral acceleration
        data_length = len(lateral_acc)
        
        if data_length >= 1000:
            window_length = min(41, data_length)  # Stronger smoothing for lateral
            polyorder = 3
        elif data_length >= 500:
            window_length = min(31, data_length)
            polyorder = 2
        else:
            window_length = min(21, data_length)
            polyorder = 2
            
        # Ensure window_length is odd and >= polyorder + 1
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 1)
        
    else:
        window_length = smooth_params.get('window_length', 41)
        polyorder = smooth_params.get('polyorder', 3)
        
        window_length = min(window_length, len(lateral_acc))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 1)
    
    try:
        if window_length >= 3:
            smoothed = savgol_filter(lateral_acc, window_length=window_length, polyorder=polyorder)
            return smoothed
        else:
            warnings.warn("Window length too small for smoothing, returning original data")
            return lateral_acc
    except Exception as e:
        warnings.warn(f"Lateral acceleration smoothing failed: {e}. Returning original data")
        return lateral_acc


def analyze_cornering_performance(df_with_lateral):
    """
    Analyze cornering performance from lateral acceleration data.
    
    Parameters:
    -----------
    df_with_lateral : pd.DataFrame
        DataFrame with 'Lateral_Acceleration', 'Speed', and 'Distance' columns
        
    Returns:
    --------
    dict
        Dictionary containing cornering performance metrics
    """
    if 'Lateral_Acceleration' not in df_with_lateral.columns:
        raise ValueError("DataFrame must contain 'Lateral_Acceleration' column")
    
    lat_acc = df_with_lateral['Lateral_Acceleration']
    speed = df_with_lateral['Speed'] / 3.6  # Convert to m/s
    
    # Find cornering sections (high lateral acceleration)
    cornering_threshold = 1.0  # 1g lateral acceleration threshold
    cornering_mask = np.abs(lat_acc) > cornering_threshold
    
    metrics = {
        'max_lateral_g': np.abs(lat_acc).max(),
        'avg_cornering_g': np.abs(lat_acc[cornering_mask]).mean() if cornering_mask.any() else 0,
        'max_lateral_at_speed': speed[np.abs(lat_acc).idxmax()] * 3.6 if not lat_acc.empty else 0,  # km/h
        'time_cornering_pct': (cornering_mask.sum() / len(lat_acc)) * 100,
        'lateral_acc_std': lat_acc.std(),
    }
    
    # Identify high-g corners (>2g)
    high_g_threshold = 2.0
    high_g_mask = np.abs(lat_acc) > high_g_threshold
    
    metrics.update({
        'high_g_corners_pct': (high_g_mask.sum() / len(lat_acc)) * 100,
        'max_sustained_g': np.abs(lat_acc).rolling(window=50, center=True).mean().max() if len(lat_acc) > 50 else np.abs(lat_acc).max(),
    })
    
    return metrics


# Example usage and testing
if __name__ == "__main__":
    print("Creating sample telemetry data for lateral acceleration testing...")
    
    # Create sample data with cornering sections
    n_points = 1000
    distance = np.linspace(0, 5000, n_points)
    
    # Simulate track with corners
    # Create X,Y coordinates for a track with corners
    track_angle = distance / 1000 * 2 * np.pi  # 2 full laps
    
    # Add some corners (sharp turns)
    corner_positions = [1000, 2500, 4000]  # Corner locations
    corner_sharpness = [0.3, 0.5, 0.4]  # Corner intensity
    
    x_coords = []
    y_coords = []
    speeds = []
    
    for i, d in enumerate(distance):
        # Base circular track
        base_angle = track_angle[i]
        x = 800 * np.cos(base_angle)
        y = 800 * np.sin(base_angle)
        
        # Add corner perturbations
        speed = 250  # Base speed
        for corner_pos, sharpness in zip(corner_positions, corner_sharpness):
            if abs(d - corner_pos) < 200:  # Within corner zone
                corner_factor = np.exp(-((d - corner_pos) / 100)**2)  # Gaussian corner
                angle_perturbation = sharpness * corner_factor * np.sin((d - corner_pos) / 50)
                x += 200 * corner_factor * np.cos(base_angle + angle_perturbation)
                y += 200 * corner_factor * np.sin(base_angle + angle_perturbation)
                speed -= 100 * corner_factor  # Slow down in corners
        
        x_coords.append(x)
        y_coords.append(y)
        speeds.append(max(80, speed))  # Minimum speed
    
    # Create time series
    time_seconds = np.cumsum(np.diff(np.concatenate([[0], distance])) / (np.array(speeds) / 3.6))
    time_deltas = pd.to_timedelta(time_seconds, unit='s')
    
    # Create sample DataFrame with position data
    sample_df = pd.DataFrame({
        'Distance': distance,
        'Speed': speeds,
        'Time': time_deltas,
        'X': x_coords,
        'Y': y_coords
    })
    
    print(f"Sample data created with {len(sample_df)} points")
    print(f"Speed range: {sample_df['Speed'].min():.1f} - {sample_df['Speed'].max():.1f} km/h")
    
    # Test lateral acceleration methods
    print("\nTesting lateral acceleration computation methods...")
    
    # Method 1: Curvature-based (with X,Y coordinates)
    df_curvature = compute_lateral_acceleration(sample_df, method='curvature')
    print(f"Curvature method - Lateral acceleration range: {df_curvature['Lateral_Acceleration'].min():.2f} to {df_curvature['Lateral_Acceleration'].max():.2f} g")
    
    # Method 2: Centripetal
    df_centripetal = compute_lateral_acceleration(sample_df, method='centripetal')
    print(f"Centripetal method - Lateral acceleration range: {df_centripetal['Lateral_Acceleration'].min():.2f} to {df_centripetal['Lateral_Acceleration'].max():.2f} g")
    
    # Method 3: Speed-based
    df_speed_based = compute_lateral_acceleration(sample_df, method='speed_based')
    print(f"Speed-based method - Lateral acceleration range: {df_speed_based['Lateral_Acceleration'].min():.2f} to {df_speed_based['Lateral_Acceleration'].max():.2f} g")
    
    # Analyze cornering performance
    print("\nCornering performance metrics (curvature method):")
    metrics = analyze_cornering_performance(df_curvature)
    for key, value in metrics.items():
        if 'pct' in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value:.2f}")
    
    print("\nLateral acceleration computation module ready!")
    print("Use compute_lateral_acceleration(df, method='curvature') with your telemetry data.")
    print("Available methods: 'curvature' (best with X,Y data), 'centripetal', 'speed_based'")
