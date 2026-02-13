# f1analytics API Reference

This document lists all public modules, classes, and functions available in the `f1analytics` package.

## Core Utilities

Foundational modules used across the package.

### `f1analytics.config`

Package-level configuration constants.

- `REPO_ROOT` (Path): Absolute path to the repository root.
- `LOGO_PATH` (Path): Absolute path to the branding logo.
- `ATTRIBUTION` (str): Standard attribution text ("Provided by: ...").
- `logger` (logging.Logger): Package-level logger instance.

### `f1analytics.plot_utils`

Shared plotting utilities for consistent branding and theming.

- `setup_dark_theme(fig, axes)`: Apply the standard dark background theme.
- `add_branding(fig, text_pos=(0.9, 0.96), logo_pos=None)`: Add attribution text and logo.
- `assign_colors(driver_specs, ...)`: Assign consistent colors to a list of driver specifications.
- `assign_colors_simple(drivers)`: Simplified color assignment for raw driver codes.
- `adjust_brightness(color, factor)`: Lighten (>1) or darken (<1) a hex/rgb color.
- `finalize_plot(fig, axes=None, save_path=None, show=False)`: Apply layout, save, and show plot.

### `f1analytics.corner_utils`

Corner resolution and labeling logic.

- `corner_identifier_to_index(circuit_info, label)`: Resolve "1", "1A" to DataFrame index.
- `corner_label(circuit_info, apex_idx)`: Get label (e.g., "1") for a corner index.
- `resolve_corner_idxs(circuit_info, corner_idxs)`: Normalize varied inputs (int, str, ranges) into a list of indices.

### `f1analytics.driver_utils`

Driver specification normalization.

- `normalize_driver_specs(drivers, max_specs=3)`: Convert dict/list/tuple inputs into a standard list of driver dictionaries.

---

## Telemetry Analysis

Core telemetry visualization and calculation.

### `f1analytics.telemetry`

- **`class Telemetry(session, session_name, year, session_type)`**
  - `compare_laps(drivers, channels=None, ...)`: Compare telemetry traces between drivers. Returns `(fig, axes)`.

### `f1analytics.cross_session_telemetry`

- **`class CrossSessionTelemetry(sessions)`**
  - `compare_laps(session_drivers, ...)`: Compare laps across different sessions/years. Returns `(fig, axes)`.

### `f1analytics.acceleration`

Longitudinal acceleration analysis.

- `compute_acceleration(df, ...)`: Calculate longitudinal g-force.
- `compute_total_acceleration(df, ...)`: Vector sum of longitudinal and lateral g-forces.
- `compare_acceleration_profiles(df1, df2, ...)`: Compare profiles between two drivers.

### `f1analytics.lateral_acceleration`

Lateral (cornering) acceleration analysis.

- `compute_lateral_acceleration(df, ...)`: Calculate lateral g-force using curvature.
- `analyze_cornering_performance(df)`: Extract cornering metrics (min/max/avg g).

### `f1analytics.delta_time_sector_constrained`

Advanced delta calculation.

- `delta_time(reference_lap, comparison_lap)`: Calculate accurate delta time constrained by sector timing lines.

---

## Corner Analysis

Specialized analysis of specific corners or track sections.

### `f1analytics.corner_analysis`

- **`class CornerAnalysis(..., drivers, corner_idxs, before=50, after=50)`**
  - `plot_all(channels=None, save_path=None)`: Plot telemetry channels for specific corners. Returns `(fig, axes)`.

### `f1analytics.corner_speed`

- **`class CornerSpeedComparator(..., drivers, mode='min')`**
  - `plot_peak_speeds(margin=50, save_path=None)`: Bar chart of min (apex) or max speeds per corner. Returns `(fig, ax)`.

### `f1analytics.min_speed_corner` / `min_throttle_corner`

- **`class CornerMetricComparator(..., metric='Speed', mode='min')`**
  - `plot(save_path=None)`: Bar chart of any metric (Speed, Throttle) per corner.
- **`class CornerMinSpeed`**: Alias for `CornerMetricComparator` (metric='Speed').
- **`class CornerMinThrottle`**: Alias for `CornerMetricComparator` (metric='Throttle').

### `f1analytics.corner_time_comparator`

- **`class CornerTimeComparator(..., drivers)`**
  - `plot_corner_time_deltas(save_path=None)`: Bar chart showing time gained/lost at each corner relative to baseline.

---

## Race Pace & Strategy

Long-run and session-level analysis.

### `f1analytics.race_pace_boxplot`

- **`class RacePaceBoxplot(..., drivers=None)`**
  - `plot(save_path=None)`: Boxplot distribution of lap times.
  - `get_table()`: Summary statistics table (mean, median, std).

### `f1analytics.fastest_sectors_deltas`

- **`class SectorDeltaPlotter`**
  - `plot(top_n=10, save_path=None)`: Stacked bar chart of sector time deltas.
  - `get_table(top_n=10)`: DataFrame of fastest sectors.

### `f1analytics.session_delta_chart`

- `create_session_delta_chart(session1, label1, session2, label2, save_path=None)`: Bar chart comparing fastest lap times between two sessions.

### `f1analytics.dual_throttle_compare`

- **`class DualThrottleComparisonVisualizer`**
  - `plot(figsize=(14,10), save_path=None)`: Concentric track map comparing throttle application traces.

### `f1analytics.model_prediction_race_pace`

Machine learning for race pace classification.

- **`class RacePaceAnalyzer`**
  - `train_model(data=None)`: Train the race pace classifier.
  - `get_race_pace_laps(event_data, ...)`: Identify and filter representative race pace laps.
  - `save_model(path)` / `load_model(path)`: Persist trained model.

---

## Data Helpers

Low-level data manipulation.

### `f1analytics.interpolate_df`

- `interpolate_dataframe(df, ...)`: Resample telemetry to a consistent distance grid for comparison.

### `f1analytics.timedelta_to_seconds`

- `timedelta_to_seconds(td)`: Convert pandas TimeDeltas to float seconds.

### `f1analytics.colors_pilots` / `team_palette`

- `colors_pilots`: Dictionary mapping driver codes to colors.
- `team_palette`: Dictionary mapping team names to hex colors.
