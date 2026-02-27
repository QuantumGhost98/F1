# f1analytics

**F1 telemetry & performance analysis toolkit** built on top of [FastF1](https://docs.fastf1.dev).

```python
import f1analytics as f1a
```

> **v0.1.0** — 11 analysis classes • 6 helper functions • hex color palette for all 2025 drivers & teams

---

## Installation

```bash
# From the repo root
pip install -e .
```

This installs `f1analytics` in editable mode along with all dependencies (FastF1, matplotlib, pandas, numpy, scipy, scikit-learn).

---

## Quick Start

```python
import fastf1
import f1analytics as f1a

# Load a session via FastF1
session = fastf1.get_session(2025, "Monza", "Q")
session.load()

# Compare telemetry
t = f1a.Telemetry("Italian GP", 2025, "Q", session=session)
fig, ax = t.compare_laps({"LEC": "fastest", "NOR": "fastest"})

# Corner analysis
ca = f1a.CornerAnalysis("Italian GP", 2025, "Q",
                         session=session,
                         drivers=["LEC", "NOR"],
                         corner_idxs=[1, 4, 11])
fig, ax = ca.plot_all()

# Track map — colored by fastest driver per mini-sector
ms = f1a.MinisectorComparator("Italian GP", 2025, "Q",
                               session=session,
                               drivers={"LEC": "fastest", "NOR": "fastest"})
fig, ax = ms.plot_track_map()
```

> **All plot methods return `(fig, ax)`** — call `plt.show()` when ready, or save with `fig.savefig(...)`.

---

## Constructor Pattern

Every class follows the same signature:

```python
Class(session_name, year, session_type, *, session=None, drivers=None, ...)
```

| Param          | Type                      | Example                                  |
| -------------- | ------------------------- | ---------------------------------------- |
| `session_name` | `str`                     | `"Italian Grand Prix"`                   |
| `year`         | `int`                     | `2025`                                   |
| `session_type` | `str`                     | `"Q"`, `"R"`, `"FP1"`                    |
| `session`      | FastF1 session            | keyword-only                             |
| `drivers`      | `dict`, `list`, or `None` | `{"LEC": "fastest"}` or `["LEC", "NOR"]` |

---

## API Reference

### Analysis Classes

| Class                                  | What it does                                                                        | Key methods                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **`Telemetry`**                        | Full-lap telemetry overlay (Speed, Throttle, Brake, Gear, RPM, DRS) with delta-time | `compare_laps()`                                                |
| **`CornerAnalysis`**                   | Zoomed corner telemetry with speed, throttle, brake, and Δ-time subplots            | `plot_all()`, `get_corner_df()`                                 |
| **`CornerTimeComparator`**             | Per-corner elapsed-time bars showing who gains/loses where                          | `plot_corner_time_deltas()`, `get_table()`, `get_delta_table()` |
| **`CornerSpeedComparator`**            | Peak speed comparison across corners                                                | `plot_peak_speeds()`, `get_peak_speed_table()`                  |
| **`CornerMetricComparator`**           | Min-speed (or any metric) per corner — generalized version                          | `plot()`, `get_table()`                                         |
| **`MinisectorComparator`**             | Track map colored by fastest driver per N equal-distance segments                   | `plot_track_map()`, `plot_bar_chart()`, `get_segment_table()`   |
| **`DualThrottleComparisonVisualizer`** | Parallel throttle heatmap traces around the circuit                                 | `plot()`                                                        |
| **`RacePaceBoxplot`**                  | Race-pace lap-time boxplots per driver                                              | `plot()`, `get_table()`                                         |
| **`SectorDeltaPlotter`**               | Fastest sector times & deltas bar chart                                             | `plot()`, `get_table()`                                         |
| **`RacePaceAnalyzer`**                 | ML-based race-pace lap classifier (Random Forest)                                   | `train_model()`, `predict()`, `tune_hyperparameters()`          |
| **`CornerMinSpeed`**                   | _(Legacy)_ — use `CornerMetricComparator` instead                                   | `plot()`                                                        |

### Standalone Functions

| Function                             | Description                                                |
| ------------------------------------ | ---------------------------------------------------------- |
| `compute_acceleration(tel)`          | Compute longitudinal + lateral acceleration from telemetry |
| `compute_acceleration_metrics(tel)`  | Summary stats (max, mean, brake-zone G)                    |
| `compare_acceleration_profiles(...)` | Multi-driver acceleration overlay                          |
| `delta_time(ref_lap, comp_lap)`      | Sector-constrained delta-time series                       |
| `interpolate_dataframe(df)`          | High-resolution distance interpolation                     |
| `timedelta_to_seconds(td)`           | Convert timedelta → float seconds                          |

### Color Data

```python
f1a.driver_colors   # {"VER": "#3671C6", "LEC": "#E80020", ...}  — 22 drivers, hex
f1a.team_colors     # {"Ferrari": "#E80020", ...}                 — 11 teams, hex
```

---

## Cross-Session Comparison

Compare the same driver across different sessions (e.g. testing days):

```python
ca = f1a.CornerAnalysis(
    "Pre-Season Testing", 2026, "",
    laps=[
        (session_day5, "LEC", "fastest", "Day 5"),
        (session_day6, "LEC", "fastest", "Day 6"),
    ],
    corner_idxs=[1, 4]
)
fig, ax = ca.plot_all()
```

This pattern works with `CornerAnalysis`, `CornerTimeComparator`, `MinisectorComparator`, and `CornerSpeedComparator`.

---

## Architecture

```
f1analytics/
├── __init__.py                    # Public API — exports everything
├── palette.py                     # Unified hex driver/team colors
├── config.py                      # REPO_ROOT, LOGO_PATH, logger
├── plot_utils.py                  # Dark theme, branding, color assignment
├── driver_utils.py                # normalize_driver_specs()
├── corner_utils.py                # Corner resolution helpers
├── interpolate_df.py              # Distance interpolation
├── delta_time_sector_constrained.py  # Sector-aware delta-time
├── timedelta_to_seconds.py        # Timedelta conversion
├── acceleration.py                # Acceleration computation
│
├── telemetry.py                   # Telemetry class
├── corner_analysis.py             # CornerAnalysis class
├── corner_time_comparator.py      # CornerTimeComparator class
├── corner_speed.py                # CornerSpeedComparator class
├── min_speed_corner.py            # CornerMetricComparator + CornerMinSpeed
├── min_throttle_corner.py         # Min-throttle variant
├── minisector.py                  # MinisectorComparator class
├── dual_throttle_compare.py       # DualThrottleComparisonVisualizer
├── race_pace_boxplot.py           # RacePaceBoxplot class
├── fastest_sectors_deltas.py      # SectorDeltaPlotter class
└── model_prediction_race_pace.py  # RacePaceAnalyzer (ML)
```

---

## Requirements

- Python ≥ 3.9
- FastF1 ≥ 3.4
- matplotlib ≥ 3.6
- pandas ≥ 1.5
- numpy ≥ 1.22
- scipy ≥ 1.9
- scikit-learn ≥ 1.2

---

## License

Part of the [F1](https://github.com/QuantumGhost98/F1) project.
