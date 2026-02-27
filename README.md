# 🏎️ F1 Analytics

Formula 1 telemetry analysis, performance comparison, and race-pace prediction — powered by [FastF1](https://docs.fastf1.dev) and the custom **f1analytics** library.

---

## Project Structure

```
F1/
├── src/
│   ├── f1analytics/          # Core analysis library (pip installable)
│   └── pipeline/             # Real-time telemetry capture & processing
├── 2024/                     # Season analysis notebooks
├── 2025/                     # Season analysis notebooks
├── 2026/                     # Pre-season testing notebooks
├── models/                   # Trained ML models (race pace prediction)
└── pyproject.toml            # Package config & dependencies
```

---

## f1analytics Library

A numpy-style analysis toolkit — `import f1analytics as f1a` and access everything directly.

```python
import fastf1
import f1analytics as f1a

session = fastf1.get_session(2025, "Monza", "Q")
session.load()

# Telemetry overlay with delta-time
fig, ax = f1a.Telemetry("Italian GP", 2025, "Q", session=session)\
    .compare_laps(drivers={"LEC": "fastest", "NOR": "fastest"})

# Track map colored by fastest driver per mini-sector
fig, ax = f1a.MinisectorComparator("Italian GP", 2025, "Q",
    session=session, drivers={"LEC": "fastest", "NOR": "fastest"})\
    .plot_track_map()
```

### Available Classes

| Class                              | Analysis                                                            |
| ---------------------------------- | ------------------------------------------------------------------- |
| `Telemetry`                        | Full-lap telemetry overlay (Speed, Throttle, Brake, Gear, RPM, DRS) |
| `CornerAnalysis`                   | Zoomed corner telemetry with speed, throttle, brake subplots        |
| `CornerTimeComparator`             | Per-corner elapsed-time deltas                                      |
| `CornerSpeedComparator`            | Peak/min speed comparison across corners                            |
| `CornerMetricComparator`           | Generalized per‑corner metric comparison                            |
| `MinisectorComparator`             | Track map colored by fastest driver per segment                     |
| `DualThrottleComparisonVisualizer` | Parallel throttle heatmap traces                                    |
| `RacePaceBoxplot`                  | Race-pace lap-time boxplots                                         |
| `SectorDeltaPlotter`               | Fastest sector times & deltas                                       |
| `RacePaceAnalyzer`                 | ML-based race‑pace classification (Random Forest)                   |

### Cross-Session Comparison

Compare the same driver across different sessions:

```python
fig, ax = f1a.Telemetry("Pre-Season", 2026, "Testing").compare_laps(
    laps=[
        (session_day5, "LEC", "fastest", "Day 5"),
        (session_day6, "LEC", "fastest", "Day 6"),
    ]
)
```

> 📖 Full API docs: [`src/f1analytics/README.md`](src/f1analytics/README.md)

---

## Pipeline

Real-time UDP telemetry capture from the F1 game, with decoding and DataFrame construction.

```
pipeline/
├── capture/          # UDP packet capture (monitor_recording.py)
├── decode/           # Binary → structured data (decode_telemetry.py)
├── build_df/         # Structured data → analysis-ready DataFrames
├── load_session.py   # FastF1 session loader with caching
├── run_pipeline.py   # Orchestrates capture → decode → build
└── tests/            # Pipeline unit tests
```

> 📖 Full pipeline docs: [`src/pipeline/README.md`](src/pipeline/README.md)

---

## Setup

```bash
# Clone
git clone https://github.com/QuantumGhost98/F1.git
cd F1

# Install f1analytics + dependencies
pip install -e .
```

### Requirements

- Python ≥ 3.9
- FastF1 ≥ 3.4
- matplotlib, pandas, numpy, scipy, scikit-learn

---

## Season Notebooks

| Folder  | Content                                                      |
| ------- | ------------------------------------------------------------ |
| `2024/` | Race weekend analyses (telemetry, strategy, pace)            |
| `2025/` | Full season analysis notebooks                               |
| `2026/` | Pre-season testing comparisons (cross-session, 2022 vs 2026) |

---

## License

Personal project by [@QuantumGhost98](https://github.com/QuantumGhost98).
