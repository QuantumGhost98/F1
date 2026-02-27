# F1 Telemetry Pipeline

Three-stage pipeline to go from live F1 timing feed → analysis-ready DataFrames.

## Stages

```
Raw SignalR Feed → capture → decode → build_df → DataFrames
```

### capture — Record Live Data

Connects to F1's live timing SignalR feed and records raw data to a `.txt` file.

```bash
python -m pipeline.capture.live_telemetry record <output_file.txt>
```

**Input:** Live F1 timing feed  
**Output:** Raw `.txt` file (mix of JSON + base64+zlib compressed entries)

### decode — Decode Compressed Data

Decodes base64+zlib compressed `CarData.z` and `Position.z` entries and splits
each topic into its own JSON file.

```bash
python -m pipeline.decode.decode_telemetry <input.txt> [output_dir]
```

**Input:** Raw `.txt` capture file  
**Output:** `decoded/` directory with one JSON per topic (CarData.json, Position.json, etc.)

### build_df — Build DataFrames

Constructs pandas DataFrames from the decoded JSON files.

```bash
python -m pipeline.build_df.build_dataframes <decoded_dir>
```

**Input:** `decoded/` directory  
**Output:** `dataframes/` directory with pickle files (laps.pkl, telemetry.pkl, positions.pkl)

## Full Pipeline (one command)

```bash
python src/pipeline/run_pipeline.py 2026/Bahrein/test_day2.txt
```

Or step-by-step:

```bash
# 1. Capture (live session)
python src/pipeline/capture/live_telemetry.py record 2026/Bahrein/test_day2.txt

# 2. Decode
python src/pipeline/decode/decode_telemetry.py 2026/Bahrein/test_day2.txt

# 3. Build DataFrames
python src/pipeline/build_df/build_dataframes.py 2026/Bahrein/decoded/
```

## Using with f1analytics

The `load_pipeline` module provides a one-shot loader that returns a
FastF1-compatible Session, ready for use with all `f1analytics` modules:

```python
from pipeline import load_pipeline

# Load (runs decode + build if needed)
session = load_pipeline("2026/Bahrein/test_day2.txt")

# Telemetry comparison
from f1analytics.telemetry import Telemetry
t = Telemetry("Pre-Season Test", 2026, "Day 2", session=session)
t.compare_laps({'LEC': 'fastest', 'VER': 'fastest'})

# Corner analysis
from f1analytics.corner_analysis import CornerAnalysis
ca = CornerAnalysis("Pre-Season Test", 2026, "Day 2",
                    session=session, drivers=['LEC', 'VER'],
                    corner_idxs=[1, 4, 10])
ca.plot_all()

# Race pace boxplot
from f1analytics.race_pace_boxplot import RacePaceBoxplot
bp = RacePaceBoxplot(session, "Pre-Season Test", 2026, "Day 2")
bp.plot()

# Minisector map
from f1analytics.minisector import MinisectorComparator
ms = MinisectorComparator("Pre-Season Test", 2026, "Day 2",
                          session=session,
                          drivers={'LEC': 'fastest', 'NOR': 'fastest'})
ms.plot_track_map()
```

## Running Tests

```bash
python -m pytest src/pipeline/tests/ -v
```
