# F1 Telemetry Pipeline

Three-stage pipeline to go from live F1 timing feed → analysis-ready DataFrames.

## Stages

```
Raw SignalR Feed → 01_parse → 02_decode → 03_build_df → DataFrames
```

### 01_parse — Capture Live Data

Connects to F1's live timing SignalR feed and records raw data to a `.txt` file.

```bash
python -m pipeline.01_parse.live_telemetry record <output_file.txt>
```

**Input:** Live F1 timing feed  
**Output:** Raw `.txt` file (mix of JSON + base64+zlib compressed entries)

### 02_decode — Decode Compressed Data

Decodes base64+zlib compressed `CarData.z` and `Position.z` entries and splits
each topic into its own JSON file.

```bash
python -m pipeline.02_decode.decode_telemetry <input.txt> [output_dir]
```

**Input:** Raw `.txt` capture file  
**Output:** `decoded/` directory with one JSON per topic (CarData.json, Position.json, etc.)

### 03_build_df — Build DataFrames

Constructs pandas DataFrames from the decoded JSON files.

```bash
python -m pipeline.03_build_df.build_dataframes <decoded_dir>
```

**Input:** `decoded/` directory  
**Output:** `dataframes/` directory with pickle files (laps.pkl, telemetry.pkl, positions.pkl)

## Full Pipeline Example

```bash
# 1. Capture (live session)
python src/pipeline/01_parse/live_telemetry.py record 2026/Bahrein/test_day2.txt

# 2. Decode
python src/pipeline/02_decode/decode_telemetry.py 2026/Bahrein/test_day2.txt

# 3. Build DataFrames
python src/pipeline/03_build_df/build_dataframes.py 2026/Bahrein/decoded/

# 4. Analyze
python -c "import pandas as pd; laps = pd.read_pickle('2026/Bahrein/dataframes/laps.pkl'); print(laps)"
```
