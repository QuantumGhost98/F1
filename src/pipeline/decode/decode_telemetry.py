#!/usr/bin/env python3
"""
Decode F1 Live Timing Telemetry Data
=====================================
Decodes the base64 + zlib compressed data from F1 live timing captures
and splits each unique topic into its own decoded JSON file.

Usage:
    python decode_telemetry.py <input_file> [output_dir]

If no output dir is specified, it creates a 'decoded/' folder next to the input file.

Output structure:
    decoded/
    ├── CarData.json          (decoded from CarData.z)
    ├── Position.json         (decoded from Position.z)
    ├── TimingData.json
    ├── WeatherData.json
    ├── DriverList.json
    ├── ... (one file per topic)
    └── _all_decoded.json     (everything combined)
"""

import sys
import ast
import json
import zlib
import base64
from pathlib import Path
from collections import defaultdict

from pipeline.log import logger


def decode_z_data(encoded_str: str) -> dict | list | str:
    """
    Decode a base64 + zlib compressed F1 telemetry payload.
    
    The F1 SignalR feed sends CarData.z and Position.z as:
      1. Raw binary data
      2. Compressed with zlib/deflate
      3. Encoded to base64 string
    
    We reverse: base64 decode → zlib decompress → JSON parse.
    """
    data = encoded_str.strip().strip('"').strip("'")
    
    try:
        raw_bytes = base64.b64decode(data)
        
        decompressed = None
        for wbits in [-15, 15, 15 + 32]:
            try:
                decompressed = zlib.decompress(raw_bytes, wbits)
                break
            except zlib.error:
                continue
        
        if decompressed is None:
            return f"[DECODE_ERROR: zlib decompression failed]"
        
        text = decompressed.decode('utf-8', errors='replace')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
            
    except Exception as e:
        return f"[DECODE_ERROR: {type(e).__name__}: {e}]"


def parse_line(line: str):
    """Parse a single line from the capture file."""
    line = line.strip()
    if not line:
        return None
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        return None


def process_entry(entry: list) -> tuple[str, dict]:
    """
    Process a single entry, decoding .z compressed data if present.
    Returns (clean_topic_name, processed_entry_dict).
    """
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None, None
    
    topic = entry[0]
    data = entry[1]
    timestamp = entry[2] if len(entry) > 2 else ''
    
    # Decode compressed topics
    if topic.endswith('.z') and isinstance(data, str):
        decoded_data = decode_z_data(data)
        clean_topic = topic[:-2]  # Remove .z suffix
        result = {'data': decoded_data}
        if timestamp:
            result['timestamp'] = timestamp
        return clean_topic, result
    
    # Parse JSON strings for non-compressed entries
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            pass
    
    result = {'data': data}
    if timestamp:
        result['timestamp'] = timestamp
    return topic, result


def process_file(input_path: str, output_dir: str = None, write_combined: bool = False):
    """
    Process the capture file, decoding all compressed entries
    and splitting into one file per unique topic.
    """
    input_file = Path(input_path)
    
    if output_dir is None:
        out_dir = input_file.parent / 'decoded'
    else:
        out_dir = Path(output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Input:      %s", input_file)
    logger.info("Output dir: %s", out_dir)
    
    # Group entries by topic
    topics_data = defaultdict(list)
    all_entries = []
    
    total_lines = 0
    decoded_count = 0
    error_count = 0
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            entry = parse_line(line)
            if entry is None:
                continue
            
            clean_topic, processed = process_entry(entry)
            if clean_topic is None:
                continue
            
            # Track compressed decode stats
            raw_topic = entry[0] if isinstance(entry, (list, tuple)) else ''
            if raw_topic.endswith('.z'):
                if isinstance(processed.get('data'), str) and '[DECODE_ERROR' in str(processed['data']):
                    error_count += 1
                else:
                    decoded_count += 1
            
            topics_data[clean_topic].append(processed)
            all_entries.append({'topic': clean_topic, **processed})
            
            if total_lines % 1000 == 0:
                logger.debug("Processed %d lines...", total_lines)
    
    # Write individual topic files
    logger.info("Writing %d topic files...", len(topics_data))
    for topic, entries in sorted(topics_data.items()):
        topic_file = out_dir / f"{topic}.json"
        with open(topic_file, 'w') as f:
            json.dump(entries, f, indent=2, default=str)
        logger.info("  ✓ %s.json (%d entries)", topic, len(entries))
    
    # Write combined file only if requested
    if write_combined:
        all_file = out_dir / '_all_decoded.json'
        with open(all_file, 'w') as f:
            json.dump(all_entries, f, indent=2, default=str)
        logger.info("  ✓ _all_decoded.json (%d total entries)", len(all_entries))
    
    # Summary
    logger.info("Total lines processed:    %d", total_lines)
    logger.info("Compressed decoded (ok):  %d", decoded_count)
    logger.info("Decode errors:            %d", error_count)
    logger.info("Unique topics:            %d", len(topics_data))
    logger.info("Output directory:         %s", out_dir)
    
    for topic in sorted(topics_data.keys()):
        count = len(topics_data[topic])
        was_compressed = topic in ('CarData', 'Position')
        tag = " (was .z compressed)" if was_compressed else ""
        logger.info("  %s %5d entries%s", topic.ljust(30, '.'), count, tag)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Decode F1 live timing telemetry data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', type=str,
                        help='Path to raw capture .txt file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output directory (default: decoded/ next to input)')
    parser.add_argument('--combined', action='store_true',
                        help='Also write _all_decoded.json (large, off by default)')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    process_file(args.input, args.output, write_combined=args.combined)


if __name__ == '__main__':
    main()
