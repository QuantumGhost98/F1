#!/usr/bin/env python3
"""
F1 Live Telemetry Capture & Analysis (Bahrain 2026)

Uses FastF1's SignalR Core client to connect directly to
livetiming.formula1.com and capture live timing data. Requires an
F1TV Access/Pro/Premium subscription for full telemetry (CarData.z,
Position.z).

Two modes:
  1. RECORD: Capture live data to a file during the session
  2. ANALYZE: Parse and display summary of a previously recorded file

Usage:
  # Record live data (run this DURING the session)
  python src/live_telemetry.py record bahrain_test_day1.txt

  # Record with longer timeout (useful for long gaps between runs)
  python src/live_telemetry.py record bahrain_test_day1.txt --timeout 300

  # Record without F1TV auth (partial data only)
  python src/live_telemetry.py record bahrain_test_day1.txt --no-auth

  # Analyze recorded data
  python src/live_telemetry.py analyze bahrain_test_day1.txt

  # Check F1TV authentication status
  python src/live_telemetry.py auth --status

  # Authenticate with F1TV (opens browser)
  python src/live_telemetry.py auth --login

  # Clear saved F1TV credentials
  python src/live_telemetry.py auth --clear
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd

from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData
from signalrcore.messages.completion_message import CompletionMessage


# ─── Enhanced SignalR Client with Real-Time Console Output ─────────────────

class LiveTelemetryClient(SignalRClient):
    """Extended SignalR Core client that prints live data to console
    while recording.

    Inherits from FastF1's SignalRClient (v3.8.0+) and adds real-time
    console display of incoming timing data, car data, and position updates.

    The new FastF1 v3.8.0 client uses:
      - signalrcore (SignalR Core protocol) instead of the old websockets
      - Synchronous execution (no asyncio)
      - F1TV account authentication via get_auth_token()
    """

    def __init__(self, filename: str, filemode: str = 'w',
                 timeout: int = 120, verbose: bool = True,
                 logger: Optional[logging.Logger] = None,
                 no_auth: bool = False):
        super().__init__(filename=filename, filemode=filemode,
                         timeout=timeout, logger=logger,
                         no_auth=no_auth)
        self.verbose = verbose
        self._msg_count = 0
        self._categories_seen = set()
        self._latest_timing = {}
        self._latest_driver_list = {}

    def _on_message(self, msg):
        """Override to add real-time console display after writing to file."""
        # Call parent to handle file writing
        super()._on_message(msg)
        self._msg_count += 1

        # Parse and display if verbose
        if self.verbose:
            self._display_message(msg)

    def _display_message(self, msg):
        """Parse and display a SignalR message in the console."""
        try:
            # v3.8.0 messages can be CompletionMessage or list
            if isinstance(msg, CompletionMessage):
                # Initial subscription response — multiple categories
                if msg.result and isinstance(msg.result, dict):
                    for category, data in msg.result.items():
                        self._categories_seen.add(category)
                        self._route_category(category, data, "")

            elif isinstance(msg, list) and len(msg) >= 2:
                category = msg[0]
                data = msg[1]
                timestamp = msg[2] if len(msg) > 2 else ""
                self._categories_seen.add(category)
                self._route_category(category, data, timestamp)

            # Periodic status line
            if self._msg_count % 50 == 0:
                cats = ", ".join(sorted(self._categories_seen))
                has_car = "CarData.z" in self._categories_seen
                has_pos = "Position.z" in self._categories_seen
                telemetry_status = "✅ FULL" if (has_car and has_pos) else "⚠️  TIMING ONLY"
                print(f"\n📊 [{self._msg_count} messages] "
                      f"Telemetry: {telemetry_status}")
                print(f"   Categories: {cats}\n")

        except Exception:
            pass  # Don't crash on display errors

    def _route_category(self, category, data, timestamp):
        """Route a message to the appropriate display handler."""
        if category == "TimingData":
            self._display_timing_data(data, timestamp)
        elif category == "CarData.z":
            self._display_car_data(data, timestamp)
        elif category == "Position.z":
            self._display_position_data(data, timestamp)
        elif category == "WeatherData":
            self._display_weather(data, timestamp)
        elif category == "RaceControlMessages":
            self._display_race_control(data, timestamp)
        elif category == "TrackStatus":
            self._display_track_status(data, timestamp)
        elif category == "SessionInfo":
            self._display_session_info(data, timestamp)
        elif category == "DriverList":
            self._update_driver_list(data)
        elif category == "SessionData":
            self._display_session_data(data, timestamp)
        elif category == "LapCount":
            self._display_lap_count(data, timestamp)

    def _get_driver_name(self, driver_num: str) -> str:
        """Look up driver abbreviation from number."""
        if driver_num in self._latest_driver_list:
            info = self._latest_driver_list[driver_num]
            return info.get("Tla", driver_num)
        return f"#{driver_num}"

    def _update_driver_list(self, data):
        """Update internal driver list mapping."""
        if isinstance(data, dict):
            self._latest_driver_list.update(data)
            if self._msg_count <= 5:  # Show driver list on first reception
                print("\n🏎️  Driver List:")
                for num, info in sorted(data.items()):
                    if isinstance(info, dict):
                        tla = info.get("Tla", "???")
                        name = info.get("FullName", "")
                        team = info.get("TeamName", "")
                        print(f"   #{num:>2} {tla} - {name} ({team})")
                print()

    def _display_timing_data(self, data, timestamp):
        """Display timing updates (lap times, sectors, gaps)."""
        if not isinstance(data, dict) or "Lines" not in data:
            return

        for driver_num, driver_data in data["Lines"].items():
            if not isinstance(driver_data, dict):
                continue

            name = self._get_driver_name(driver_num)
            parts = []

            # Last lap time
            if "LastLapTime" in driver_data:
                lt = driver_data["LastLapTime"]
                if isinstance(lt, dict) and "Value" in lt:
                    parts.append(f"Lap: {lt['Value']}")

            # Sector times
            if "Sectors" in driver_data:
                sectors = driver_data["Sectors"]
                if isinstance(sectors, dict):
                    for i in range(3):
                        if str(i) in sectors:
                            sec = sectors[str(i)]
                            if isinstance(sec, dict) and "Value" in sec:
                                parts.append(f"S{i+1}: {sec['Value']}")

            # Speed traps
            if "Speeds" in driver_data:
                speeds = driver_data["Speeds"]
                if isinstance(speeds, dict):
                    for trap, val in speeds.items():
                        if isinstance(val, dict) and "Value" in val:
                            parts.append(f"{trap}: {val['Value']} km/h")

            # Gap to leader / interval
            if "GapToLeader" in driver_data:
                gap = driver_data["GapToLeader"]
                if isinstance(gap, dict) and "Value" in gap:
                    parts.append(f"Gap: {gap['Value']}")

            if "IntervalToPositionAhead" in driver_data:
                iv = driver_data["IntervalToPositionAhead"]
                if isinstance(iv, dict) and "Value" in iv:
                    parts.append(f"Int: {iv['Value']}")

            if parts:
                ts = timestamp[:12] if timestamp else ""
                print(f"⏱️  [{ts}] {name:>4} │ {' │ '.join(parts)}")

    def _display_car_data(self, data, timestamp):
        """CarData.z arrives compressed — show receipt confirmation."""
        if self._msg_count <= 10:
            print(f"📡 CarData.z received (compressed telemetry stream active)")

    def _display_position_data(self, data, timestamp):
        """Position.z arrives compressed — show receipt confirmation."""
        if self._msg_count <= 10:
            print(f"📍 Position.z received (position stream active)")

    def _display_weather(self, data, timestamp):
        """Display weather updates."""
        if isinstance(data, dict):
            temp_air = data.get("AirTemp", "?")
            temp_track = data.get("TrackTemp", "?")
            humidity = data.get("Humidity", "?")
            wind_speed = data.get("WindSpeed", "?")
            wind_dir = data.get("WindDirection", "?")
            rain = data.get("Rainfall", "0")
            print(f"🌦️  Weather │ Air: {temp_air}°C │ Track: {temp_track}°C │ "
                  f"Humidity: {humidity}% │ Wind: {wind_speed} km/h ({wind_dir}°) │ "
                  f"Rain: {rain}")

    def _display_race_control(self, data, timestamp):
        """Display race control messages (flags, incidents, etc)."""
        if isinstance(data, dict):
            messages = data.get("Messages", {})
            if isinstance(messages, dict):
                for _, msg in messages.items():
                    if isinstance(msg, dict):
                        cat = msg.get("Category", "")
                        text = msg.get("Message", "")
                        print(f"🚩 Race Control │ [{cat}] {text}")

    def _display_track_status(self, data, timestamp):
        """Display track status changes."""
        if isinstance(data, dict):
            status = data.get("Status", "")
            message = data.get("Message", "")
            print(f"🏁 Track Status │ {status} - {message}")

    def _display_session_info(self, data, timestamp):
        """Display session information."""
        if isinstance(data, dict):
            meeting = data.get("Meeting", {})
            name = meeting.get("Name", "") if isinstance(meeting, dict) else ""
            stype = data.get("Name", "")
            print(f"\n{'='*60}")
            print(f"📋 Session: {name} - {stype}")
            print(f"{'='*60}\n")

    def _display_session_data(self, data, timestamp):
        """Display session status changes."""
        if isinstance(data, dict):
            series = data.get("StatusSeries", {})
            if isinstance(series, dict):
                for _, entry in series.items():
                    if isinstance(entry, dict):
                        status = entry.get("SessionStatus", "")
                        if status:
                            emoji = {"Started": "🟢", "Finished": "🏁",
                                     "Inactive": "⚪", "Aborted": "🔴"
                                     }.get(status, "📋")
                            print(f"{emoji} Session Status: {status}")

    def _display_lap_count(self, data, timestamp):
        """Display lap count updates."""
        if isinstance(data, dict):
            current = data.get("CurrentLap", "?")
            total = data.get("TotalLaps", "?")
            print(f"🔄 Lap {current}/{total}")


# ─── Post-Session Analysis ─────────────────────────────────────────────────

def analyze_recorded_data(filename: str):
    """Parse and display summary of a recorded live timing file.

    Args:
        filename: Path to the recorded .txt file
    """
    print(f"\n📂 Loading recorded data from: {filename}\n")

    livedata = LiveTimingData(filename)
    livedata.load()

    categories = livedata.list_categories()
    print(f"📊 Available data categories ({len(categories)}):")

    has_car_data = False
    has_position = False
    for cat in sorted(categories):
        entries = livedata.get(cat)
        marker = ""
        if cat == "CarData.z":
            has_car_data = True
            marker = " ✅"
        elif cat == "Position.z":
            has_position = True
            marker = " ✅"
        print(f"   • {cat}: {len(entries)} entries{marker}")

    if has_car_data and has_position:
        print(f"\n✅ Full telemetry available (CarData.z + Position.z)")
    else:
        missing = []
        if not has_car_data:
            missing.append("CarData.z")
        if not has_position:
            missing.append("Position.z")
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        print(f"   Tip: Use --no-auth=false and ensure F1TV subscription is active")

    print(f"\n{'='*60}")
    print("To load this data into FastF1 for full telemetry analysis:")
    print("=" * 60)
    print(f"""
from fastf1.livetiming.data import LiveTimingData
import fastf1

# Load the recorded data
livedata = LiveTimingData('{filename}')

# Use it with FastF1's session loading
session = fastf1.get_testing_session(2026, 2, 1)  # Test 2, Day 1
session.load(livedata=livedata)

# Now you have full access to telemetry!
laps = session.laps
tel = laps.pick_driver('VER').pick_fastest().get_telemetry()
print(tel[['Speed', 'RPM', 'nGear', 'Throttle', 'Brake']].head())
""")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Live Telemetry Capture & Analysis (Bahrain 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record live data during a session (with F1TV auth)
  python src/live_telemetry.py record bahrain_test_day1.txt

  # Record with 5-minute timeout (for long gaps between runs)
  python src/live_telemetry.py record bahrain_test_day1.txt --timeout 300

  # Record without auth (timing data only, no CarData.z)
  python src/live_telemetry.py record bahrain_test_day1.txt --no-auth

  # Quietly record (no console output)
  python src/live_telemetry.py record bahrain_test_day1.txt --quiet

  # Analyze a recorded file
  python src/live_telemetry.py analyze bahrain_test_day1.txt

  # Check F1TV auth status
  python src/live_telemetry.py auth --status
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Record command
    rec = subparsers.add_parser("record", help="Record live timing data")
    rec.add_argument("file", type=str, help="Output file path (.txt)")
    rec.add_argument("--timeout", type=int, default=120,
                     help="Seconds of silence before auto-exit (default: 120)")
    rec.add_argument("--append", action="store_true",
                     help="Append to existing file instead of overwriting")
    rec.add_argument("--quiet", action="store_true",
                     help="Don't print live data to console")
    rec.add_argument("--no-auth", action="store_true",
                     help="Skip F1TV authentication (partial data only)")

    # Analyze command
    ana = subparsers.add_parser("analyze", help="Analyze recorded data")
    ana.add_argument("file", type=str, help="Recorded data file (.txt)")

    # Auth command
    auth = subparsers.add_parser("auth", help="Manage F1TV authentication")
    auth_group = auth.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--status", action="store_true",
                            help="Check current auth status")
    auth_group.add_argument("--login", action="store_true",
                            help="Authenticate with F1TV (opens browser)")
    auth_group.add_argument("--clear", action="store_true",
                            help="Clear saved authentication token")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "record":
        print("=" * 60)
        print("🏎️  F1 Live Telemetry Recorder (FastF1 v3.8.0)")
        print("=" * 60)
        print(f"📁 Output file:  {args.file}")
        print(f"⏱️  Timeout:      {args.timeout}s")
        print(f"📝 Mode:         {'append' if args.append else 'overwrite'}")
        print(f"🔊 Verbose:      {not args.quiet}")
        print(f"🔐 Auth:         {'disabled (--no-auth)' if args.no_auth else 'F1TV'}")
        print("=" * 60)

        if not args.no_auth:
            print("\n🔐 Authenticating with F1TV account...")
            print("   (First time? A browser window will open for login)\n")
        else:
            print("\n⚠️  Running without auth — CarData.z/Position.z may not")
            print("   be available. Use F1TV auth for full telemetry.\n")

        print("Connecting to livetiming.formula1.com/signalrcore ...")
        print("Press Ctrl+C to stop recording\n")

        mode = "a" if args.append else "w"
        client = LiveTelemetryClient(
            filename=args.file,
            filemode=mode,
            timeout=args.timeout,
            verbose=not args.quiet,
            no_auth=args.no_auth,
        )
        client.start()

    elif args.command == "analyze":
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        analyze_recorded_data(args.file)

    elif args.command == "auth":
        from fastf1.internals.f1auth import (
            get_auth_token, clear_auth_token,
            print_auth_status, print_auth_token
        )

        if args.status:
            print("\n🔐 F1TV Authentication Status\n" + "=" * 40)
            print_auth_status()

        elif args.login:
            print("\n🔐 Starting F1TV authentication...")
            print("   A browser window will open for login.\n")
            token = get_auth_token()
            if token:
                print("\n✅ Authentication successful!")
                print("   Token is cached — you won't need to login again")
                print("   until it expires.\n")
            else:
                print("\n❌ Authentication failed. Please try again.\n")

        elif args.clear:
            clear_auth_token()
            print("✅ Authentication token cleared.\n")


if __name__ == "__main__":
    main()
