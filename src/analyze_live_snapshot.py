#!/usr/bin/env python3
"""
Analyze the current state of the live recording WITHOUT stopping it.

This script reads the current recording file, loads it with FastF1,
and generates telemetry analysis using your existing Telemetry class.

Usage:
    python src/analyze_live_snapshot.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastf1.livetiming.data import LiveTimingData
import fastf1
from f1analytics.telemetry import Telemetry
import warnings
warnings.filterwarnings('ignore')

RECORDING_FILE = "2026/Bahrein/test_day1.txt"

def main():
    print("=" * 60)
    print("🏎️  F1 Live Recording Analysis (Snapshot)")
    print("=" * 60)
    
    if not os.path.exists(RECORDING_FILE):
        print(f"❌ Recording file not found: {RECORDING_FILE}")
        print("Make sure the recording is still running!")
        return
    
    # Count messages
    with open(RECORDING_FILE) as f:
        msg_count = sum(1 for _ in f)
    
    print(f"📊 Current recording: {msg_count:,} messages")
    print(f"📁 File: {RECORDING_FILE}")
    print("\n🔄 Loading data with FastF1...")
    
    try:
        # Load the live timing data
        livedata = LiveTimingData(RECORDING_FILE)
        livedata.load()
        
        categories = livedata.list_categories()
        print(f"✅ Loaded {len(categories)} data categories:")
        for cat in sorted(categories):
            entries = livedata.get(cat)
            print(f"   • {cat}: {len(entries)} entries")
        
        # Check for essential telemetry data
        has_car_data = 'CarData.z' in categories
        has_position = 'Position.z' in categories
        
        if not has_car_data or not has_position:
            print("\n⚠️  Full telemetry not available yet")
            print(f"   CarData.z: {'✅' if has_car_data else '❌'}")
            print(f"   Position.z: {'✅' if has_position else '❌'}")
            print("\n💡 This is normal early in the session!")
            print("   Full telemetry appears once cars complete flying laps.")
            print("   Keep recording and try again in a few minutes.\n")
            
            # Show what we DO have from TimingData
            if 'TimingData' in categories:
                print("📊 Available timing data:")
                timing_data = livedata.get('TimingData')
                print(f"   {len(timing_data)} timing updates")
                
                # Try to extract some basic info
                drivers_seen = set()
                for entry in timing_data:
                    if 'Lines' in entry:
                        drivers_seen.update(entry['Lines'].keys())
                
                print(f"   Drivers seen: {', '.join(sorted(drivers_seen))}")
            
            return
        
        # Create a session
        print("\n🏁 Creating session...")
        session = fastf1.get_testing_session(2026, 2, 1)  # Year 2026, Test 2, Day 1
        
        print("⏳ Loading session data (this may take a minute)...")
        session.load(livedata=livedata)
        
        print(f"✅ Session loaded successfully!")
        
        # Check if laps are available
        if not hasattr(session, '_laps') or session._laps is None:
            print("\n⚠️  No lap data loaded yet")
            print("   This means cars haven't completed full flying laps with telemetry.")
            print("   Keep recording and try again later!")
            return
        
        laps = session.laps
        print(f"   Laps in session: {len(laps)}")
        
        if len(laps) == 0:
            print("\n⚠️  No complete laps yet")
            print("   Cars are still on out-laps or data is too early.")
            print("   Keep recording and try again in a few minutes!")
            return
        
        # List drivers with laps
        print("\n🏎️  Drivers with recorded laps:")
        drivers_with_laps = laps.groupby('Driver').size().sort_values(ascending=False)
        for driver, count in drivers_with_laps.items():
            print(f"   {driver}: {count} lap(s)")
        
        # Find the fastest lap overall
        fastest_lap = laps.pick_fastest()
        fastest_driver = fastest_lap['Driver']
        fastest_time = fastest_lap['LapTime']
        
        print(f"\n⚡ Fastest lap so far:")
        print(f"   Driver: {fastest_driver}")
        print(f"   Time: {fastest_time}")
        
        # Use YOUR Telemetry class!
        print("\n📈 Generating telemetry analysis with your Telemetry class...")
        tel = Telemetry(session, "Bahrain Pre-Season Test", 2026, "T")
        
        # Plot the fastest lap
        print(f"   Plotting {fastest_driver}'s fastest lap...")
        tel.compare_laps([fastest_driver], 
                        channels=['Speed', 'Throttle', 'Brake', 'nGear', 'RPM'],
                        session_label="LIVE Recording Snapshot")
        
        print("\n✅ Analysis complete! Check the plot window.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: The recording might need more time to capture complete laps with full telemetry.")


if __name__ == "__main__":
    main()
