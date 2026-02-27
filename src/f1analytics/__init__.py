# src/f1analytics/__init__.py
"""
f1analytics — F1 telemetry & performance analysis toolkit.

Usage:
    import f1analytics as f1a

    # Load a session
    session = fastf1.get_session(2025, 'Monza', 'Q')
    session.load()

    # Telemetry comparison
    t = f1a.Telemetry("Italian Grand Prix", 2025, "Q", session=session)
    t.compare_laps({'LEC': 'fastest', 'NOR': 'fastest'})

    # Corner analysis
    ca = f1a.CornerAnalysis("Italian Grand Prix", 2025, "Q",
                            session=session, drivers=['LEC', 'NOR'],
                            corner_idxs=[1, 4, 11])
    ca.plot_all()

    # Minisector map
    ms = f1a.MinisectorComparator("Italian Grand Prix", 2025, "Q",
                                  session=session,
                                  drivers={'LEC': 'fastest', 'NOR': 'fastest'})
    ms.plot_track_map()

    # Race pace
    bp = f1a.RacePaceBoxplot("Italian Grand Prix", 2025, "R", session=session)
    bp.plot()
"""

__version__ = "0.1.0"

# ── Direct class imports (the "numpy-style" API) ─────────────────────────────
from f1analytics.telemetry import Telemetry
from f1analytics.corner_analysis import CornerAnalysis
from f1analytics.corner_time_comparator import CornerTimeComparator
from f1analytics.corner_speed import CornerSpeedComparator
from f1analytics.minisector import MinisectorComparator
from f1analytics.dual_throttle_compare import DualThrottleComparisonVisualizer
from f1analytics.race_pace_boxplot import RacePaceBoxplot
from f1analytics.fastest_sectors_deltas import SectorDeltaPlotter
from f1analytics.min_speed_corner import CornerMetricComparator, CornerMinSpeed
from f1analytics.model_prediction_race_pace import RacePaceAnalyzer

# ── Standalone functions ──────────────────────────────────────────────────────
from f1analytics.acceleration import (
    compute_acceleration,
    compute_acceleration_metrics,
    compare_acceleration_profiles,
)
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.timedelta_to_seconds import timedelta_to_seconds

# ── Data / config ────────────────────────────────────────────────────────────
from f1analytics.palette import driver_colors, team_colors
from f1analytics.config import REPO_ROOT, LOGO_PATH

__all__ = [
    # Classes
    "Telemetry",
    "CornerAnalysis",
    "CornerTimeComparator",
    "CornerSpeedComparator",
    "MinisectorComparator",
    "DualThrottleComparisonVisualizer",
    "RacePaceBoxplot",
    "SectorDeltaPlotter",
    "CornerMetricComparator",
    "CornerMinSpeed",
    "RacePaceAnalyzer",
    # Functions
    "compute_acceleration",
    "compute_acceleration_metrics",
    "compare_acceleration_profiles",
    "delta_time",
    "interpolate_dataframe",
    "timedelta_to_seconds",
    # Data
    "driver_colors",
    "team_colors",
]