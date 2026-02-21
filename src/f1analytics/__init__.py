# src/f1analytics/__init__.py
"""
Top-level API for f1analytics.

- Lazily exposes all submodules so you can do:
    import f1analytics as f1
    f1.acceleration.compute_acceleration(...)
    f1.corner_time_comparator.compare_corners(...)

- Provides a few convenient function shortcuts:
    f1.compute_acceleration(...)
    f1.compute_total_acceleration(...)
    f1.compare_corners(...)
    f1.analyze_corners(...)
    f1.timedelta_to_seconds(...)
"""

from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List

__version__ = "0.0.1"

# Every module in your package (keep this list in sync if you add new files)
_SUBMODULES: List[str] = [
    "acceleration",
    "colors_pilots",
    "config",
    "corner_analysis",
    "corner_speed",
    "corner_time_comparator",
    "corner_utils",
    "delta_time_sector_constrained",
    "driver_utils",
    "fastest_sectors_deltas",
    "interpolate_df",
    "min_speed_corner",
    "min_throttle_corner",
    "plot_utils",
    "team_palette",
    "telemetry",
    "timedelta_to_seconds",
    "dual_throttle_compare",
    "model_prediction_race_pace",
    "race_pace_boxplot",
    "cross_session_telemetry",
    "session_delta_chart",
    "minisector",

]

# Internal cache for loaded submodules
_loaded: Dict[str, ModuleType] = {}

def _load(name: str) -> ModuleType:
    mod = _loaded.get(name)
    if mod is None:
        mod = import_module(f".{name}", __name__)
        _loaded[name] = mod
    return mod

def __getattr__(name: str) -> Any:
    # Lazy expose submodules
    if name in _SUBMODULES:
        return _load(name)

    # Lazy function shortcuts
    if name == "compute_acceleration":
        return _load("acceleration").compute_acceleration
    if name == "compute_total_acceleration":
        return _load("acceleration").compute_total_acceleration

    if name == "timedelta_to_seconds":
        return _load("timedelta_to_seconds").timedelta_to_seconds

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> List[str]:
    # For tab-completion: expose modules + shortcuts + dunders
    base = [
        "__doc__", "__loader__", "__name__", "__package__", "__spec__", "__version__",
        "compute_acceleration", "compute_total_acceleration",
        "timedelta_to_seconds",
    ]
    return sorted(set(base + _SUBMODULES))

# Optional: explicitly declare what `from f1analytics import *` exports
__all__ = sorted(set(_SUBMODULES + [
    "compute_acceleration",
    "compute_total_acceleration",

    "timedelta_to_seconds",
]))