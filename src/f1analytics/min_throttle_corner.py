"""
Backward-compatible shim — delegates to CornerMetricComparator.
"""
from f1analytics.min_speed_corner import CornerMetricComparator


class CornerMinThrottle(CornerMetricComparator):
    """Analyze minimum throttle per corner. Thin wrapper around CornerMetricComparator."""

    def __init__(self, session, session_name, year, session_type, drivers, margin=50):
        super().__init__(
            session, session_name, year, session_type,
            drivers, metric='Throttle', mode='min', margin=margin,
        )