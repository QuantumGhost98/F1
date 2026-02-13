"""
Corner metric comparator — unified implementation for min-speed and min-throttle analysis.

Replaces the near-identical CornerMinSpeed and CornerMinThrottle classes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.corner_utils import corner_identifier_to_index, corner_label
from f1analytics.plot_utils import assign_colors_simple, setup_dark_theme, add_branding


class CornerMetricComparator:
    """
    Compare a per-corner metric (min speed, min throttle, etc.) across drivers.
    """

    def __init__(self, session, session_name, year, session_type, drivers, metric='Speed', mode='min', margin=50):
        """
        Parameters
        ----------
        session : FastF1 loaded session
        drivers : dict  {'LEC': 'fastest', 'VER': 14}
        metric  : column name in telemetry DataFrame, e.g. 'Speed', 'Throttle'
        mode    : 'min' or 'max' — how to aggregate across the corner window
        margin  : meters before/after apex
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.metric = metric
        self.mode = mode
        self.margin = margin
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None

        self.laps = session.laps
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()

        if not isinstance(drivers, dict):
            raise ValueError("drivers must be a dict like {'LEC': 'fastest', 'VER': 14}")
        self.driver_map = drivers
        self.drivers = list(drivers.keys())
        self.palette = assign_colors_simple(self.drivers)

        self.telemetry = {}
        self._load_laps()

    def _load_laps(self):
        for d, lap_id in self.driver_map.items():
            if lap_id == 'fastest':
                lap = self.transformed_laps.pick_drivers(d).pick_fastest()
            else:
                try:
                    lap_num = int(lap_id)
                    lap = self.transformed_laps.pick_drivers(d).pick_laps(lap_num).iloc[0]
                except Exception as e:
                    raise ValueError(f"Invalid lap selection for {d}: {lap_id}") from e
            df = lap.get_car_data().add_distance()
            df = interpolate_dataframe(df)
            self.telemetry[d] = df

    def compute(self):
        """Return DataFrame with one row per corner and one column per driver."""
        corners_df = self.circuit_info.corners
        results = {'Corner': []}
        for d in self.drivers:
            results[d] = []

        for apex_idx in range(len(corners_df)):
            apex_dist = corners_df.iloc[apex_idx]['Distance']
            results['Corner'].append(corner_label(self.circuit_info, apex_idx))
            for d in self.drivers:
                df = self.telemetry[d]
                window = df[(df['Distance'] >= apex_dist - self.margin) & (df['Distance'] <= apex_dist + self.margin)]
                if window.empty or self.metric not in window.columns:
                    results[d].append(np.nan)
                elif self.mode == 'min':
                    results[d].append(window[self.metric].min())
                else:
                    results[d].append(window[self.metric].max())

        return pd.DataFrame(results)

    def plot(self, save_path=None):
        """Plot grouped bar chart of per-corner metrics. Returns (fig, ax)."""
        df = self.compute()
        n_corners = len(df)
        n_drivers = len(self.drivers)
        x = np.arange(n_corners)
        bar_w = 0.8 / n_drivers

        fig, ax = plt.subplots(figsize=(max(12, n_corners * 0.8), 6))
        setup_dark_theme(fig, [ax])

        for i, (drv, col) in enumerate(zip(self.drivers, self.palette)):
            ax.bar(x + i * bar_w, df[drv], width=bar_w,
                   label=drv, color=col, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + bar_w * (n_drivers - 1) / 2)
        ax.set_xticklabels(df['Corner'], color='white')

        mode_label = self.mode.capitalize()
        unit = 'km/h' if self.metric == 'Speed' else '%' if self.metric == 'Throttle' else ''
        y_label = f"{mode_label} {self.metric}" + (f" ({unit})" if unit else "")

        ax.set_ylabel(y_label, color='white')
        ax.set_title(f'{self.session_name} {self.year} {self.session_type} — {mode_label} {self.metric} per Corner',
                      color='white', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.tick_params(colors='white')

        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        add_branding(fig, text_pos=(0.95, 0.91), logo_pos=[0.80, 0.91, 0.08, 0.08])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, ax


# ── Backward-compatible aliases ────────────────────────────────────────────────

class CornerMinSpeed(CornerMetricComparator):
    """Backward compat: CornerMinSpeed(session, ..., drivers) → CornerMetricComparator(metric='Speed', mode='min')"""

    def __init__(self, session, session_name, year, session_type, drivers, margin=50):
        super().__init__(
            session, session_name, year, session_type,
            drivers, metric='Speed', mode='min', margin=margin,
        )