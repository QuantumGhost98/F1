"""
Corner minimum throttle analysis.

Restored original API: CornerMinThrottle(session, ..., corner_idxs=[11], before=30, after=30, n_drivers=10)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.corner_utils import corner_identifier_to_index, corner_label
from f1analytics.plot_utils import add_branding, setup_dark_theme


class CornerMinThrottle:
    """
    Analyze minimum throttle across all drivers for a specific corner or group of corners.

    Original API:
        CornerMinThrottle(session, name, year, type, corner_idxs=[11], before=100, after=100, n_drivers=10)
    """

    def __init__(self, session, session_name: str, year: int, session_type: str,
                 corner_idxs, before=100, after=100, n_drivers=None):
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.before = before
        self.after = after
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        self.laps = session.laps
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()

        # Resolve corner identifiers
        if isinstance(corner_idxs, (int, str)):
            self.corner_list = [corner_identifier_to_index(self.circuit_info, corner_idxs)]
        elif hasattr(corner_idxs, '__iter__'):
            self.corner_list = [corner_identifier_to_index(self.circuit_info, x) for x in corner_idxs]
        else:
            raise ValueError("corner_idxs must be int, string like '1A', or iterable thereof")

        if not self.corner_list:
            raise ValueError("No valid corners resolved from corner_idxs")

        # Load driver list
        laps = session.laps.pick_quicklaps()
        self.drivers = laps['Driver'].unique()

        if n_drivers:
            fastest_laps = laps.groupby('Driver')['LapTime'].min().nsmallest(n_drivers)
            self.drivers = fastest_laps.index.tolist()

        self.colors = {d: colors_pilots.get(d, 'white') for d in self.drivers}
        self.min_throttle = {}
        self._compute_min_throttle()

    def _compute_min_throttle(self):
        corners = self.circuit_info.corners['Distance'].values
        dists = [corners[idx] for idx in self.corner_list]
        start = min(dists) - self.before
        end = max(dists) + self.after

        for d in self.drivers:
            try:
                lap = self.session.laps.pick_drivers(d).pick_fastest()
                tel = lap.get_car_data().add_distance()
                tel = interpolate_dataframe(tel)
                df = tel[(tel['Distance'] >= start) & (tel['Distance'] <= end)]
                self.min_throttle[d] = df['Throttle'].min() if not df.empty else None
            except Exception as e:
                logger.warning("Skipping %s: %s", d, e)
                self.min_throttle[d] = None

    def plot(self, save_path=None):
        """Plot min-throttle bar chart. Returns (fig, ax)."""
        vals = {d: v for d, v in self.min_throttle.items() if v is not None}
        if not vals:
            logger.warning("No valid throttle data to plot.")
            return None, None
        sorted_vals = dict(sorted(vals.items(), key=lambda item: item[1]))

        fig, ax = plt.subplots(figsize=(14, 7))
        setup_dark_theme(fig, [ax])

        bars = ax.bar(
            sorted_vals.keys(),
            sorted_vals.values(),
            color=[self.colors[d] for d in sorted_vals.keys()],
            edgecolor='white', linewidth=0.3
        )

        ax.set_ylabel("Minimum Throttle (%)", color='white', fontsize=11)

        # Build corner label string
        if len(self.corner_list) == 1:
            clabel = f"Corner {corner_label(self.circuit_info, self.corner_list[0])}"
        else:
            sorted_idxs = sorted(self.corner_list)
            is_contiguous = all(sorted_idxs[i] + 1 == sorted_idxs[i + 1] for i in range(len(sorted_idxs) - 1))
            if is_contiguous:
                clabel = f"Corners {corner_label(self.circuit_info, sorted_idxs[0])}-{corner_label(self.circuit_info, sorted_idxs[-1])}"
            else:
                clabel = "Corners " + ",".join(corner_label(self.circuit_info, i) for i in sorted_idxs)

        parts = [f"{self.session_name} {self.year}"]
        if self.session_type:
            parts.append(self.session_type)
        parts.append(f"Minimum Throttle — {clabel}")
        ax.set_title(" — ".join(parts), color='white', fontsize=13)
        ax.set_ylim(0, max(sorted_vals.values()) + 10)
        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f"{height:.0f}%", ha='center', va='bottom', color='white', fontsize=9)

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        return fig, ax