import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.corner_utils import (
    corner_identifier_to_index, corner_label, indices_between,
)
from f1analytics.plot_utils import (
    assign_colors_simple, setup_dark_theme, add_branding,
)


class CornerTimeComparator:
    """
    Compare time deltas across corners between 2-3 drivers/laps.
    """

    def __init__(self, session, session_name, year, session_type, drivers, margin=50):
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.margin = margin
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None

        self.laps = session.laps
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()

        self.driver_specs = normalize_driver_specs(drivers, max_specs=3)
        self.drivers = [s['driver'] for s in self.driver_specs]
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.palette = assign_colors_simple(self.drivers)

        self.telemetry = {}
        self.lap_objs = {}
        self._load_laps()

    def _load_laps(self):
        for spec in self.driver_specs:
            d = spec['driver']
            lap_id = spec['lap']
            disp = spec['display_name']
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
            self.telemetry[disp] = df
            self.lap_objs[disp] = lap

    def _corner_windows(self):
        """Return list of (corner_label, start_dist, end_dist) for each corner."""
        corners_df = self.circuit_info.corners
        windows = []
        for apex_idx in range(len(corners_df)):
            apex_dist = corners_df.iloc[apex_idx]['Distance']
            label = corner_label(self.circuit_info, apex_idx)
            windows.append((label, apex_dist - self.margin, apex_dist + self.margin))
        return windows

    def compute_corner_time_deltas(self):
        """
        Compute time deltas between drivers at each corner.
        Returns a DataFrame with corner labels, and delta columns for each non-baseline driver.
        """
        if len(self.driver_specs) < 2:
            raise ValueError("Need at least 2 drivers for time comparison")

        # Choose fastest lap as baseline
        lap_times = {d: self.lap_objs[d]['LapTime'].total_seconds() for d in self.display_names}
        baseline = min(lap_times, key=lap_times.get)

        ref_lap = self.lap_objs[baseline]
        windows = self._corner_windows()

        results = {'Corner': [w[0] for w in windows]}

        for disp in self.display_names:
            if disp == baseline:
                continue
            comp_lap = self.lap_objs[disp]
            try:
                delta_series, ref_tel, comp_tel = delta_time(ref_lap, comp_lap)
            except Exception as e:
                logger.warning("Delta computation failed for %s vs %s: %s", disp, baseline, e)
                results[f"Δ({disp})"] = [np.nan] * len(windows)
                continue

            deltas = []
            for (label, start_d, end_d) in windows:
                mask = (ref_tel['Distance'] >= start_d) & (ref_tel['Distance'] <= end_d)
                if mask.any():
                    deltas.append(delta_series[mask].mean())
                else:
                    deltas.append(np.nan)
            results[f"Δ({disp})"] = deltas

        return pd.DataFrame(results), baseline

    def plot_corner_time_deltas(self, save_path=None):
        """Plot grouped bar chart of corner time deltas. Returns (fig, ax)."""
        df, baseline = self.compute_corner_time_deltas()
        delta_cols = [c for c in df.columns if c.startswith('Δ')]

        if not delta_cols:
            raise ValueError("No comparison drivers found")

        n_corners = len(df)
        n_comparisons = len(delta_cols)
        x = np.arange(n_corners)
        bar_w = 0.8 / n_comparisons

        fig, ax = plt.subplots(figsize=(max(12, n_corners * 0.8), 6))
        setup_dark_theme(fig, [ax])

        comp_colors = [self.palette[i] for i, d in enumerate(self.display_names) if d != baseline]
        for i, (col_name, col_color) in enumerate(zip(delta_cols, comp_colors)):
            ax.bar(x + i * bar_w, df[col_name], width=bar_w,
                   label=col_name, color=col_color, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + bar_w * (n_comparisons - 1) / 2)
        ax.set_xticklabels(df['Corner'], color='white')
        ax.axhline(0, color='white', linestyle='--', linewidth=0.8)

        ax.set_ylabel('Δ Time (s)', color='white')
        ax.set_title(f'{self.session_name} {self.year} {self.session_type} — Corner Time Delta (vs {baseline})',
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
