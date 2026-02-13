import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.corner_utils import corner_label
from f1analytics.plot_utils import (
    assign_colors_simple, setup_dark_theme, add_branding,
)


class CornerSpeedComparator:
    """
    Compare peak (minimum or maximum) corner speeds across 1-3 drivers/laps.
    """

    def __init__(self, session, session_name, year, session_type, drivers, mode='min'):
        """
        Parameters
        ----------
        session   : loaded FastF1 session
        session_name : e.g. "Hungary Grand Prix"
        year      : e.g. 2025
        session_type : e.g. "Q" or "R"
        drivers   : flexible format accepted by normalize_driver_specs
        mode      : 'min' (apex speed) or 'max' (peak speed)
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.mode = mode
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

    def compute_peak_speeds(self, margin=50):
        """
        Return a DataFrame with one row per corner and one column per driver/display_name,
        plus a 'Corner' label column. Speed metric is min or max depending on self.mode.
        """
        corners_df = self.circuit_info.corners
        results = {'Corner': []}
        for disp in self.display_names:
            results[disp] = []

        for apex_idx in range(len(corners_df)):
            apex_dist = corners_df.iloc[apex_idx]['Distance']
            results['Corner'].append(corner_label(self.circuit_info, apex_idx))

            for disp in self.display_names:
                df = self.telemetry[disp]
                window = df[(df['Distance'] >= apex_dist - margin) & (df['Distance'] <= apex_dist + margin)]
                if window.empty:
                    results[disp].append(np.nan)
                elif self.mode == 'min':
                    results[disp].append(window['Speed'].min())
                else:
                    results[disp].append(window['Speed'].max())

        return pd.DataFrame(results)

    def plot_peak_speeds(self, margin=50, save_path=None):
        """Plot grouped bar chart of corner speeds. Returns (fig, ax)."""
        df = self.compute_peak_speeds(margin)

        n_corners = len(df)
        n_drivers = len(self.display_names)
        x = np.arange(n_corners)
        bar_w = 0.8 / n_drivers

        fig, ax = plt.subplots(figsize=(max(12, n_corners * 0.8), 6))
        setup_dark_theme(fig, [ax])

        for i, (disp, col) in enumerate(zip(self.display_names, self.palette)):
            ax.bar(x + i * bar_w, df[disp], width=bar_w, label=disp, color=col, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + bar_w * (n_drivers - 1) / 2)
        ax.set_xticklabels(df['Corner'], color='white')

        mode_label = "Min" if self.mode == 'min' else "Max"
        ax.set_ylabel(f'{mode_label} Speed (km/h)', color='white')
        ax.set_title(f'{self.session_name} {self.year} {self.session_type} — {mode_label} Corner Speed',
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