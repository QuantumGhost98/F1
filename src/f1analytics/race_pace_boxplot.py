"""
Race-pace boxplot visualizer.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.plot_utils import setup_dark_theme, add_branding


class RacePaceBoxplot:
    """
    Generate boxplots of race-pace laps per driver, with driver-specific colors.
    """

    def __init__(self, session, session_name, year, session_type, drivers=None):
        """
        Parameters
        ----------
        session : FastF1 loaded session
        drivers : None (all), or list of driver codes to include
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps
        self.drivers = drivers

    def plot(self, save_path=None):
        """
        Boxplot of race-pace lap times. Returns (fig, ax).
        """
        laps = self.laps.copy()
        laps.loc[:, 'LapTime_s'] = laps['LapTime'].dt.total_seconds()

        # Filter out pit laps, etc.
        laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()].copy()

        if self.drivers:
            laps = laps[laps['Driver'].isin(self.drivers)]

        # Sort drivers by median pace
        medians = laps.groupby('Driver')['LapTime_s'].median().sort_values()
        sorted_drivers = medians.index.tolist()

        fig, ax = plt.subplots(figsize=(max(14, len(sorted_drivers) * 0.9), 7))
        setup_dark_theme(fig, [ax])

        data = [laps[laps['Driver'] == d]['LapTime_s'].dropna().values for d in sorted_drivers]
        bp = ax.boxplot(data, patch_artist=True, labels=sorted_drivers)

        for i, d in enumerate(sorted_drivers):
            color = colors_pilots.get(d, 'white')
            bp['boxes'][i].set_facecolor(color)
            bp['boxes'][i].set_edgecolor('white')
            bp['medians'][i].set_color('white')

        ax.set_ylabel('Lap Time (s)', color='white', fontsize=11)
        parts = [f"{self.session_name} {self.year}"]
        if self.session_type:
            parts.append(self.session_type)
        parts.append("Race Pace")
        ax.set_title(" — ".join(parts), color='white', fontsize=13)
        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
        ax.tick_params(colors='white')

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, ax

    def get_table(self, save_path=None):
        """Return summary stats DataFrame."""
        laps = self.laps.copy()
        laps.loc[:, 'LapTime_s'] = laps['LapTime'].dt.total_seconds()
        laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]

        if self.drivers:
            laps = laps[laps['Driver'].isin(self.drivers)]

        stats = laps.groupby('Driver')['LapTime_s'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
        stats = stats.sort_values('median')
        stats.columns = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Count']
        return stats