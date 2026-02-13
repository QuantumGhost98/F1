"""
Fastest-sector delta plotter.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.plot_utils import setup_dark_theme, add_branding


class SectorDeltaPlotter:
    """
    Calculates and plots fastest sector times & deltas for each driver.
    """

    def __init__(self, session, session_name, year, session_type):
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps

    def compute_sector_data(self, top_n=10):
        """
        Returns a DataFrame with fastest sector times per driver and delta to best.
        """
        laps = self.laps.copy()
        laps.loc[:, 'S1'] = laps['Sector1Time'].dt.total_seconds()
        laps.loc[:, 'S2'] = laps['Sector2Time'].dt.total_seconds()
        laps.loc[:, 'S3'] = laps['Sector3Time'].dt.total_seconds()

        grouped = laps.groupby('Driver').agg(
            S1=('S1', 'min'),
            S2=('S2', 'min'),
            S3=('S3', 'min'),
        ).reset_index()
        grouped['Theoretical'] = grouped['S1'] + grouped['S2'] + grouped['S3']
        grouped = grouped.sort_values('Theoretical').head(top_n).reset_index(drop=True)

        best_s1, best_s2, best_s3 = grouped['S1'].min(), grouped['S2'].min(), grouped['S3'].min()
        grouped['ΔS1'] = grouped['S1'] - best_s1
        grouped['ΔS2'] = grouped['S2'] - best_s2
        grouped['ΔS3'] = grouped['S3'] - best_s3
        grouped['ΔTotal'] = grouped['Theoretical'] - grouped['Theoretical'].min()

        return grouped

    def plot(self, top_n=10, save_path=None):
        """
        Stacked-bar chart of sector deltas. Returns (fig, ax).
        """
        df = self.compute_sector_data(top_n)
        n_drivers = len(df)
        x = np.arange(n_drivers)

        fig, ax = plt.subplots(figsize=(max(10, n_drivers * 0.9), 6))
        setup_dark_theme(fig, [ax])

        s1_bars = ax.bar(x, df['ΔS1'], label='Δ S1', color='#e74c3c', edgecolor='white', linewidth=0.5)
        s2_bars = ax.bar(x, df['ΔS2'], bottom=df['ΔS1'], label='Δ S2', color='#3498db', edgecolor='white', linewidth=0.5)
        s3_bars = ax.bar(x, df['ΔS3'], bottom=df['ΔS1'] + df['ΔS2'], label='Δ S3', color='#2ecc71', edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(df['Driver'], color='white')
        ax.set_ylabel('Δ Time (s)', color='white')
        ax.set_title(f'{self.session_name} {self.year} {self.session_type} — Fastest Sector Deltas',
                      color='white', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.tick_params(colors='white')

        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        add_branding(fig, text_pos=(0.95, 0.91), logo_pos=[0.80, 0.91, 0.08, 0.08])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, ax

    def get_table(self, top_n=10):
        """Return the sector data as a formatted table."""
        return self.compute_sector_data(top_n)