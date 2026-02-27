"""
Fastest-sector delta plotter.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.palette import driver_colors
from f1analytics.config import logger
from f1analytics.plot_utils import add_branding, setup_dark_theme


class SectorDeltaPlotter:
    """
    Calculates and plots fastest sector times & deltas for each driver.
    """

    def __init__(self, session_name, year, session_type, *, session=None):
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps
        self.driver_colors = driver_colors
        self._df = None

    def _compute_table(self):
        """Compute per-driver fastest sector times and deltas to best."""
        sect_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        fastest_laps = []

        transformed_laps = self.laps
        if getattr(transformed_laps, 'empty', False):
            return pd.DataFrame(
                columns=['Driver', *sect_cols, 'Sector1_Delta', 'Sector2_Delta', 'Sector3_Delta']
            )

        for drv in transformed_laps['Driver'].unique():
            drv_laps = transformed_laps.pick_drivers(drv)
            if drv_laps is None or getattr(drv_laps, 'empty', False):
                continue

            lap = drv_laps.pick_fastest()
            if lap is None:
                continue

            if any(c not in lap.index for c in sect_cols):
                continue
            if any(pd.isna(lap[c]) for c in sect_cols):
                continue

            fastest_laps.append({
                'Driver': lap['Driver'],
                'Sector1Time': lap['Sector1Time'],
                'Sector2Time': lap['Sector2Time'],
                'Sector3Time': lap['Sector3Time'],
            })

        df = pd.DataFrame(fastest_laps)

        if df.empty:
            return pd.DataFrame(
                columns=['Driver', *sect_cols, 'Sector1_Delta', 'Sector2_Delta', 'Sector3_Delta']
            )

        for c in sect_cols:
            if pd.api.types.is_timedelta64_dtype(df[c]):
                df[c] = df[c].dt.total_seconds()
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        best = {c: df[c].min(skipna=True) for c in sect_cols}
        for c in sect_cols:
            delta_col = f"{c.split('Time')[0]}_Delta"
            df[delta_col] = (df[c] - best[c]).clip(lower=0)

        return df

    def plot(self, figsize=(18, 10), sharey=False, suptitle=None, save_path=None):
        """Render the three-panel sector delta chart. Returns (fig, axes)."""
        if self._df is None:
            self._df = self._compute_table()
        df = self._df

        # Build three sorted views
        s1 = df[['Driver', 'Sector1Time', 'Sector1_Delta']].sort_values('Sector1_Delta')
        s2 = df[['Driver', 'Sector2Time', 'Sector2_Delta']].sort_values('Sector2_Delta')
        s3 = df[['Driver', 'Sector3Time', 'Sector3_Delta']].sort_values('Sector3_Delta')

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=sharey)
        setup_dark_theme(fig, axes)

        config = [
            ("Sector 1", s1, 'Sector1Time', 'Sector1_Delta'),
            ("Sector 2", s2, 'Sector2Time', 'Sector2_Delta'),
            ("Sector 3", s3, 'Sector3Time', 'Sector3_Delta'),
        ]

        for ax, (title, df_, time_col, delta_col) in zip(axes, config):
            drivers = df_['Driver']
            times = df_[time_col]
            deltas = df_[delta_col]
            cols = [self.driver_colors.get(d, '#888888') for d in drivers]

            bars = ax.barh(drivers, deltas, color=cols)
            for bar, dval, tval in zip(bars, deltas, times):
                txt = f"{tval:.3f}s" if dval == 0 else f"+{dval:.3f}s"
                ax.text(
                    bar.get_width() + (deltas.max() * 0.01 if deltas.max() > 0 else 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    txt,
                    va='center', ha='left', fontsize=9, color='white'
                )
            pad = max(deltas.max(), 0) * 0.15 + 0.05
            ax.set_xlim(0, max(deltas.max(), 0) + pad)
            ax.set_title(title, fontsize=14, color='white')
            ax.invert_yaxis()
            ax.tick_params(left=False, labelleft=True, bottom=False, labelbottom=False, colors='white')
            ax.set_xlabel("")
            ax.grid(False)

        if suptitle is None:
            parts = ["Delta to Fastest Sector Times", f"{self.session_name} {self.year}"]
            if self.session_type:
                parts.append(self.session_type)
            suptitle = " — ".join(parts)
        fig.suptitle(suptitle, fontsize=18, y=0.98, ha='center', color='white')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        return fig, axes

    def get_table(self):
        """Return the computed sector table (times and deltas in seconds)."""
        if self._df is None:
            self._df = self._compute_table()
        return self._df.copy()