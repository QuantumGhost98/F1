# sector_deltas.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys
from f1analytics.colors_pilots import colors_pilots
from typing import Optional


class SectorDeltaPlotter:
    """
    Compute each driver's fastest sector times and deltas to the overall best,
    then render three horizontal bar charts (S1/S2/S3).
    """

    def __init__(self, session, session_name: str, year: int, session_type: str):
        """
        session: FastF1 session (already loaded)
        laps_df: usually session.laps (or your transformed_laps)
        colors: optional dict of {driver_code: color}; defaults to colors_pilots
        session_type: "Q", "R", etc. (used in title)
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps
        # Transform the laps in total seconds
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()
        self.colors_pilots = colors_pilots
        self._df = None  # computed table of fastest sectors + deltas

    def _compute_table(self):
        fastest_laps = []
        sect_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']

        # Be robust to empty frames
        if getattr(self.transformed_laps, 'empty', False):
            return pd.DataFrame(
                columns=['Driver', *sect_cols, 'Sector1_Delta', 'Sector2_Delta', 'Sector3_Delta']
            )

        for drv in self.transformed_laps['Driver'].unique():
            drv_laps = self.transformed_laps.pick_drivers(drv)
            if drv_laps is None or getattr(drv_laps, 'empty', False):
                continue

            lap = drv_laps.pick_fastest()
            # Some drivers may have no valid fastest lap -> None
            if lap is None:
                continue

            # Ensure the three sector columns exist and are not NaT/NaN
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

        # If nothing valid collected, return an empty frame with expected columns
        if df.empty:
            return pd.DataFrame(
                columns=['Driver', *sect_cols, 'Sector1_Delta', 'Sector2_Delta', 'Sector3_Delta']
            )

        # Ensure sector times are numeric (seconds). If they are timedeltas, convert.
        for c in sect_cols:
            if pd.api.types.is_timedelta64_dtype(df[c]):
                df[c] = df[c].dt.total_seconds()
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Compute best per sector and deltas to best
        best = {c: df[c].min(skipna=True) for c in sect_cols}
        for c in sect_cols:
            delta_col = f"{c.split('Time')[0]}_Delta"  # e.g., 'Sector1_Delta'
            df[delta_col] = (df[c] - best[c]).clip(lower=0)

        return df

    def plot(self, figsize=(18, 10), sharey=False, suptitle: Optional[str] =None):
        """Render the three-panel sector delta chart."""
        if self._df is None:
            self._df = self._compute_table()
        df = self._df

        # build three sorted views
        s1 = df[['Driver','Sector1Time','Sector1_Delta']].sort_values('Sector1_Delta')
        s2 = df[['Driver','Sector2Time','Sector2_Delta']].sort_values('Sector2_Delta')
        s3 = df[['Driver','Sector3Time','Sector3_Delta']].sort_values('Sector3_Delta')

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=sharey)
        config = [
            ("Sector 1", s1, 'Sector1Time', 'Sector1_Delta'),
            ("Sector 2", s2, 'Sector2Time', 'Sector2_Delta'),
            ("Sector 3", s3, 'Sector3Time', 'Sector3_Delta'),
        ]

        for ax, (title, df_, time_col, delta_col) in zip(axes, config):
            drivers = df_['Driver']
            times = df_[time_col]
            deltas = df_[delta_col]
            cols = [self.colors_pilots.get(d, 'black') for d in drivers]

            bars = ax.barh(drivers, deltas, color=cols)
            for bar, dval, tval in zip(bars, deltas, times):
                txt = f"{tval:.3f}s" if dval == 0 else f"+{dval:.3f}s"
                ax.text(
                    bar.get_width() + (deltas.max()*0.01 if deltas.max() > 0 else 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    txt,
                    va='center', ha='left', fontsize=9, color='white'
                )
            pad = max(deltas.max(), 0) * 0.15 + 0.05
            ax.set_xlim(0, max(deltas.max(), 0) + pad)
            ax.set_title(title, fontsize=14)
            ax.invert_yaxis()
            ax.tick_params(left=False, labelleft=True, bottom=False, labelbottom=False)
            ax.set_xlabel("")
            ax.grid(False)

        # title and credits
        if suptitle is None:
            suptitle = f"Delta to Fastest Sector Times - {self.session.event['EventName']} {self.session.event.year} - {self.session_type}"
        fig.suptitle(suptitle, fontsize=18, y=0.98, ha='center', color='white')
        fig.text(0.9, 0.90, "Provided by:\nPietro Paolo Melella", ha='right', fontsize=13, color='white')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        # optional logo
        sys.path.append('/Users/PietroPaolo/Desktop/GitHub/F1/')
        logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')  # or .jpg etc.
        if logo_path and os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            logo_ax = fig.add_axes([0.73, 0.895, 0.08, 0.08], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')

        plt.show()

    def table(self) -> pd.DataFrame:
        """Return the computed sector table (times and deltas in seconds)."""
        if self._df is None:
            self._df = self._compute_table()
        return self._df.copy()