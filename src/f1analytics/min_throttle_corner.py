import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os
from f1analytics.interpolate_df import interpolate_dataframe
import sys
from f1analytics.colors_pilots import colors_pilots
import re

class CornerMinThrottle:
    def __init__(self, session, session_name: str, year: int, session_type: str, corner_idxs, before=100, after=100, n_drivers=None):
        """
        Analyze minimum throttle across all drivers for a specific corner or group of corners.

        :param session: FastF1 session object
        :param corner_idxs: int, str like "1A", or list of those (1-based corner identifiers)
        :param before: meters before apex to start extracting
        :param after: meters after apex to end extracting
        :param session_label: optional label to display in plot title
        :param n_drivers: limit to N fastest drivers (optional)
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.before = before
        self.after = after
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        self.laps = session.laps
        # Transform the laps in total seconds
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()

        # Normalize corner identifiers to zero-based indices
        def resolve_single(ci):
            return self._corner_identifier_to_index(ci)

        if isinstance(corner_idxs, (int, str)):
            self.corner_list = [resolve_single(corner_idxs)]
        elif hasattr(corner_idxs, '__iter__'):
            try:
                self.corner_list = [resolve_single(x) for x in corner_idxs]
            except Exception as e:
                raise ValueError(f"Invalid corner_idxs iterable: {e}") from e
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

    def _corner_identifier_to_index(self, label):
        corner_df = self.circuit_info.corners  # fix typo
        zero_based_numbering = corner_df['Number'].min() == 0

        s = str(label).strip()
        m = re.fullmatch(r'(\d+)([A-Za-z]?)', s)
        if not m:
            raise ValueError(f"Invalid corner identifier '{label}' (e.g. '1', '1A')")
        display_num = int(m.group(1))
        letter = m.group(2).upper()

        internal_number = display_num - 1 if zero_based_numbering else display_num
        subset = corner_df[corner_df['Number'] == internal_number]
        if subset.empty:
            raise ValueError(f"Corner number '{display_num}' not found")

        if letter:
            subset = subset[subset['Letter'].astype(str).str.upper() == letter]
            if subset.empty:
                raise ValueError(f"Corner '{label}' not found")

        # return **position** (works with corners.values positional indexing)
        label_idx = subset.index[0]
        pos = int(corner_df.index.get_loc(label_idx))
        return pos

    def _corner_label(self, apex_idx):
        """Return combined Number+Letter label for a given corner index (handles zero-based Number)."""
        corner_df = self.circuit_info.corners
        zero_based = corner_df['Number'].min() == 0
        row = corner_df.iloc[apex_idx]
        num = int(row['Number']) + (1 if zero_based else 0)
        letter = ''
        if 'Letter' in corner_df.columns and pd.notna(row.get('Letter', None)) and str(row['Letter']).strip():
            letter = str(row['Letter']).strip()
        return f"{num}{letter}"

    def _compute_min_throttle(self):
        corners = self.circuit_info.corners['Distance'].values
        # Determine range from selected corners
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
                print(f"[WARN] Skipping {d}: {e}")
                self.min_throttle[d] = None

    def plot(self):
        plt.style.use('dark_background')  # move up

        vals = {d: v for d, v in self.min_throttle.items() if v is not None}
        if not vals:
            print("[INFO] No throttle data to plot for the selected corners.")
            return None, None

        sorted_vals = dict(sorted(vals.items(), key=lambda item: item[1]))
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            list(sorted_vals.keys()),
            list(sorted_vals.values()),
            color=[self.colors[d] for d in sorted_vals.keys()]
        )

        ax.set_ylabel("Minimum Throttle (%)", color='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.tick_params(colors='white')

        # Build corner label string
        if len(self.corner_list) == 1:
            corner_label = f"Corner {self._corner_label(self.corner_list[0])}"
        else:
            sorted_idxs = sorted(self.corner_list)
            is_contiguous = all(sorted_idxs[i] + 1 == sorted_idxs[i + 1] for i in range(len(sorted_idxs) - 1))
            if is_contiguous:
                corner_label = f"Corners {self._corner_label(sorted_idxs[0])}-{self._corner_label(sorted_idxs[-1])}"
            else:
                corner_label = "Corners " + ",".join(self._corner_label(i) for i in sorted_idxs)

        title = f"{self.session.event['EventName']} {self.session.event.year} {self.session_type}\n Minium Throttle - {corner_label}"
        ax.set_title(title)
        ax.set_ylim(0, 105)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                    f"{height:.0f}%", ha='center', va='bottom', color='white')

        fig.text(
            0.3, 0.91,
            "Provided by: Pietro Paolo Melella",
            ha='right', va='bottom',
            color='white', fontsize=15
        )
        plt.tight_layout()
            # Add logo below the "Provided by" text
        sys.path.append('/Users/PietroPaolo/Desktop/GitHub/F1/')
        logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')  # or .jpg etc.

        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            # [left, bottom, width, height] — values are in 0–1 relative figure coords
            logo_ax = fig.add_axes([0.80, 0.895, 0.10, 0.10], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")
        plt.show()