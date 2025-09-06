import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
import warnings
from scipy.signal import savgol_filter
import sys
from f1analytics.colors_pilots import colors_pilots


def normalize_driver_specs(drivers):
    """
    Normalize into a list of dicts: [{'driver':..., 'lap':..., 'display_name':...}]
    - drivers can be:
        dict: {"LEC": "fastest"} or {"LEC": ["fastest", 14], "VER": 7}
        list/tuple:
          ["LEC", "VER"]
          [("LEC","fastest"), ("VER",14)]
          [{"LEC":"fastest"}, {"VER":14}]
    - lap can be 'fastest' or a lap number (int)
    - returns 1..3 specs; raises if outside that range
    """
    driver_specs = []
    if isinstance(drivers, dict):
        for drv, lap_sel in drivers.items():
            if isinstance(lap_sel, (list, tuple)):
                for sel in lap_sel:
                    name = drv if sel == 'fastest' else f"{drv}_{sel}"
                    driver_specs.append({'driver': drv, 'lap': sel, 'display_name': name})
            else:
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
    elif isinstance(drivers, (list, tuple)):
        for entry in drivers:
            if isinstance(entry, str):
                driver_specs.append({'driver': entry, 'lap': 'fastest', 'display_name': entry})
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                drv, lap_sel = entry
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(f"Invalid driver dict entry: {entry}")
                drv, lap_sel = next(iter(entry.items()))
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
            else:
                raise ValueError(f"Unsupported driver entry: {entry}")
    else:
        raise ValueError("drivers must be dict, list, or tuple of supported specs.")

    if not (1 <= len(driver_specs) <= 3):
        raise ValueError("Must compare between 1 and 3 laps/drivers.")

    return driver_specs


class CornerSpeedComparator:
    def __init__(self, drivers, session, session_name: str, year: int, session_type: str, n_interp=200):
        """
        :param drivers: flexible selector, e.g.
                        ['LEC','VER']
                        {'LEC': 'fastest', 'VER': 14}
                        [('LEC','fastest'), ('VER', 14)]
        :param n_interp: number of points to interpolate in each corner
        :param session_name: string to include in plot title
        """
        # Normalize incoming driver specs (driver, lap, display_name)
        self.driver_specs = normalize_driver_specs(drivers)
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.n_interp = n_interp
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None

        # Transform the laps in total seconds
        self.transformed_laps = self.laps.copy()

        self._assign_colors()
        self._load_laps()
        self._compute_corner_windows()
        self._compute_peaks()

    def _assign_colors(self):
        fallback_shades = {
            'red': ['white', 'lightcoral'],
            'blue': ['cyan', 'lightblue'],
            'orange': ['gold', 'wheat'],
            'grey': ['white', 'silver'],
            'green': ['lime', 'springgreen'],
            'pink': ['violet', 'lightpink'],
            'olive': ['khaki'],
            'navy': ['skyblue'],
            '#9932CC': ['plum'],
            'lime': ['yellowgreen']
        }

        used_colors = {}
        self.driver_color_map = {}
        self.palette = []

        # Color is keyed by display name (so two laps of same driver get different shades)
        for spec in self.driver_specs:
            drv = spec['driver']
            disp = spec['display_name']
            base_color = colors_pilots.get(drv, 'white')
            count = used_colors.get(base_color, 0)
            if count == 0:
                color = base_color
            else:
                fallback = fallback_shades.get(base_color, ['white'])
                color = fallback[count - 1] if count - 1 < len(fallback) else 'white'
            used_colors[base_color] = count + 1
            self.driver_color_map[disp] = color
            self.palette.append(color)

    def _corner_label(self, apex_idx):
        """Return combined Number+Letter label for a given corner index (handles zero-based Number)."""
        corner_df = self.circuit_info.corners
        zero_based = corner_df['Number'].min() == 0
        # support both label-based and positional index
        try:
            if apex_idx in corner_df.index:
                row = corner_df.loc[apex_idx]
            else:
                row = corner_df.iloc[apex_idx]
        except Exception:
            row = corner_df.iloc[apex_idx]
        num = int(row['Number']) + (1 if zero_based else 0)
        letter = ''
        if 'Letter' in corner_df.columns and pd.notna(row.get('Letter', None)) and str(row['Letter']).strip():
            letter = str(row['Letter']).strip()
        return f"{num}{letter}"

    def _compute_corner_windows(self):
        # preserve original indices to get labels later
        corner_df_sorted = self.circuit_info.corners.sort_values('Distance')
        self.apex_order = list(corner_df_sorted.index)  # original indices
        distances = corner_df_sorted['Distance'].values
        self.entries = [0.0] + [(distances[i - 1] + distances[i]) / 2 for i in range(1, len(distances))]
        self.exits = [(distances[i] + distances[i + 1]) / 2 for i in range(len(distances) - 1)] + [
            max(lap['df']['Distance'].max() for lap in self.laps_loaded)
        ]
        self.apex_distances = distances

    def _pick_lap_object(self, drv, lap_sel):
        laps = self.transformed_laps.pick_drivers(drv)
        if lap_sel == 'fastest':
            return laps.pick_fastest()
        # assume lap number (int-like)
        try:
            lap_num = int(lap_sel)
        except Exception:
            raise ValueError(f"Unsupported lap selector for {drv}: {lap_sel}")
        picked = laps.pick_laps(lap_num)
        # FastF1 returns a Laps obj; get the single Lap
        if len(picked) == 0:
            raise ValueError(f"No lap {lap_num} found for {drv}")
        return picked.iloc[0]

    def _load_laps(self):
        self.lap_objs = []
        self.laps_loaded = []
        for spec in self.driver_specs:
            drv = spec['driver']
            lap_sel = spec['lap']
            disp = spec['display_name']

            lap_obj = self._pick_lap_object(drv, lap_sel)
            df = interpolate_dataframe(lap_obj.get_car_data().add_distance())

            self.lap_objs.append(lap_obj)
            self.laps_loaded.append({'driver': drv, 'name': disp, 'df': df})

    def _compute_peaks(self):
        peak_dict = {}
        for i, (start, apex_dist, end, apex_idx) in enumerate(
                zip(self.entries, self.apex_distances, self.exits, self.apex_order)):

            # Create a focused window around the actual corner apex
            apex_window = min(100, (end - start) * 0.3)  # 100m or 30% of corner window
            apex_start = apex_dist - apex_window / 2
            apex_end = apex_dist + apex_window / 2

            grid = np.linspace(start, end, self.n_interp)
            apex_grid = np.linspace(apex_start, apex_end, max(20, self.n_interp // 10))

            corner_speeds = {}
            for lap in self.laps_loaded:
                try:
                    distance = lap['df']['Distance'].values
                    speed = lap['df']['Speed'].values

                    # Remove NaN values
                    valid_mask = ~(np.isnan(distance) | np.isnan(speed))
                    if valid_mask.sum() < 3:
                        continue

                    distance_clean = distance[valid_mask]
                    speed_clean = speed[valid_mask]

                    # Interpolate speed over the full corner window
                    full_speeds = np.interp(grid, distance_clean, speed_clean)

                    # Interpolate speed specifically around the apex
                    apex_speeds = np.interp(apex_grid, distance_clean, speed_clean)

                    # Find the actual corner speed (minimum in apex region)
                    min_apex_speed = np.min(apex_speeds)

                    # Also check for realistic corner speed vs straight speed
                    max_full_speed = np.max(full_speeds)

                    # If the apex speed is too close to max speed, it's probably a fast corner
                    if min_apex_speed / max_full_speed > 0.8:
                        corner_speed = min_apex_speed
                    else:
                        corner_speed = min_apex_speed

                    # Key by display name so duplicate drivers show separately
                    corner_speeds[lap['name']] = corner_speed

                except Exception as e:
                    print(f"Warning: Error processing {lap['name']} in corner {i}: {e}")
                    continue

            if corner_speeds:
                label = self._corner_label(apex_idx)
                peak_dict[label] = corner_speeds

        if peak_dict:
            self.df_peaks = pd.DataFrame(peak_dict).T
        else:
            print("Warning: No valid corner speed data computed")
            self.df_peaks = pd.DataFrame()

    def plot_peak_speeds(self, figsize=(12,6)):
        fig, ax = plt.subplots(figsize=figsize)
        # Ensure color order matches column order
        colors = [self.driver_color_map.get(col, 'white') for col in self.df_peaks.columns]
        self.df_peaks.plot.bar(ax=ax, rot=0, color=colors)

        event = self.session.event['EventName']
        year  = self.session.event.year
        ax.set_xlabel("Turn")
        ax.set_ylabel("Corner Speed (km/h)")
        ax.set_title(f"Turn Corner Speeds {event} {year} {self.session_type}")
        ax.legend(
            title="Driver/Lap",
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10
        )

        for i, corner in enumerate(self.df_peaks.index):
            speeds = self.df_peaks.loc[corner]
            winner = speeds.idxmax()
            win_speed = speeds[winner]
            lines = [f"{winner}: {win_speed:.0f}"]
            for disp in self.df_peaks.columns:
                if disp == winner:
                    continue
                diff = int(speeds[disp] - win_speed)
                lines.append(f"{disp}: {diff:+d}")
            text = "\n".join(lines)
            ax.text(i, win_speed + 3, text, ha='center', va='bottom',
                    color='white', fontsize=6)
        
        # Collect lap time info
        lap_info_lines = []
        for lap_obj, spec in zip(self.lap_objs, self.driver_specs):
            lap_time_str = lap_obj['LapTime'].total_seconds()
            lap_time_fmt = f"{lap_time_str // 60:.0f}:{(lap_time_str % 60):06.3f}"  # mm:ss.sss format
            line = f"{spec['display_name']}: {lap_time_fmt}"
            lap_info_lines.append(line)

        # Add text box to figure
        props = dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='white')
        fig.text(
            1.1, 1.8,
            "\n".join(lap_info_lines),
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment='top',
            color='white',
            bbox=props
        )

        fig.text(
            0.3, 1.75,
            "Provided by: Pietro Paolo Melella",
            ha='right', va='bottom',
            color='white', fontsize=10
        )

        plt.tight_layout(rect=[0, 0, 1.3, 1.8])

        # Add logo below the "Provided by" text
        # Insert logo at specified coordinates (repo-relative path)
        repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(repo_path)
        logo_path = os.path.join(repo_path, "logo-square.png")
        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            # [left, bottom, width, height] — values in 0–1 figure coords (keep within [0,1])
            logo_ax = fig.add_axes([1.15, 1.4, 0.12, 0.12], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")
        plt.show()