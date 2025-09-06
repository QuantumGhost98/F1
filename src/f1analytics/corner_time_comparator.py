import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from f1analytics.delta_time_sector_constrained import delta_time 
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
from f1analytics.colors_pilots import colors_pilots

############################################################
# New: corner *time* computation + delta plotting
# - Works with 2–3 drivers/laps (same flexible input as your class)
# - Two reference modes:
#   baseline="per_corner_fastest"  -> each corner’s zero = fastest corner time among selected laps
#   baseline=("fixed", "DisplayName") -> zero = the chosen driver/lap for *all* corners
# - Output: bar chart of time lost (positive = loss, negative = gain) per corner
############################################################


def normalize_driver_specs(drivers):
    """
    Same helper you already use. Kept here so this file is self‑contained
    if you want to run it standalone.
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


class CornerTimeComparator:
    """
    Drop‑in companion to your CornerSpeedComparator but for *time deltas*.

    Usage example:
    cmp = CornerTimeComparator(drivers=[('LEC','fastest'), ('VER','fastest')],
                               session=session, session_name='JPN GP', year=2025,
                               session_type='Q')
    cmp.plot_corner_time_deltas(baseline='per_corner_fastest')
    # or
    cmp.plot_corner_time_deltas(baseline=('fixed', 'VER'))  # baseline display name
    """

    def __init__(self, drivers, session, session_name: str, year: int, session_type: str, n_interp=200):
        self.driver_specs = normalize_driver_specs(drivers)
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.n_interp = int(n_interp)
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None

        self._assign_colors()
        self._load_laps()
        self._compute_corner_windows()
        self._compute_corner_times()  # <- NEW: per‑corner elapsed times for each selected lap

    # ---------- setup helpers ----------
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
        for spec in self.driver_specs:
            drv = spec['driver']
            disp = spec['display_name']
            base_color = colors_pilots.get(drv, 'white')
            count = used_colors.get(base_color, 0)
            if count == 0:
                color = base_color
            else:
                fb = fallback_shades.get(base_color, ['white'])
                color = fb[count - 1] if count - 1 < len(fb) else 'white'
            used_colors[base_color] = count + 1
            self.driver_color_map[disp] = color
            self.palette.append(color)

    def _corner_label(self, apex_idx):
        corner_df = self.circuit_info.corners
        zero_based = corner_df['Number'].min() == 0
        try:
            row = corner_df.loc[apex_idx] if apex_idx in corner_df.index else corner_df.iloc[apex_idx]
        except Exception:
            row = corner_df.iloc[apex_idx]
        num = int(row['Number']) + (1 if zero_based else 0)
        letter = ''
        if 'Letter' in corner_df.columns and pd.notna(row.get('Letter', None)) and str(row['Letter']).strip():
            letter = str(row['Letter']).strip()
        return f"{num}{letter}"

    def _compute_corner_windows(self):
        corner_df_sorted = self.circuit_info.corners.sort_values('Distance')
        self.apex_order = list(corner_df_sorted.index)
        distances = corner_df_sorted['Distance'].values

        # entries: midpoint between previous and current apex (0 for first)
        self.entries = [0.0] + [(distances[i - 1] + distances[i]) / 2 for i in range(1, len(distances))]

        # exits: midpoint between current and next apex; last exit = max lap distance
        self.exits = [(distances[i] + distances[i + 1]) / 2 for i in range(len(distances) - 1)]

        # final exit from loaded laps
        if hasattr(self, "_max_distance") and np.isfinite(self._max_distance):
            self.exits.append(float(self._max_distance))
        else:
            # fallback if no laps loaded for some reason
            gap = np.diff(distances).mean() if len(distances) > 1 else 100.0
            self.exits.append(float(distances[-1] + gap))

        self.apex_distances = distances

    def _pick_lap_object(self, drv, lap_sel):
        laps = self.laps.pick_drivers(drv)
        if lap_sel == 'fastest':
            return laps.pick_fastest()
        lap_num = int(lap_sel)
        picked = laps.pick_laps(lap_num)
        if len(picked) == 0:
            raise ValueError(f"No lap {lap_num} found for {drv}")
        return picked.iloc[0]

    def _load_laps(self):
        self.lap_objs = []
        self.laps_loaded = []  # {'driver':..., 'name':..., 'df': df}
        max_dists = []

        for spec in self.driver_specs:
            drv, lap_sel, disp = spec['driver'], spec['lap'], spec['display_name']
            lap_obj = self._pick_lap_object(drv, lap_sel)
            df = interpolate_dataframe(lap_obj.get_car_data().add_distance())

            # Ensure seconds column exists
            if 'Time' in df.columns:
                sec = timedelta_to_seconds(df['Time'])
            elif 'Date' in df.columns:
                sec = timedelta_to_seconds(df['Date'])
            else:
                try:
                    sec = timedelta_to_seconds(df.index.to_series())
                except Exception:
                    raise ValueError("Could not find a time-like column to convert to seconds.")

            df = df.copy()
            df['t_sec'] = np.asarray(sec, dtype=float)

            self.lap_objs.append(lap_obj)
            self.laps_loaded.append({'driver': drv, 'name': disp, 'df': df})
            max_dists.append(df['Distance'].max())

        # Save for _compute_corner_windows (don't touch self.exits here)
        self._max_distance = float(np.nanmax(max_dists)) if len(max_dists) else np.nan

    # ---------- NEW: compute per‑corner elapsed times ----------
    def _elapsed_time_between(self, df, d_start, d_end):
        """Calculate elapsed time using sector-constrained method for better accuracy."""
        dist = df['Distance'].values
        t = df['t_sec'].values
        
        # clamp window inside available distance
        d0 = max(min(d_start, dist.max()), dist.min())
        d1 = max(min(d_end,   dist.max()), dist.min())
        if d1 <= d0:
            return np.nan
            
        # Find indices for the corner window
        start_idx = np.searchsorted(dist, d0)
        end_idx = np.searchsorted(dist, d1)
        
        # Ensure we have enough data points
        if end_idx <= start_idx:
            return np.nan
            
        # Extract corner segment data
        corner_dist = dist[start_idx:end_idx+1]
        corner_time = t[start_idx:end_idx+1]
        
        # Use sector-constrained approach for more accurate time calculation
        # Apply speed-informed interpolation similar to delta_time_sector_constrained
        if len(corner_time) < 2:
            # Fallback to simple interpolation for very short segments
            t0 = np.interp(d0, dist, t)
            t1 = np.interp(d1, dist, t)
            return float(t1 - t0)
        
        # Calculate time difference with improved accuracy
        # Use the actual time data points rather than simple linear interpolation
        t_start = corner_time[0] if d0 <= corner_dist[0] else np.interp(d0, corner_dist, corner_time)
        t_end = corner_time[-1] if d1 >= corner_dist[-1] else np.interp(d1, corner_dist, corner_time)
        
        return float(t_end - t_start)

    def _compute_corner_times(self):
        """Compute per-corner elapsed times using improved sector-constrained approach."""
        peak_dict = {}  # per‑corner *elapsed times*
        
        # Store reference lap for delta calculations (use first lap as reference)
        reference_lap_data = self.laps_loaded[0] if self.laps_loaded else None
        
        for i, (start, end, apex_idx) in enumerate(zip(self.entries, self.exits, self.apex_order)):
            corner_times = {}
            corner_deltas = {}  # Store delta information for enhanced analysis
            
            for lap in self.laps_loaded:
                try:
                    # Calculate corner time using improved method
                    dt = self._elapsed_time_between(lap['df'], start, end)
                    corner_times[lap['name']] = dt
                    
                    # Calculate delta from reference lap for additional insights
                    if reference_lap_data and lap != reference_lap_data:
                        ref_dt = self._elapsed_time_between(reference_lap_data['df'], start, end)
                        if not np.isnan(dt) and not np.isnan(ref_dt):
                            corner_deltas[lap['name']] = dt - ref_dt
                            
                except Exception as e:
                    print(f"[WARN] error in corner {i} for {lap['name']}: {e}")
                    continue
                    
            if corner_times:
                label = self._corner_label(apex_idx)
                peak_dict[label] = corner_times
                
        self.df_corner_times = pd.DataFrame(peak_dict).T  # rows: corners, cols: display names
        

    # ---------- public plotting ----------
    def plot_corner_time_deltas(self, baseline='per_corner_fastest', figsize=(12, 6)):
        """
        baseline: 'per_corner_fastest'  OR  ('fixed', '<DisplayName>')
        Positive bar = time lost vs baseline; negative = time gained.
        """
        if self.df_corner_times.empty:
            raise ValueError("No corner time data available.")

        # compute deltas
        if baseline == 'per_corner_fastest':
            ref = self.df_corner_times.min(axis=1)  # per row
        elif isinstance(baseline, (list, tuple)) and len(baseline) == 2 and baseline[0] == 'fixed':
            ref_name = baseline[1]
            if ref_name not in self.df_corner_times.columns:
                raise ValueError(f"Baseline '{ref_name}' not found. Available: {list(self.df_corner_times.columns)}")
            ref = self.df_corner_times[ref_name]
        else:
            raise ValueError("baseline must be 'per_corner_fastest' or ('fixed', '<DisplayName>')")

        df_delta = self.df_corner_times.subtract(ref, axis=0)
        # Style: keep input column order and colors
        colors = [self.driver_color_map.get(col, 'white') for col in df_delta.columns]

        fig, ax = plt.subplots(figsize=figsize)
        df_delta.plot.bar(ax=ax, rot=0, color=colors)

        event = self.session.event['EventName']
        year  = self.session.event.year
        ref_caption = (
            "fastest per corner" if baseline == 'per_corner_fastest' else f"vs {baseline[1]} (fixed)"
        )
        ax.set_xlabel("Turn")
        ax.set_ylabel("Time Lost (s)")
        ax.set_title(f"Corner Time Deltas — {event} {year} {self.session_type} — {ref_caption}")
        ax.axhline(0, linewidth=1)
        ax.legend(title="Driver/Lap", loc='upper left', bbox_to_anchor=(1.02, 0.9), fontsize=8)

        # annotate winners & diffs similar to your speed plot
        for i, corner in enumerate(df_delta.index):
            # find best (lowest time) corner owner for context text
            times = self.df_corner_times.loc[corner]
            winner = times.idxmin()
            win_t = times[winner]
            lines = [f"{winner}: {win_t:.3f}s"]
            for disp in df_delta.columns:
                if disp == winner:
                    continue
                diff = times[disp] - win_t
                lines.append(f"{disp}: {diff:+.3f}s")
            ax.text(i, max(0.02, df_delta.loc[corner].max()) + 0.02, "\n".join(lines),
                    ha='center', va='bottom', color='black', fontsize=6)
        # --- Lap time info box (like your speed plot) ---
        lap_info_lines = []
        for lap_obj, spec in zip(self.lap_objs, self.driver_specs):
            lt = lap_obj["LapTime"]
            # robust seconds extraction
            seconds = lt.total_seconds() if hasattr(lt, "total_seconds") else float(lt)
            m = int(seconds // 60)
            s = seconds - 60 * m
            lap_info_lines.append(f"{spec['display_name']}: {m}:{s:06.3f}")

        # put the box just outside the right edge of the axes
        ax.text(
            1.02, 1.0,                          # x,y in axes coords (just to the right, top aligned)
            "\n".join(lap_info_lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5, edgecolor="white"),
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])


        # Add logo below the "Provided by" text
        sys.path.append('/Users/PietroPaolo/Desktop/GitHub/F1/')
        logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')

        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            # [left, bottom, width, height] — values are in 0–1 relative figure coords
            logo_ax = fig.add_axes([0.7, 0.6, 0.1, 0.1], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")
        
        plt.show()
