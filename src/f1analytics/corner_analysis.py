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
import re

class CornerAnalysis:
    """
    Analyze telemetry for one or more corners (or a corner range).
    Supports flexible channel selection (e.g., include/exclude 'Acc') and optional delta-time
    (default when comparing multiple drivers unless suppressed; can be triggered explicitly
    via 'Delta', 'DeltaTime', or 'Δ').
    corner_idxs now also supports ranges like "3-5" and nested lists/tuples like [3,4] or (7, 9) which are expanded inclusively.
    """

    def __init__(self, session, session_name: str, year: int, session_type: str, drivers, corner_idxs, before=50, after=50):
        if not isinstance(drivers, dict):
            raise ValueError("drivers must be a dict of form {'LEC': 'fastest' or lap_number_str}")

        if not (1 <= len(drivers) <= 4):
            raise ValueError("drivers dict must contain between 1 and 4 entries")
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.driver_lap_map = drivers
        self.drivers = list(drivers.keys())
        self.before = before
        self.after = after
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        self.laps = session.laps
        # Transform the laps in total seconds
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()

        # Normalize user-provided corner_idxs. Accepts:
        # - single int or string label (e.g., 3, "1A")
        # - iterable of items which can be:
        #     * ints/strings
        #     * two-length iterables (lists/tuples) indicating inclusive ranges, e.g., [3,4] or ("3", "4B")
        #     * strings with a hyphen indicating an inclusive range, e.g., "3-5" or "1A-2B"
        def resolve_single(ci):
            return self._corner_identifier_to_index(ci)

        def expand_item(item):
            """Return a list of corner *indices* expanded from the given item."""
            # Range encoded as string with '-': "3-5" or "1A-2B"
            if isinstance(item, str) and '-' in item:
                left, right = [p.strip() for p in item.split('-', 1)]
                a = resolve_single(left)
                b = resolve_single(right)
                return self._indices_between(a, b)

            # Simple scalar (int/str corner id)
            if isinstance(item, (int, str)):
                return [resolve_single(item)]

            # Two-length iterable representing inclusive range, e.g., [3,4] or ("3","4B")
            if hasattr(item, '__iter__') and not isinstance(item, (bytes, bytearray)):
                try:
                    it = list(item)
                except Exception:
                    raise ValueError(f"Invalid corner item: {item}")
                if len(it) == 2 and all(isinstance(x, (int, str)) for x in it):
                    a = resolve_single(it[0])
                    b = resolve_single(it[1])
                    return self._indices_between(a, b)
                # Nested iterables: flatten recursively
                expanded = []
                for sub in it:
                    expanded.extend(expand_item(sub))
                return expanded

            raise ValueError(f"Invalid corner item: {item}")

        # Build full list, de-duplicate while preserving order
        if isinstance(corner_idxs, (int, str)):
            raw_indices = expand_item(corner_idxs)
        elif hasattr(corner_idxs, '__iter__'):
            raw_indices = []
            for it in corner_idxs:
                raw_indices.extend(expand_item(it))
        else:
            raise ValueError("corner_idxs must be an int, string like '1A', or iterable thereof")

        # Preserve order and uniqueness
        seen = set()
        self.corner_list = []
        for idx in raw_indices:
            if idx not in seen:
                seen.add(idx)
                self.corner_list.append(idx)

        if not self.corner_list:
            raise ValueError("No valid corners resolved from corner_idxs")

        self.start_idx = min(self.corner_list)
        self.end_idx = max(self.corner_list)

        self.telemetry = {}
        self.lap_objs = {}
        self._load_data()
        self.palette = self._assign_colors()

    def _corner_identifier_to_index(self, label):
        """
        Resolve a corner identifier like 3, "3", "1A", "12B" to the
        DataFrame index in self.circuit_info.corners.
        """
        corner_df = self.circuit_info.corners
        zero_based_numbering = corner_df['Number'].min() == 0

        # Parse input like '12A'
        s = str(label).strip()
        m = re.fullmatch(r'(\d+)([A-Za-z]?)', s)
        if not m:
            raise ValueError(f"Invalid corner identifier '{label}' (expected digits with optional letter)")

        display_num = int(m.group(1))
        letter = m.group(2).upper()

        # Adjust for zero-based corner numbering if needed
        internal_number = display_num - 1 if zero_based_numbering else display_num

        # Filter by number
        subset = corner_df[corner_df['Number'] == internal_number]
        if subset.empty:
            raise ValueError(f"Corner number '{display_num}' not found in circuit_info.corners")

        # Filter by letter if given
        if letter:
            subset = subset[subset['Letter'].astype(str).str.upper() == letter]
            if subset.empty:
                raise ValueError(f"Corner '{label}' not found (number {display_num} with letter '{letter}')")
        else:
            # Prefer no-letter version if multiple rows for same number
            no_letter_mask = subset['Letter'].isna() | (subset['Letter'].astype(str).str.strip() == '')
            subset = subset[no_letter_mask] if not subset[no_letter_mask].empty else subset

        # Return the *DataFrame index* (this is the position used by circuit_info.corners['Distance'])
        return subset.index[0]

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

    def _indices_between(self, a_idx, b_idx):
        """Inclusive list of DataFrame indices between two apex indices (order-agnostic)."""
        start, end = sorted((int(a_idx), int(b_idx)))
        return list(range(start, end + 1))

    def _compress_indices_to_ranges(self, idx_list):
        """Compress a sorted list of integer indices into list of (start,end) tuples for contiguous runs."""
        if not idx_list:
            return []
        sorted_idxs = sorted(idx_list)
        ranges = []
        run_start = prev = sorted_idxs[0]
        for x in sorted_idxs[1:]:
            if x == prev + 1:
                prev = x
                continue
            ranges.append((run_start, prev))
            run_start = prev = x
        ranges.append((run_start, prev))
        return ranges

    def _format_corner_label_list(self, idx_list):
        """Return a compact label like '1,3-4,6' from a list of corner apex indices."""
        parts = []
        for a, b in self._compress_indices_to_ranges(idx_list):
            if a == b:
                parts.append(self._corner_label(a))
            else:
                parts.append(f"{self._corner_label(a)}-{self._corner_label(b)}")
        return ",".join(parts)

    def _load_data(self):
        """Load and interpolate telemetry for each driver based on lap selection, storing both lap and telemetry."""
        for d, lap_id in self.driver_lap_map.items():
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
            self.telemetry[d] = df
            self.lap_objs[d] = lap

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
        palette = []

        for driver in self.drivers:
            base_color = colors_pilots.get(driver, 'white')
            count = used_colors.get(base_color, 0)
            if count == 0:
                palette.append(base_color)
            else:
                fallback = fallback_shades.get(base_color, ['white'])
                alt_color = fallback[count - 1] if count - 1 < len(fallback) else 'white'
                palette.append(alt_color)
            used_colors[base_color] = count + 1

        return palette

    def get_corner_df(self, driver):
        df = self.telemetry[driver]
        corners = self.circuit_info.corners['Distance'].values
        start_dist = corners[self.start_idx] - self.before
        end_dist = corners[self.end_idx] + self.after
        dfc = df[(df['Distance'] >= start_dist) & (df['Distance'] <= end_dist)].copy()
        dfc['Speed_ms'] = dfc['Speed'] / 3.6
        dfc['Sess_s'] = dfc['SessionTime'].dt.total_seconds()
        # compute acceleration in time domain
        dfc['Acc'] = np.gradient(dfc['Speed_ms'], dfc['Sess_s'])
        return dfc

    def plot_all(self, channels=None):
        """
        Plot requested channels around the corner(s). Acceptable channel names include
    'Speed', 'Throttle', 'Brake', and delta aliases ('Delta', 'DeltaTime', 'Δ').
    If multiple drivers are compared, delta-time is shown by default unless channels
    are provided without a delta alias.
        """
        default_channels = ['Speed', 'Throttle', 'Brake']
        user_provided_channels = channels is not None
        channels = channels or default_channels

        # Delta-time logic
        delta_aliases = {'delta', 'deltatime', 'Δ'}
        wants_delta_token = any(str(ch).lower() in delta_aliases for ch in channels)
        channels = [ch for ch in channels if str(ch).lower() not in delta_aliases]

        # Only show delta if there are 2+ drivers
        wants_delta = (len(self.drivers) > 1) and (wants_delta_token or not user_provided_channels)

        # Do we also want the throttle scatter alt view?
        include_throttle_scatter = any(str(ch).lower() == 'throttle' for ch in channels)

        n_line = len(channels)
        n_extra = (1 if wants_delta else 0) + (1 if include_throttle_scatter else 0)
        total_plots = max(1, n_line + n_extra)

        fig, axs = plt.subplots(total_plots, 1, figsize=(10, 3 * total_plots), sharex=False)

        # normalize axs into a list
        if isinstance(axs, np.ndarray):
            axes_list = axs.ravel().tolist()
        else:
            axes_list = [axs]

        # dark style
        plt.style.use('dark_background')
        fig.patch.set_facecolor('black')
        for ax in axes_list:
            ax.set_facecolor('black')

        # Build title text for single or multiple corners (compact: e.g., 1,3-4,6)
        if len(self.corner_list) == 1:
            corner_label = f"Corner {self._corner_label(self.corner_list[0])}"
        else:
            corner_label = f"Corners {self._format_corner_label_list(self.corner_list)}"

        title = f"{self.session_name} {self.year} {self.session_type} {corner_label}"
        fig.suptitle(title, color='white')
        fig.subplots_adjust(top=0.92)

        plot_idx = 0

        # Line plots for requested channels
        for ch in channels:
            ax = axes_list[plot_idx]
            for d, col in zip(self.drivers, self.palette):
                dfc = self.get_corner_df(d)
                if ch not in dfc.columns:
                    continue
                ax.plot(dfc['Distance'], dfc[ch], color=col, label=f"{d} {ch}")
            ax.set_ylabel(ch, color='white')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.tick_params(colors='white')
            plot_idx += 1

        # Δ time subplot (only if 2+ drivers)
        if wants_delta:
            ax_dt = axes_list[plot_idx]
            # choose baseline as the fastest lap
            lap_times = [self.lap_objs[d]['LapTime'].total_seconds() for d in self.drivers]
            baseline_idx = lap_times.index(min(lap_times))
            baseline_driver = self.drivers[baseline_idx]
            ref_lap = self.lap_objs[baseline_driver]

            corners = self.circuit_info.corners['Distance'].values
            start_dist = corners[self.start_idx] - self.before
            end_dist = corners[self.end_idx] + self.after

            for comp_driver, col in zip(self.drivers, self.palette):
                if comp_driver == baseline_driver:
                    continue
                comp_lap = self.lap_objs[comp_driver]
                delta_series, ref_tel, comp_tel = delta_time(ref_lap, comp_lap)
                mask = (ref_tel['Distance'] >= start_dist) & (ref_tel['Distance'] <= end_dist)
                ax_dt.plot(ref_tel['Distance'][mask], delta_series[mask],
                        color=col, linestyle='-', label=f"Δ ({comp_driver} - {baseline_driver})")

            ax_dt.set_ylabel('Δ Time (s)', color='white')
            ax_dt.axhline(0, color='white', linestyle='--')
            ax_dt.grid(True, linestyle='--', linewidth=0.5)
            ax_dt.tick_params(colors='white')
            ax_dt.legend(loc='upper right', title=f"Benchmark: {baseline_driver}")
            plot_idx += 1
        else:
            # if user explicitly asked for delta with only one driver, gently note it
            if wants_delta_token and len(self.drivers) == 1:
                print("[INFO] Delta requested but only one driver provided; skipping Δ plot.")

        # Throttle scatter (optional)
        if include_throttle_scatter:
            ax_throttle_alt = axes_list[plot_idx]
            for d, col in zip(self.drivers, self.palette):
                dfc = self.get_corner_df(d)
                if dfc.empty or 'Distance' not in dfc or 'Throttle' not in dfc:
                    continue
                ax_throttle_alt.scatter(dfc['Distance'].to_numpy(),
                                        dfc['Throttle'].to_numpy(),
                                        s=10, color=col, alpha=0.6, label=d)
            ax_throttle_alt.set_xlabel('Distance (m)', color='white')
            ax_throttle_alt.set_ylabel('Throttle %', color='white')
            ax_throttle_alt.set_title('Throttle (Scatter View)', color='white')
            ax_throttle_alt.legend(loc='upper right')
            ax_throttle_alt.grid(True, linestyle='--', linewidth=0.5)
            ax_throttle_alt.tick_params(colors='white')
            plot_idx += 1

        # Put x-label on the last axis if not already done
        if axes_list:
            axes_list[-1].set_xlabel('Distance (m)', color='white')

        # Signature & logo (layout first, then logo to avoid tight_layout warning)
        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        fig.text(0.95, 0.91, "Provided by: Pietro Paolo Melella",
                ha='right', va='bottom', color='white', fontsize=15)

        sys.path.append('/Users/PietroPaolo/Desktop/GitHub/F1/')
        logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')
        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            logo_ax = fig.add_axes([0.80, 0.91, 0.08, 0.08], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")

        plt.show()