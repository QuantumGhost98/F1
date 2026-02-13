"""
Corner time comparator — analyze time deltas per corner.

Restored original API: CornerTimeComparator(drivers, session, ...)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.corner_utils import corner_label
from f1analytics.plot_utils import add_branding, setup_dark_theme


class CornerTimeComparator:
    """
    Analyze time deltas across corners between 2-3 drivers/laps.

    Usage:
        cmp = CornerTimeComparator(drivers=[('LEC','fastest'), ('VER','fastest')],
                                   session=session, ...)
        cmp.plot_corner_time_deltas(baseline='per_corner_fastest')
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
        self._compute_corner_times()

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
        return corner_label(self.circuit_info, apex_idx)

    def _compute_corner_windows(self):
        """Determine entry/exit distances for each corner."""
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
        try:
            lap_num = int(lap_sel)
            picked = laps.pick_laps(lap_num)
            if len(picked) == 0:
                raise ValueError(f"No lap {lap_num} found for {drv}")
            return picked.iloc[0]
        except ValueError:
            # fallback if 'fastest' string passed in weird way or just error
            raise ValueError(f"Invalid lap selection: {lap_sel}")

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
            elif df.index.dtype.kind == 'm':  # TimedeltaIndex
                sec = timedelta_to_seconds(df.index.to_series())
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

        # Save for _compute_corner_windows
        self._max_distance = float(np.nanmax(max_dists)) if len(max_dists) else np.nan

    def _elapsed_time_between(self, df, d_start, d_end):
        """Calculate elapsed time using sector-constrained method for better accuracy."""
        dist = df['Distance'].values
        t = df['t_sec'].values

        # clamp window inside available distance
        d0 = max(min(d_start, dist.max()), dist.min())
        d1 = max(min(d_end, dist.max()), dist.min())
        if d1 <= d0:
            return np.nan

        start_idx = np.searchsorted(dist, d0)
        end_idx = np.searchsorted(dist, d1)

        if end_idx <= start_idx:
            return np.nan

        corner_dist = dist[start_idx:end_idx + 1]
        corner_time = t[start_idx:end_idx + 1]

        if len(corner_time) < 2:
            t0 = np.interp(d0, dist, t)
            t1 = np.interp(d1, dist, t)
            return float(t1 - t0)

        t_start = corner_time[0] if d0 <= corner_dist[0] else np.interp(d0, corner_dist, corner_time)
        t_end = corner_time[-1] if d1 >= corner_dist[-1] else np.interp(d1, corner_dist, corner_time)

        return float(t_end - t_start)

    def _compute_corner_times(self):
        """Compute per-corner elapsed times."""
        peak_dict = {}

        for i, (start, end, apex_idx) in enumerate(zip(self.entries, self.exits, self.apex_order)):
            corner_times = {}
            for lap in self.laps_loaded:
                try:
                    dt = self._elapsed_time_between(lap['df'], start, end)
                    corner_times[lap['name']] = dt
                except Exception as e:
                    logger.warning(f"Error in corner {i} for {lap['name']}: {e}")
                    continue

            if corner_times:
                label = self._corner_label(apex_idx)
                peak_dict[label] = corner_times

        self.df_corner_times = pd.DataFrame(peak_dict).T  # rows: corners, cols: display names

    def plot_corner_time_deltas(self, baseline='per_corner_fastest', figsize=(12, 6), save_path=None):
        """
        Plot grouped bar chart of corner time deltas. Returns (fig, ax).

        baseline: 'per_corner_fastest'  OR  ('fixed', '<DisplayName>')
        """
        if self.df_corner_times.empty:
            raise ValueError("No corner time data available.")

        # Compute deltas
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
        colors = [self.driver_color_map.get(col, 'white') for col in df_delta.columns]

        fig, ax = plt.subplots(figsize=figsize)
        # Apply dark theme using helper
        setup_dark_theme(fig, [ax])

        df_delta.plot.bar(ax=ax, rot=0, color=colors)

        event = getattr(self.session.event, 'EventName', str(self.session.event)) if hasattr(self.session, 'event') else self.session_name
        ref_caption = (
            "fastest per corner" if baseline == 'per_corner_fastest' else f"vs {baseline[1]} (fixed)"
        )
        ax.set_xlabel("Turn", color='white')
        ax.set_ylabel("Time Lost (s)", color='white')
        ax.set_title(f"Corner Time Deltas — {event} {self.year} {self.session_type} — {ref_caption}", color='white')
        ax.axhline(0, linewidth=1, color='white')
        legend = ax.legend(title="Driver/Lap", loc='upper left', bbox_to_anchor=(1.02, 0.9), fontsize=8)
        plt.setp(legend.get_title(), color='white')
        for text in legend.get_texts():
            text.set_color('white')

        # Annotate winners & diffs
        for i, corner in enumerate(df_delta.index):
            times = self.df_corner_times.loc[corner]
            winner = times.idxmin()
            win_t = times[winner]
            lines = [f"{winner}: {win_t:.3f}s"]
            for disp in df_delta.columns:
                if disp == winner:
                    continue
                diff = times[disp] - win_t
                lines.append(f"{disp}: {diff:+.3f}s")
            
            # Position text above largest bar for this corner
            max_y = max(0.02, df_delta.loc[corner].max())
            ax.text(i, max_y + 0.02, "\n".join(lines),
                    ha='center', va='bottom', color='white', fontsize=6)

        # Lap time info box
        lap_info_lines = []
        for lap_obj, spec in zip(self.lap_objs, self.driver_specs):
            lt = lap_obj["LapTime"]
            seconds = lt.total_seconds() if hasattr(lt, "total_seconds") else float(lt)
            m = int(seconds // 60)
            s = seconds - 60 * m
            lap_info_lines.append(f"{spec['display_name']}: {m}:{s:06.3f}")

        ax.text(
            1.02, 1.0,
            "\n".join(lap_info_lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="#1e1e1e", alpha=0.8, edgecolor="white"),
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        add_branding(fig, text_pos=(0.85, 0.02), logo_pos=[0.75, 0.5, 0.1, 0.1])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, ax
