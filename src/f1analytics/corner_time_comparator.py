"""
Corner time comparator — analyze time deltas per corner.

Computes elapsed time per corner window for each driver/lap, then plots
grouped bar charts showing who gained/lost time at each corner.

The corner windows partition the entire lap distance so that the sum of
corner times equals (approximately) the total lap time.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
from f1analytics.palette import driver_colors as colors_pilots
from f1analytics.config import logger
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.corner_utils import corner_label
from f1analytics.plot_utils import add_branding, setup_dark_theme, assign_colors_simple


def _extract_seconds(df):
    """Extract a float-seconds array from the first time-like column found."""
    if 'Time' in df.columns:
        return timedelta_to_seconds(df['Time'])
    if 'SessionTime' in df.columns:
        return timedelta_to_seconds(df['SessionTime'])
    raise ValueError("No time-like column ('Time' or 'SessionTime') found in telemetry.")


def _build_fine_grid(tel_df):
    """
    Build a high-resolution (1m) distance-time grid from telemetry,
    matching the approach in delta_time_sector_constrained.
    """
    tel = tel_df.dropna(subset=['Distance', 'Time', 'Speed']).sort_values('Distance')
    tel = tel.drop_duplicates(subset=['Distance'], keep='first')

    dist = tel['Distance'].values
    time_s = tel['Time'].dt.total_seconds().values

    # 1m resolution grid
    grid = np.arange(dist.min(), dist.max(), 1.0)
    time_grid = np.interp(grid, dist, time_s)

    return pd.DataFrame({'Distance': grid, 't_sec': time_grid})


class CornerTimeComparator:
    """
    Analyze time deltas across corners between 2-4 drivers/laps.

    Single-session usage:
        cmp = CornerTimeComparator(
            session=session,
            drivers={'LEC': 'fastest', 'VER': 'fastest'},
            session_name="Pre-Season Testing",
            year=2026,
            session_type="",
        )

    Cross-session usage:
        cmp = CornerTimeComparator(
            laps=[
                (session_day5, 'LEC', 'fastest', 'Day 5'),
                (session_day6, 'LEC', 'fastest', 'Day 6'),
            ],
            session_name="Pre-Season Testing",
            year=2026,
            session_type="",
        )
    """

    def __init__(self, session_name: str, year: int, session_type: str, *,
                 session=None, drivers=None, laps=None, n_interp=200):
        self.driver_specs = normalize_driver_specs(drivers=drivers, laps=laps, max_specs=4)
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.n_interp = int(n_interp)
        self.session_name = session_name
        self.year = year
        self.session_type = session_type

        # Determine session: explicit or from first spec
        self.session = session
        first_spec_session = self.driver_specs[0].get('session')
        if self.session is None and first_spec_session is not None:
            self.session = first_spec_session

        self.circuit_info = (
            self.session.get_circuit_info()
            if self.session and hasattr(self.session, "get_circuit_info")
            else None
        )

        # Colors — use shared utility
        driver_codes = [s['driver'] for s in self.driver_specs]
        self.palette = assign_colors_simple(driver_codes)
        self.driver_color_map = dict(zip(self.display_names, self.palette))

        self._load_laps()
        self._compute_corner_windows()
        self._compute_corner_times()
        self._normalize_to_lap_times()

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_laps(self):
        """Load telemetry using get_telemetry() for accurate timing, on 1m grid."""
        self.lap_objs = []
        self.laps_loaded = []
        max_dists = []

        for spec in self.driver_specs:
            drv, lap_sel, disp = spec['driver'], spec['lap'], spec['display_name']
            lap_obj = self._pick_lap(spec)

            # Use get_telemetry() for higher resolution
            raw_tel = lap_obj.get_telemetry()
            df = _build_fine_grid(raw_tel)

            self.lap_objs.append(lap_obj)
            self.laps_loaded.append({'driver': drv, 'name': disp, 'df': df})
            max_dists.append(df['Distance'].max())

        self._max_distance = float(np.nanmax(max_dists)) if max_dists else np.nan

    def _pick_lap(self, spec):
        """Pick a lap object — uses per-spec session if available, else self.session."""
        sess = spec.get('session') or self.session
        drv = spec['driver']
        lap_sel = spec['lap']

        drv_laps = sess.laps.pick_drivers(drv)
        if lap_sel == 'fastest':
            return drv_laps.pick_fastest()
        lap_num = int(lap_sel)
        picked = drv_laps.pick_laps(lap_num)
        if len(picked) == 0:
            raise ValueError(f"No lap {lap_num} found for {drv}")
        return picked.iloc[0]

    # ── Corner window computation ─────────────────────────────────────────

    def _corner_label(self, apex_idx):
        return corner_label(self.circuit_info, apex_idx)

    def _compute_corner_windows(self):
        """
        Determine entry/exit distances for each corner.

        Windows partition the full lap: entry[0]=0, exit[-1]=max_distance.
        Boundaries are midpoints between consecutive apex distances.
        """
        corner_df = self.circuit_info.corners.sort_values('Distance')
        self.apex_order = list(corner_df.index)
        distances = corner_df['Distance'].values

        # entries: [0, midpoint(apex0, apex1), midpoint(apex1, apex2), ...]
        self.entries = [0.0] + [
            (distances[i - 1] + distances[i]) / 2 for i in range(1, len(distances))
        ]

        # exits: [midpoint(apex0, apex1), ..., max_lap_distance]
        self.exits = [
            (distances[i] + distances[i + 1]) / 2 for i in range(len(distances) - 1)
        ]
        self.exits.append(float(self._max_distance))

        self.apex_distances = distances

    # ── Corner time computation ───────────────────────────────────────────

    def _elapsed_time_between(self, df, d_start, d_end):
        """Compute elapsed time between two distances via interpolation."""
        dist = df['Distance'].values
        t = df['t_sec'].values

        d0 = np.clip(d_start, dist.min(), dist.max())
        d1 = np.clip(d_end, dist.min(), dist.max())
        if d1 <= d0:
            return np.nan

        t0 = np.interp(d0, dist, t)
        t1 = np.interp(d1, dist, t)
        return float(t1 - t0)

    def _compute_corner_times(self):
        """Compute per-corner elapsed times for each driver."""
        peak_dict = {}

        for i, (start, end, apex_idx) in enumerate(
            zip(self.entries, self.exits, self.apex_order)
        ):
            corner_times = {}
            for lap in self.laps_loaded:
                try:
                    dt = self._elapsed_time_between(lap['df'], start, end)
                    corner_times[lap['name']] = dt
                except Exception as e:
                    logger.warning("Error in corner %d for %s: %s", i, lap['name'], e)
                    continue

            if corner_times:
                label = self._corner_label(apex_idx)
                peak_dict[label] = corner_times

        self.df_corner_times = pd.DataFrame(peak_dict).T

    def _normalize_to_lap_times(self):
        """
        Distribute the residual (sum_corners - lap_time) proportionally
        so that corner times sum exactly to the actual lap time.
        """
        for lap_obj, spec in zip(self.lap_objs, self.driver_specs):
            disp = spec['display_name']
            if disp not in self.df_corner_times.columns:
                continue

            lt = lap_obj['LapTime']
            lap_seconds = lt.total_seconds() if hasattr(lt, 'total_seconds') else float(lt)
            corner_sum = self.df_corner_times[disp].sum()

            if corner_sum > 0 and not np.isnan(corner_sum):
                scale = lap_seconds / corner_sum
                self.df_corner_times[disp] *= scale
                logger.debug(
                    "%s: normalized corner times by factor %.6f (was %.3f, lap %.3f)",
                    disp, scale, corner_sum, lap_seconds,
                )

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self):
        """
        Sanity check: sum of corner times vs actual lap time.
        Returns a DataFrame with columns [Driver, Sum_Corners, Lap_Time, Diff, Error_%].
        """
        rows = []
        for lap_obj, spec in zip(self.lap_objs, self.driver_specs):
            disp = spec['display_name']
            lt = lap_obj['LapTime']
            lap_seconds = lt.total_seconds() if hasattr(lt, 'total_seconds') else float(lt)

            if disp in self.df_corner_times.columns:
                corner_sum = self.df_corner_times[disp].sum()
            else:
                corner_sum = np.nan

            diff = corner_sum - lap_seconds
            error_pct = (abs(diff) / lap_seconds) * 100 if lap_seconds > 0 else np.nan

            rows.append({
                'Driver': disp,
                'Sum Corners (s)': round(corner_sum, 3),
                'Lap Time (s)': round(lap_seconds, 3),
                'Diff (s)': round(diff, 3),
                'Error (%)': round(error_pct, 2),
            })

        df = pd.DataFrame(rows).set_index('Driver')
        return df

    # ── Data accessors ────────────────────────────────────────────────────

    def get_table(self):
        """Return the raw corner times DataFrame (rows=corners, cols=drivers)."""
        return self.df_corner_times.copy()

    def get_delta_table(self, baseline='per_corner_fastest'):
        """Return corner time deltas relative to the baseline."""
        if baseline == 'per_corner_fastest':
            ref = self.df_corner_times.min(axis=1)
        elif isinstance(baseline, (list, tuple)) and baseline[0] == 'fixed':
            ref = self.df_corner_times[baseline[1]]
        else:
            raise ValueError("baseline must be 'per_corner_fastest' or ('fixed', '<name>')")
        return self.df_corner_times.subtract(ref, axis=0)

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_corner_time_deltas(self, baseline='per_corner_fastest',
                                figsize=(16, 7), save_path=None):
        """
        Plot grouped bar chart of corner time deltas. Returns (fig, ax).

        baseline: 'per_corner_fastest' or ('fixed', '<DisplayName>')
        """
        if self.df_corner_times.empty:
            raise ValueError("No corner time data available.")

        df_delta = self.get_delta_table(baseline)
        colors = [self.driver_color_map.get(col, 'white') for col in df_delta.columns]

        fig, ax = plt.subplots(figsize=figsize)
        setup_dark_theme(fig, [ax])

        df_delta.plot.bar(ax=ax, rot=0, color=colors, edgecolor='white', linewidth=0.3)

        # Title with lap times subtitle
        event = self.session_name
        if self.session and hasattr(self.session, 'event'):
            event = getattr(self.session.event, 'EventName', self.session_name)
        ref_caption = (
            "fastest per corner" if baseline == 'per_corner_fastest'
            else f"vs {baseline[1]} (fixed)"
        )
        title_parts = [f"{event} {self.year}"]
        if self.session_type:
            title_parts.append(self.session_type)
        title_parts.append(ref_caption)

        lap_strs = []
        for lap_obj, spec in zip(self.lap_objs, self.driver_specs):
            lt = lap_obj["LapTime"]
            seconds = lt.total_seconds() if hasattr(lt, "total_seconds") else float(lt)
            m = int(seconds // 60)
            s = seconds - 60 * m
            lap_strs.append(f"{spec['display_name']}: {m}:{s:06.3f}")

        ax.set_title(
            "Corner Time Deltas — " + " — ".join(title_parts) + "\n" + "  |  ".join(lap_strs),
            color='white', fontsize=13, pad=15
        )
        ax.set_xlabel("Turn", color='white', fontsize=11)
        ax.set_ylabel("Time Lost (s)", color='white', fontsize=11)
        ax.axhline(0, linewidth=1, color='white', alpha=0.5)

        # Legend
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
        legend.get_frame().set_facecolor('#1e1e1e')
        legend.get_frame().set_edgecolor('white')
        for text in legend.get_texts():
            text.set_color('white')

        # Annotate max delta per corner
        for i, corner in enumerate(df_delta.index):
            max_delta = df_delta.loc[corner].max()
            if max_delta > 0.005:
                ax.text(i, max_delta + 0.005, f"+{max_delta:.3f}s",
                        ha='center', va='bottom', color='white',
                        fontsize=7, fontweight='bold')

        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        return fig, ax
