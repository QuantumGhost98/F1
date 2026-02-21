"""
Minisector comparator — divide the lap into N equal-distance segments
and compare elapsed time per segment across 2-3 drivers.

Two visualizations:
  1. plot_track_map()  — circuit map colored by fastest driver per segment
  2. plot_bar_chart()  — grouped bar chart of time deltas per segment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.plot_utils import assign_colors_simple, setup_dark_theme, add_branding
from f1analytics.config import logger


def _build_fine_grid(tel_df):
    """
    Build a 1m resolution distance-time grid from telemetry
    for accurate elapsed-time computation.
    """
    tel = tel_df.dropna(subset=['Distance', 'Time', 'Speed']).sort_values('Distance')
    tel = tel.drop_duplicates(subset=['Distance'], keep='first')
    dist = tel['Distance'].values
    time_s = tel['Time'].dt.total_seconds().values
    grid = np.arange(dist.min(), dist.max(), 1.0)
    time_grid = np.interp(grid, dist, time_s)
    return pd.DataFrame({'Distance': grid, 't_sec': time_grid})


class MinisectorComparator:
    """
    Compare lap performance across N equal-distance mini-sectors.

    Usage:
        ms = MinisectorComparator(
            session=session,
            session_name="Pre-Season Testing",
            year=2026,
            session_type="",
            drivers={'LEC': 'fastest', 'NOR': 'fastest'},
            n_sectors=25,
        )
        ms.plot_track_map()
        ms.plot_bar_chart()
    """

    def __init__(self, session, session_name, year, session_type, drivers,
                 n_sectors=25):
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.n_sectors = n_sectors
        self.circuit_info = (
            session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        )

        self.driver_specs = normalize_driver_specs(drivers, max_specs=4)
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.palette = assign_colors_simple([s['driver'] for s in self.driver_specs])

        self.laps = session.laps
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = (
            self.laps["LapTime"].dt.total_seconds()
        )

        # Loaded data
        self.telemetry = {}   # display_name -> DataFrame
        self.lap_objs = {}    # display_name -> lap Series
        self._load_laps()

        # Computed results
        self.segment_edges = None
        self.segment_times = None  # DataFrame: rows=segments, cols=display_names
        self.segment_deltas = None
        self.segment_winner = None  # Series: segment_idx -> display_name
        self._compute_segments()
        self._normalize_to_lap_times()

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_laps(self):
        for spec in self.driver_specs:
            d = spec['driver']
            lap_sel = spec['lap']
            disp = spec['display_name']

            if lap_sel == 'fastest':
                lap = self.transformed_laps.pick_drivers(d).pick_fastest()
            else:
                lap_num = int(lap_sel)
                lap = self.transformed_laps.pick_drivers(d).pick_laps(lap_num).iloc[0]

            # lap.telemetry for X/Y coordinates (needed for track map)
            tel = lap.telemetry.copy()

            # Ensure t_sec column from Time
            if 'Time' in tel.columns:
                tel['t_sec'] = np.asarray(
                    timedelta_to_seconds(tel['Time']), dtype=float
                )
            elif 'SessionTime' in tel.columns:
                tel['t_sec'] = np.asarray(
                    timedelta_to_seconds(tel['SessionTime']), dtype=float
                )
                tel['t_sec'] = tel['t_sec'] - tel['t_sec'].iloc[0]

            self.telemetry[disp] = tel
            self.lap_objs[disp] = lap

            # Also build a fine 1m grid for accurate timing
            raw_tel = lap.get_telemetry()
            self.telemetry[f'{disp}_fine'] = _build_fine_grid(raw_tel)

    # ── Segment computation ───────────────────────────────────────────────

    def _compute_segments(self):
        """Divide the lap into N equal-distance segments and compute times."""
        # Use the first driver's fine grid for reference distance
        ref_disp = self.display_names[0]
        ref_fine = self.telemetry[f'{ref_disp}_fine']
        max_dist = ref_fine['Distance'].max()

        # Create N+1 edges
        self.segment_edges = np.linspace(0, max_dist, self.n_sectors + 1)

        # Compute elapsed time per segment per driver using the fine grid
        times = {}
        for disp in self.display_names:
            fine = self.telemetry[f'{disp}_fine']
            dist = fine['Distance'].values
            t = fine['t_sec'].values
            seg_times = []
            for i in range(self.n_sectors):
                d_start = self.segment_edges[i]
                d_end = self.segment_edges[i + 1]
                t_start = np.interp(d_start, dist, t)
                t_end = np.interp(d_end, dist, t)
                seg_times.append(t_end - t_start)
            times[disp] = seg_times

        self.segment_times = pd.DataFrame(times)

        # Deltas (relative to fastest per segment)
        ref = self.segment_times.min(axis=1)
        self.segment_deltas = self.segment_times.subtract(ref, axis=0)

        # Winner per segment
        self.segment_winner = self.segment_times.idxmin(axis=1)

    def _normalize_to_lap_times(self):
        """
        Scale segment times proportionally so they sum exactly
        to the official lap time for each driver.
        """
        for disp in self.display_names:
            if disp not in self.segment_times.columns:
                continue
            lt = self.lap_objs[disp]['LapTime']
            lap_seconds = lt.total_seconds() if hasattr(lt, 'total_seconds') else float(lt)
            seg_sum = self.segment_times[disp].sum()

            if seg_sum > 0 and not np.isnan(seg_sum):
                scale = lap_seconds / seg_sum
                self.segment_times[disp] *= scale
                logger.debug(
                    "%s: normalized segment times by factor %.6f (was %.3f, lap %.3f)",
                    disp, scale, seg_sum, lap_seconds,
                )

        # Recompute deltas and winners after normalization
        ref = self.segment_times.min(axis=1)
        self.segment_deltas = self.segment_times.subtract(ref, axis=0)
        self.segment_winner = self.segment_times.idxmin(axis=1)

    # ── Track map visualization ───────────────────────────────────────────

    def plot_track_map(self, figsize=(14, 10), save_path=None):
        """
        Plot the circuit map colored by the fastest driver per segment.
        Returns (fig, ax).
        """
        # Use reference driver's X/Y coordinates
        ref_disp = self.display_names[0]
        ref_tel = self.telemetry[ref_disp]

        if 'X' not in ref_tel.columns or 'Y' not in ref_tel.columns:
            raise ValueError(
                "Telemetry has no X/Y columns. "
                "Make sure the session is loaded with telemetry=True."
            )

        x = ref_tel['X'].to_numpy(dtype=float)
        y = ref_tel['Y'].to_numpy(dtype=float)
        dist = ref_tel['Distance'].to_numpy(dtype=float)

        # Rotation
        angle = 0.0
        if self.circuit_info is not None:
            angle = float(self.circuit_info.rotation) / 180.0 * np.pi

        xy = np.c_[x, y]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        xy_rot = xy.dot(rot_mat)

        # Build color per telemetry point based on which segment it falls in
        seg_indices = np.digitize(dist, self.segment_edges) - 1
        seg_indices = np.clip(seg_indices, 0, self.n_sectors - 1)

        # Map display_name -> color
        color_map = dict(zip(self.display_names, self.palette))
        winner_colors = [
            color_map[self.segment_winner[seg_idx]]
            for seg_idx in seg_indices
        ]

        # Build line segments
        points = xy_rot.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection
        lc = LineCollection(
            segments,
            colors=winner_colors[:-1],
            linewidths=5,
            capstyle='round',
            zorder=1,
        )

        fig, ax = plt.subplots(figsize=figsize)
        setup_dark_theme(fig, [ax])
        ax.axis('off')

        ax.add_collection(lc)

        # Set limits
        margin = 0.08 * max(
            xy_rot[:, 0].max() - xy_rot[:, 0].min(),
            xy_rot[:, 1].max() - xy_rot[:, 1].min(),
        )
        ax.set_xlim(xy_rot[:, 0].min() - margin, xy_rot[:, 0].max() + margin)
        ax.set_ylim(xy_rot[:, 1].min() - margin, xy_rot[:, 1].max() + margin)
        ax.set_aspect('equal', adjustable='box')

        # Corner annotations
        if self.circuit_info is not None and hasattr(self.circuit_info, 'corners'):
            for _, corner in self.circuit_info.corners.iterrows():
                base = np.array([corner['X'], corner['Y']], dtype=float)
                rpt = base.reshape(1, 2).dot(rot_mat)[0]
                label = f"{int(corner['Number'])}"
                letter = corner.get('Letter', '')
                if letter is not None and str(letter).strip() and not pd.isna(letter):
                    label += str(letter).strip()
                ax.scatter(rpt[0], rpt[1], color='white', s=120,
                           edgecolor='orange', linewidth=1.5, zorder=4)
                ax.text(rpt[0], rpt[1], label, ha='center', va='center',
                        color='black', fontsize=8, weight='bold', zorder=5)

        # Legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=color_map[disp], linewidth=6, label=disp)
            for disp in self.display_names
        ]
        legend = ax.legend(
            handles=handles, loc='upper left', fontsize=12,
            facecolor='#1e1e1e', edgecolor='white', framealpha=0.9,
        )
        for text in legend.get_texts():
            text.set_color('white')

        # Count segments won
        win_counts = self.segment_winner.value_counts()
        info_lines = []
        for disp in self.display_names:
            n_won = win_counts.get(disp, 0)
            lt = self.lap_objs[disp]['LapTime']
            secs = lt.total_seconds() if hasattr(lt, 'total_seconds') else float(lt)
            m = int(secs // 60)
            s = secs - 60 * m
            info_lines.append(f"{disp}: {m}:{s:06.3f}  ({n_won}/{self.n_sectors} sectors)")

        ax.text(
            0.02, 0.02, "\n".join(info_lines),
            transform=ax.transAxes, ha='left', va='bottom',
            color='white', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.85,
                      edgecolor='white'),
            zorder=10,
        )

        # Title
        parts = [f"{self.session_name} {self.year}"]
        if self.session_type:
            parts.append(self.session_type)
        parts.append("Minisector Comparison")
        ax.set_title(
            " — ".join(parts), fontsize=15, color='white', pad=20, weight='bold',
        )

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info("Saved track map to %s", save_path)

        plt.show()
        return fig, ax

    # ── Bar chart visualization ───────────────────────────────────────────

    def plot_bar_chart(self, figsize=(16, 7), save_path=None):
        """
        Plot a grouped bar chart of time deltas per mini-sector.
        Returns (fig, ax).
        """
        df_delta = self.segment_deltas
        n_seg = len(df_delta)
        n_drivers = len(self.display_names)
        x = np.arange(n_seg)
        bar_w = 0.8 / n_drivers

        fig, ax = plt.subplots(figsize=figsize)
        setup_dark_theme(fig, [ax])

        for i, (disp, col) in enumerate(zip(self.display_names, self.palette)):
            ax.bar(
                x + i * bar_w, df_delta[disp], width=bar_w,
                label=disp, color=col, edgecolor='white', linewidth=0.3,
            )

        ax.set_xticks(x + bar_w * (n_drivers - 1) / 2)
        ax.set_xticklabels([f"S{i+1}" for i in range(n_seg)], fontsize=8)

        # Title with lap times
        parts = [f"{self.session_name} {self.year}"]
        if self.session_type:
            parts.append(self.session_type)
        parts.append("Minisector Deltas")

        lap_strs = []
        for disp in self.display_names:
            lt = self.lap_objs[disp]['LapTime']
            secs = lt.total_seconds() if hasattr(lt, 'total_seconds') else float(lt)
            m = int(secs // 60)
            s = secs - 60 * m
            lap_strs.append(f"{disp}: {m}:{s:06.3f}")

        ax.set_title(
            " — ".join(parts) + "\n" + "  |  ".join(lap_strs),
            color='white', fontsize=13, pad=15,
        )
        ax.set_xlabel("Mini-sector", color='white', fontsize=11)
        ax.set_ylabel("Time Lost (s)", color='white', fontsize=11)
        ax.axhline(0, linewidth=1, color='white', alpha=0.5)

        # Legend inside
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
        legend.get_frame().set_facecolor('#1e1e1e')
        legend.get_frame().set_edgecolor('white')
        for text in legend.get_texts():
            text.set_color('white')

        # Annotate max delta per segment
        for i in range(n_seg):
            max_d = df_delta.iloc[i].max()
            if max_d > 0.005:
                ax.text(
                    i + bar_w * (n_drivers - 1) / 2, max_d + 0.003,
                    f"+{max_d:.3f}s",
                    ha='center', va='bottom', color='white',
                    fontsize=6, fontweight='bold',
                )

        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info("Saved bar chart to %s", save_path)

        plt.show()
        return fig, ax

    # ── Data accessors ────────────────────────────────────────────────────

    def get_segment_table(self):
        """Return segment times DataFrame (rows=segments, cols=drivers)."""
        return self.segment_times.copy()

    def get_delta_table(self):
        """Return segment deltas DataFrame (relative to fastest per segment)."""
        return self.segment_deltas.copy()

    def get_winner_summary(self):
        """Return dict of {display_name: number_of_segments_won}."""
        return self.segment_winner.value_counts().to_dict()

    def validate(self):
        """
        Sanity check: sum of segment times vs actual lap time.
        Returns a DataFrame with [Sum Segments, Lap Time, Diff, Error %].
        """
        rows = []
        for disp in self.display_names:
            lt = self.lap_objs[disp]['LapTime']
            lap_seconds = lt.total_seconds() if hasattr(lt, 'total_seconds') else float(lt)
            seg_sum = self.segment_times[disp].sum() if disp in self.segment_times.columns else np.nan
            diff = seg_sum - lap_seconds
            error_pct = (abs(diff) / lap_seconds) * 100 if lap_seconds > 0 else np.nan

            rows.append({
                'Driver': disp,
                'Sum Segments (s)': round(seg_sum, 3),
                'Lap Time (s)': round(lap_seconds, 3),
                'Diff (s)': round(diff, 3),
                'Error (%)': round(error_pct, 2),
            })

        return pd.DataFrame(rows).set_index('Driver')
