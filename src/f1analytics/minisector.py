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

    Single-session usage:
        ms = MinisectorComparator(
            session=session,
            drivers={'LEC': 'fastest', 'NOR': 'fastest'},
            session_name="Pre-Season Testing", year=2026, session_type="",
        )

    Cross-session usage:
        ms = MinisectorComparator(
            laps=[
                (session_day5, 'LEC', 'fastest', 'Day 5'),
                (session_day6, 'LEC', 'fastest', 'Day 6'),
            ],
            session_name="Pre-Season Testing", year=2026, session_type="",
        )
    """

    def __init__(self, session_name, year, session_type,
                 session=None, drivers=None, laps=None, n_sectors=25):
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.n_sectors = n_sectors

        self.driver_specs = normalize_driver_specs(drivers=drivers, laps=laps, max_specs=4)
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.palette = assign_colors_simple([s['driver'] for s in self.driver_specs])

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
            sess = spec.get('session') or self.session

            drv_laps = sess.laps.pick_drivers(d)
            if lap_sel == 'fastest':
                lap = drv_laps.pick_fastest()
            else:
                lap_num = int(lap_sel)
                lap = drv_laps.pick_drivers(d).pick_laps(lap_num).iloc[0]

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
        """Divide the lap into N segments aligned with official sector boundaries.

        The 3 official sector boundaries are always minisector boundaries.
        The n_sectors minisectors are distributed proportionally across the
        3 sectors (proportional to sector distance length).
        """
        # Use the first driver's fine grid for reference distance
        ref_disp = self.display_names[0]
        ref_fine = self.telemetry[f'{ref_disp}_fine']
        ref_dist = ref_fine['Distance'].values
        ref_t = ref_fine['t_sec'].values
        max_dist = ref_dist.max()

        # ── Determine sector boundary distances from the reference lap ──
        ref_lap = self.lap_objs[ref_disp]
        s1t = ref_lap.get('Sector1Time', None)
        s2t = ref_lap.get('Sector2Time', None)

        sector_dists = [0.0]  # always start at 0
        if s1t is not None and s2t is not None:
            s1_sec = s1t.total_seconds() if hasattr(s1t, 'total_seconds') else float(s1t)
            s2_sec = s2t.total_seconds() if hasattr(s2t, 'total_seconds') else float(s2t)
            # Sector 1 ends at cumulative s1_sec, Sector 2 ends at s1_sec + s2_sec
            s1_end_t = s1_sec
            s2_end_t = s1_sec + s2_sec
            s1_end_d = float(np.interp(s1_end_t, ref_t, ref_dist))
            s2_end_d = float(np.interp(s2_end_t, ref_t, ref_dist))
            sector_dists.extend([s1_end_d, s2_end_d])
        else:
            # Fallback: split evenly into 3 if sector times unavailable
            sector_dists.extend([max_dist / 3, 2 * max_dist / 3])
            logger.warning("Sector times unavailable – falling back to equal thirds.")

        sector_dists.append(max_dist)
        self.sector_boundary_distances = np.array(sector_dists)  # [0, s1, s2, max]

        # ── Distribute n_sectors proportionally across the 3 sectors ──
        sector_lengths = np.diff(self.sector_boundary_distances)  # 3 values
        total_len = sector_lengths.sum()
        # Raw fractional allocation
        raw_alloc = (sector_lengths / total_len) * self.n_sectors
        # Integer allocation with rounding, ensuring the total is n_sectors
        alloc = np.round(raw_alloc).astype(int)
        # Adjust rounding error on the longest sector
        diff = self.n_sectors - alloc.sum()
        alloc[np.argmax(sector_lengths)] += diff
        alloc = np.maximum(alloc, 1)  # every sector gets at least 1 minisector

        # Build edges: per-sector linspace concatenated
        edges = np.array([self.sector_boundary_distances[0]])
        # Track which edge indices are sector boundaries (for label skip)
        sector_boundary_edge_indices = []
        for s_idx in range(3):
            s_start = self.sector_boundary_distances[s_idx]
            s_end = self.sector_boundary_distances[s_idx + 1]
            # alloc[s_idx] sub-segments inside this sector
            sub_edges = np.linspace(s_start, s_end, alloc[s_idx] + 1)[1:]
            edges = np.concatenate([edges, sub_edges])
            # The last edge just added is what marks the sector boundary
            if s_idx < 2:  # skip the final boundary (= track end)
                sector_boundary_edge_indices.append(len(edges) - 1)

        self.segment_edges = edges
        self.sector_boundary_edge_indices = set(sector_boundary_edge_indices)
        actual_n = len(edges) - 1
        if actual_n != self.n_sectors:
            logger.warning(
                "Sector-aligned allocation produced %d minisectors (requested %d).",
                actual_n, self.n_sectors,
            )
            self.n_sectors = actual_n

        # ── Compute elapsed time per segment per driver ──
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

    def plot_track_map(self, figsize=(14, 10), save_path=None, show_sectors=True):
        """
        Plot the circuit map colored by the fastest driver per segment.

        Parameters
        ----------
        show_sectors : bool, default True
            If True, draw numbered markers at each minisector boundary
            so the sectors are identifiable on the map.

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

        # Corner annotations — only shown when minisector labels are off
        if not show_sectors and self.circuit_info is not None and hasattr(self.circuit_info, 'corners'):
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

        # Minisector boundary labels with perpendicular offset to avoid overlap
        if show_sectors:
            skip = getattr(self, 'sector_boundary_edge_indices', set())

            # Compute track centroid for outward direction
            track_center = np.mean(xy_rot, axis=0)
            # Offset distance scaled to track size
            span = max(
                xy_rot[:, 0].max() - xy_rot[:, 0].min(),
                xy_rot[:, 1].max() - xy_rot[:, 1].min(),
            )
            offset_dist = span * 0.030
            min_label_sep = offset_dist * 1.2  # minimum separation between labels

            # Helper: compute outward perpendicular at a track point
            def _outward_offset(d_edge, mult=1.0):
                pt_idx = np.argmin(np.abs(dist - d_edge))
                # Tangent via a small window for stability
                lo = max(0, pt_idx - 3)
                hi = min(len(xy_rot) - 1, pt_idx + 3)
                tangent = xy_rot[hi] - xy_rot[lo]
                perp = np.array([-tangent[1], tangent[0]], dtype=float)
                norm = np.linalg.norm(perp)
                if norm > 0:
                    perp /= norm
                # Point outward (away from track centroid)
                pt_on_track = xy_rot[pt_idx]
                if np.dot(perp, pt_on_track - track_center) < 0:
                    perp = -perp
                return pt_on_track, perp * offset_dist * mult

            # ── Collect all label entries (minisector + sector boundaries) ──
            label_entries = []  # list of dicts with info for each label
            for idx in range(1, self.n_sectors + 1):
                d_edge = self.segment_edges[min(idx, len(self.segment_edges) - 1)]
                is_sector_bdy = idx in skip
                label_entries.append({
                    'idx': idx, 'd_edge': d_edge,
                    'is_sector_bdy': is_sector_bdy,
                })

            # ── Compute positions with collision avoidance ──
            placed_positions = []  # list of np arrays
            for entry in label_entries:
                base_mult = 1.6 if entry['is_sector_bdy'] else 1.0
                pt_on_track, off = _outward_offset(entry['d_edge'], mult=base_mult)
                label_pos = pt_on_track + off

                # Repel from already-placed labels
                for attempt in range(5):
                    too_close = False
                    for existing in placed_positions:
                        sep = np.linalg.norm(label_pos - existing)
                        if sep < min_label_sep:
                            too_close = True
                            break
                    if not too_close:
                        break
                    # Push further outward
                    base_mult += 0.6
                    _, off = _outward_offset(entry['d_edge'], mult=base_mult)
                    label_pos = pt_on_track + off

                placed_positions.append(label_pos)
                entry['pt_on_track'] = pt_on_track
                entry['label_pos'] = label_pos

            # ── Draw labels ──
            for entry in label_entries:
                pt_on = entry['pt_on_track']
                lbl_pos = entry['label_pos']

                if entry['is_sector_bdy']:
                    # Sector boundary: marks end of sector (S1, S2)
                    sb_num = sorted(skip).index(entry['idx']) + 1
                    ax.plot(
                        [pt_on[0], lbl_pos[0]], [pt_on[1], lbl_pos[1]],
                        color='orange', linewidth=1.2, alpha=0.7, zorder=5,
                    )
                    ax.text(
                        lbl_pos[0], lbl_pos[1], f'S{sb_num}',
                        ha='center', va='center', color='orange',
                        fontsize=10, weight='bold', zorder=9,
                        bbox=dict(
                            boxstyle='round,pad=0.3', facecolor='#1e1e1e',
                            edgecolor='orange', linewidth=2, alpha=0.95,
                        ),
                    )
                else:
                    # Minisector boundary: cyan dot + leader line
                    ax.plot(
                        [pt_on[0], lbl_pos[0]], [pt_on[1], lbl_pos[1]],
                        color='#00e5ff', linewidth=0.5, alpha=0.5, zorder=5,
                    )
                    ax.scatter(
                        lbl_pos[0], lbl_pos[1], color='#00e5ff', s=65,
                        edgecolor='white', linewidth=0.8, zorder=6,
                    )
                    ax.text(
                        lbl_pos[0], lbl_pos[1], str(entry['idx']),
                        ha='center', va='center', color='black',
                        fontsize=5.5, weight='bold', zorder=7,
                    )

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

        # Sector boundary dividers
        skip = getattr(self, 'sector_boundary_edge_indices', set())
        for sb_idx in skip:
            if 0 < sb_idx < self.n_sectors:
                x_pos = sb_idx + bar_w * (n_drivers - 1) / 2 - 0.5
                ax.axvline(
                    x_pos, color='orange', linewidth=1.5,
                    linestyle='--', alpha=0.8, zorder=3,
                )

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info("Saved bar chart to %s", save_path)

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
