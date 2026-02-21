"""
Dual throttle comparison — two drivers drawn as parallel heatmap traces
around the full circuit, colored by throttle percentage.

  • Reference driver = inner line (offset -d)
  • Comparison driver = outer line (offset +d)
  • Both colored by the SAME throttle colormap (0–100 %)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from f1analytics.config import logger
from f1analytics.plot_utils import add_branding, setup_dark_theme


class DualThrottleComparisonVisualizer:
    """
    Concentric/parallel throttle comparison for two drivers.

    Usage:
        vis = DualThrottleComparisonVisualizer(
            session=session,
            reference_driver='LEC',
            comparison_driver='NOR',
            event_name=session_name,
            year=session_year,
            session_name=session_type,
        )
        vis.plot()
    """

    def __init__(self, session, reference_driver, comparison_driver,
                 event_name=None, year=None, session_name=None,
                 offset_distance=200.0, annotate_corners=True,
                 annotate_sectors=False, highlight_brake_zones=False,
                 brake_threshold=20.0, line_width=8.0,
                 use_speed_width=True, use_bgr_scale=True):
        """
        Parameters
        ----------
        session             : FastF1 loaded session
        reference_driver    : driver code for inner line, e.g. 'LEC'
        comparison_driver   : driver code for outer line, e.g. 'NOR'
        event_name          : e.g. "Pre-Season Testing"
        year                : e.g. 2026
        session_name        : e.g. "Day 6"
        offset_distance     : distance between parallel lines (track coord units)
        annotate_corners    : show corner numbers on the map
        annotate_sectors    : show approximate sector boundary markers
        highlight_brake_zones: highlight points where throttle < brake_threshold
        brake_threshold     : throttle (%) below which is considered braking
        line_width          : line width when use_speed_width=False
        use_speed_width     : scale line width by speed
        use_bgr_scale       : True -> Blue→Green→Red, False -> 'viridis'
        """
        self.session = session
        self.reference_driver = reference_driver
        self.comparison_driver = comparison_driver
        self.event_name = event_name
        self.year = year
        self.session_name = session_name
        self.offset_distance = offset_distance
        self.annotate_corners = annotate_corners
        self.annotate_sectors = annotate_sectors
        self.highlight_brake_zones = highlight_brake_zones
        self.brake_threshold = brake_threshold
        self.line_width = line_width
        self.use_speed_width = use_speed_width
        self.use_bgr_scale = use_bgr_scale

        self.circuit_info = (
            session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        )
        self.laps = session.laps

        # Rotation angle (degrees -> radians)
        self._angle = float(self.circuit_info.rotation) / 180.0 * np.pi

        self._load_laps()

    def _load_laps(self):
        """Load fastest laps and telemetry for both drivers."""
        self._lap_ref = self.laps.pick_drivers(self.reference_driver).pick_fastest()
        self._lap_comp = self.laps.pick_drivers(self.comparison_driver).pick_fastest()

        tel_ref = self._lap_ref.telemetry
        tel_comp = self._lap_comp.telemetry

        self.x_ref = tel_ref['X'].to_numpy()
        self.y_ref = tel_ref['Y'].to_numpy()
        self.throttle_ref = tel_ref['Throttle'].to_numpy()
        self.speed_ref = tel_ref['Speed'].to_numpy()

        self.x_comp = tel_comp['X'].to_numpy()
        self.y_comp = tel_comp['Y'].to_numpy()
        self.throttle_comp = tel_comp['Throttle'].to_numpy()
        self.speed_comp = tel_comp['Speed'].to_numpy()

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _rotate(xy, *, angle):
        """Rotate XY coordinates by +angle (radians)."""
        mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
        return xy.dot(mat)

    @staticmethod
    def _parallel_offset(x, y, offset):
        """Create a parallel polyline offset by 'offset' (track coords)."""
        dx = np.gradient(x)
        dy = np.gradient(y)
        perp_x = -dy
        perp_y = dx
        norm = np.hypot(perp_x, perp_y)
        norm[norm == 0] = 1.0
        perp_x /= norm
        perp_y /= norm
        return x + perp_x * offset, y + perp_y * offset

    def _choose_cmap(self):
        if self.use_bgr_scale:
            return LinearSegmentedColormap.from_list('bgr', ['blue', 'green', 'red'])
        return mpl.cm.viridis

    # ── Plot ──────────────────────────────────────────────────────────────

    def plot(self, figsize=(14, 10), save_path=None):
        """Plot the dual throttle comparison. Returns (fig, ax)."""
        fig, ax = plt.subplots(figsize=figsize)
        setup_dark_theme(fig, [ax])
        ax.axis('off')

        # Rotate coords
        rot_ref = self._rotate(np.c_[self.x_ref, self.y_ref], angle=self._angle)
        rot_comp = self._rotate(np.c_[self.x_comp, self.y_comp], angle=self._angle)

        # Parallel offsets (inner = negative, outer = positive)
        X_ref, Y_ref = self._parallel_offset(
            rot_ref[:, 0], rot_ref[:, 1], -self.offset_distance
        )
        X_comp, Y_comp = self._parallel_offset(
            rot_comp[:, 0], rot_comp[:, 1], +self.offset_distance
        )

        # Build line segments
        def _segments(x, y):
            pts = np.c_[x, y].reshape(-1, 1, 2)
            return np.concatenate([pts[:-1], pts[1:]], axis=1)

        segs_ref = _segments(X_ref, Y_ref)
        segs_comp = _segments(X_comp, Y_comp)

        # Colormap & normalization for throttle (0..100 %)
        cmap = self._choose_cmap()
        norm = mpl.colors.Normalize(vmin=0.0, vmax=100.0)

        # Widths (constant or speed-scaled across BOTH traces)
        if self.use_speed_width:
            all_speed = np.r_[self.speed_ref, self.speed_comp]
            smin, smax = float(all_speed.min()), float(all_speed.max())
            if smax > smin:
                width_ref = 1.0 + 8.0 * (self.speed_ref - smin) / (smax - smin)
                width_comp = 1.0 + 8.0 * (self.speed_comp - smin) / (smax - smin)
            else:
                width_ref = np.full_like(self.speed_ref, 2.0)
                width_comp = np.full_like(self.speed_comp, 2.0)
        else:
            width_ref = np.full_like(self.speed_ref, self.line_width)
            width_comp = np.full_like(self.speed_comp, self.line_width)

        # Line collections
        lc_ref = LineCollection(
            segs_ref, cmap=cmap, norm=norm, linewidths=width_ref,
            capstyle='butt', antialiased=True, zorder=1,
            label=self.reference_driver,
        )
        lc_ref.set_array(self.throttle_ref)
        ax.add_collection(lc_ref)

        lc_comp = LineCollection(
            segs_comp, cmap=cmap, norm=norm, linewidths=width_comp,
            capstyle='butt', antialiased=True, zorder=2,
            label=self.comparison_driver,
        )
        lc_comp.set_array(self.throttle_comp)
        ax.add_collection(lc_comp)

        # Limits & aspect
        all_x = np.r_[X_ref, X_comp]
        all_y = np.r_[Y_ref, Y_comp]
        margin = 0.1 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_aspect('equal', adjustable='box')

        # Brake zones (optional)
        if self.highlight_brake_zones:
            br_ref = self.throttle_ref < self.brake_threshold
            br_comp = self.throttle_comp < self.brake_threshold
            if np.any(br_ref):
                ax.scatter(X_ref[br_ref], Y_ref[br_ref], color='cyan',
                           s=20, alpha=0.7, zorder=3, label='Brake zones')
            if np.any(br_comp):
                ax.scatter(X_comp[br_comp], Y_comp[br_comp], color='cyan',
                           s=20, alpha=0.7, zorder=3)

        # Sector markers (approx thirds)
        if self.annotate_sectors:
            coords_raw = np.c_[self.x_ref, self.y_ref]
            deltas = np.hypot(np.diff(coords_raw[:, 0]), np.diff(coords_raw[:, 1]))
            cumdist = np.insert(np.cumsum(deltas), 0, 0.0)
            total = cumdist[-1]
            for i, frac in enumerate([1 / 3, 2 / 3], start=1):
                target = total * frac
                idx = int(np.argmin(np.abs(cumdist - target)))
                pt = coords_raw[idx]
                rpt = self._rotate(pt.reshape(1, 2), angle=self._angle)[0]

                dx = np.gradient(self.x_ref)[idx]
                dy = np.gradient(self.y_ref)[idx]
                perp = np.array([-dy, dx], dtype=float)
                nrm = np.hypot(perp[0], perp[1]) or 1.0
                perp /= nrm
                off_pt = rpt + perp * (self.offset_distance * 0.5)

                ax.scatter(off_pt[0], off_pt[1], color='yellow', s=100,
                           edgecolor='black', linewidth=2, zorder=4, marker='s',
                           label=('Sectors' if i == 1 else None))
                ax.text(off_pt[0], off_pt[1] + 80, f"S{i}", color='white',
                        fontsize=10, ha='center', zorder=5)

        # Corner annotations
        if self.annotate_corners and hasattr(self.circuit_info, "corners"):
            for _, corner in self.circuit_info.corners.iterrows():
                base = np.array([corner['X'], corner['Y']], dtype=float)
                rpt = self._rotate(base.reshape(1, 2), angle=self._angle)[0]
                ax.scatter(rpt[0], rpt[1], color='white', s=150,
                           edgecolor='orange', linewidth=2, zorder=4)
                label = f"{corner['Number']}"
                letter = corner.get('Letter', '')
                if letter is not None and str(letter).strip() != '' and not pd.isna(letter):
                    label += str(letter)
                ax.text(rpt[0], rpt[1], label, ha='center', va='center',
                        color='black', fontsize=9, weight='bold', zorder=5)

        # Colorbar
        cbar = fig.colorbar(lc_ref, ax=ax, orientation='vertical',
                            pad=0.02, shrink=0.6, aspect=30)
        cbar.set_label('Throttle (%)', labelpad=10, fontsize=12, color='white')
        cbar.set_ticks([0.0, 50.0, 100.0])
        cbar.set_ticklabels(['0', '50', '100'])
        cbar.ax.yaxis.set_tick_params(labelsize=10, color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Legend
        legend = ax.legend(
            handles=[
                Line2D([0], [0], color='white', linewidth=6,
                       label=f'{self.reference_driver} (inner)'),
                Line2D([0], [0], color='white', linewidth=6,
                       label=f'{self.comparison_driver} (outer)'),
            ],
            loc='upper left', fontsize=12, facecolor='#333333',
            edgecolor='white', framealpha=0.9,
        )
        for text in legend.get_texts():
            text.set_color('white')

        # Title
        parts = []
        if self.event_name:
            title_loc = f"{self.event_name}"
            if self.year:
                title_loc += f" {self.year}"
            if self.session_name:
                title_loc += f" {self.session_name}"
            parts.append(title_loc)
        parts.append(f"{self.reference_driver} vs {self.comparison_driver}")
        ax.set_title(
            " — ".join(parts), fontsize=16, color='white', pad=25, weight='bold',
        )

        # Driver labels at polyline starts
        ax.text(
            X_ref[0], Y_ref[0], f'  {self.reference_driver}  ',
            ha='center', va='center', color='white', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8),
            zorder=10,
        )
        ax.text(
            X_comp[0], Y_comp[0], f'  {self.comparison_driver}  ',
            ha='center', va='center', color='white', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8),
            zorder=10,
        )

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, ax