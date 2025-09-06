# f1_analytics/dual_throttle_compare.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class DualThrottleComparisonVisualizer:
    """
    Concentric/parallel throttle comparison: two drivers drawn as
    parallel heatmap traces around the full lap.

      • Reference driver = inner line (offset -d)
      • Comparison driver = outer line (offset +d)
      • Both colored by the SAME throttle colormap (0–100 %)

    Pass:
      laps:          FastF1 Laps (e.g. session.laps)
      circuit_info:  session.get_circuit_info()
      reference_driver: 'VER', 'LEC', ...
      comparison_driver: 'NOR', 'ALO', ...
    """
    laps: any
    circuit_info: any
    reference_driver: str
    comparison_driver: str

    # appearance / options
    offset_distance: float = 200.0         # offset between parallel lines (track coord units)
    annotate_corners: bool = True
    annotate_sectors: bool = False
    highlight_brake_zones: bool = False
    brake_threshold: float = 20.0          # throttle (%)

    # optional title metadata
    event_name: Optional[str] = None
    year: Optional[int] = None
    session_name: Optional[str] = None

    line_width: float = 8.0                # used when use_speed_width=False
    use_speed_width: bool = True           # scale line width by speed
    use_bgr_scale: bool = True             # True -> Blue→Green→Red, False -> 'viridis'

    # internal fields (filled in __post_init__)
    x_ref: np.ndarray = field(init=False)
    y_ref: np.ndarray = field(init=False)
    throttle_ref: np.ndarray = field(init=False)
    speed_ref: np.ndarray = field(init=False)

    x_comp: np.ndarray = field(init=False)
    y_comp: np.ndarray = field(init=False)
    throttle_comp: np.ndarray = field(init=False)
    speed_comp: np.ndarray = field(init=False)

    _angle: float = field(init=False)
    _lap_ref: any = field(init=False)
    _lap_comp: any = field(init=False)

    def __post_init__(self):
        # Extract fastest laps
        lap_ref = self.laps.pick_drivers([self.reference_driver]).pick_fastest()
        lap_comp = self.laps.pick_drivers([self.comparison_driver]).pick_fastest()

        # Use FastF1 telemetry (already merged & sampled)
        tel_ref = lap_ref.telemetry
        tel_comp = lap_comp.telemetry

        self.x_ref = tel_ref['X'].to_numpy()
        self.y_ref = tel_ref['Y'].to_numpy()
        self.throttle_ref = tel_ref['Throttle'].to_numpy()
        self.speed_ref = tel_ref['Speed'].to_numpy()

        self.x_comp = tel_comp['X'].to_numpy()
        self.y_comp = tel_comp['Y'].to_numpy()
        self.throttle_comp = tel_comp['Throttle'].to_numpy()
        self.speed_comp = tel_comp['Speed'].to_numpy()

        # Rotation (FastF1 gives degrees)
        self._angle = float(self.circuit_info.rotation) / 180.0 * np.pi

        self._lap_ref = lap_ref
        self._lap_comp = lap_comp

    @staticmethod
    def _rotate(xy: np.ndarray, *, angle: float) -> np.ndarray:
        """Rotate XY coordinates by +angle (radians)."""
        mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
        return xy.dot(mat)

    @staticmethod
    def _parallel_offset(x: np.ndarray, y: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
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

    def plot(self, figsize=(14, 10)):
        # Figure / axes
        fig, ax = plt.subplots(figsize=figsize, facecolor='#222222')
        ax.set_facecolor('#222222')
        ax.axis('off')

        # Rotate coords
        rot_ref = self._rotate(np.c_[self.x_ref, self.y_ref], angle=self._angle)
        rot_comp = self._rotate(np.c_[self.x_comp, self.y_comp], angle=self._angle)

        # Parallel offsets (inner = negative, outer = positive)
        X_ref, Y_ref = self._parallel_offset(rot_ref[:, 0], rot_ref[:, 1], -self.offset_distance)
        X_comp, Y_comp = self._parallel_offset(rot_comp[:, 0], rot_comp[:, 1], +self.offset_distance)

        # Build line segments
        def _segments(x, y):
            pts = np.c_[x, y].reshape(-1, 1, 2)
            return np.concatenate([pts[:-1], pts[1:]], axis=1)

        segs_ref = _segments(X_ref, Y_ref)
        segs_comp = _segments(X_comp, Y_comp)

        # Colormap & normalization for throttle (0..100 %)
        cmap = self._choose_cmap()
        vmin, vmax = 0.0, 100.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

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
        lc_ref = LineCollection(segs_ref, cmap=cmap, norm=norm, linewidths=width_ref,
                                capstyle='butt', antialiased=True, zorder=1,
                                label=f'{self.reference_driver}')
        lc_ref.set_array(self.throttle_ref)
        ax.add_collection(lc_ref)

        lc_comp = LineCollection(segs_comp, cmap=cmap, norm=norm, linewidths=width_comp,
                                 capstyle='butt', antialiased=True, zorder=2,
                                 label=f'{self.comparison_driver}')
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
                ax.scatter(X_ref[br_ref], Y_ref[br_ref], color='cyan', s=20, alpha=0.7, zorder=3, label='Brake zones')
            if np.any(br_comp):
                ax.scatter(X_comp[br_comp], Y_comp[br_comp], color='cyan', s=20, alpha=0.7, zorder=3)

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

                # Offset marker halfway between lines
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

        # Corner annotations (optional)
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

        # Colorbar for throttle (0..100)
        cbar = fig.colorbar(lc_ref, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=30)
        cbar.set_label('Throttle (%)', labelpad=10, fontsize=12, color='white')
        mid = 50.0
        cbar.set_ticks([0.0, mid, 100.0])
        cbar.set_ticklabels(['0', '50', '100'])
        cbar.ax.yaxis.set_tick_params(labelsize=10, color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Legend
        from matplotlib.lines import Line2D
        legend = ax.legend(
            handles=[
                Line2D([0], [0], color='white', linewidth=6, label=f'{self.reference_driver} (inner)'),
                Line2D([0], [0], color='white', linewidth=6, label=f'{self.comparison_driver} (outer)'),
            ],
            loc='upper left', fontsize=12, facecolor='#333333', edgecolor='white', framealpha=0.9
        )
        for text in legend.get_texts():
            text.set_color('white')

        # Title
        title = f"THROTTLE COMPARE — {self.reference_driver} vs {self.comparison_driver}"
        if self.event_name and self.year and self.session_name:
            title = f"{self.event_name} {self.year} {self.session_name} — {self.reference_driver} vs {self.comparison_driver}"
        ax.set_title(title, fontsize=18, color='white', pad=25, weight='bold')

        # Driver labels at polyline starts
        ax.text(X_ref[0], Y_ref[0], f'  {self.reference_driver}  ',
                ha='center', va='center', color='white', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8), zorder=10)
        ax.text(X_comp[0], Y_comp[0], f'  {self.comparison_driver}  ',
                ha='center', va='center', color='white', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8), zorder=10)

        plt.tight_layout()
        plt.show()

    def save(self, path: str, figsize=(14, 10)):
        """Render and save to a file path."""
        self.plot(figsize=figsize)
        plt.savefig(path, dpi=200, bbox_inches='tight')