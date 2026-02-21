"""
Corner metric comparator — min-speed and min-throttle analysis per corner.

CornerMinSpeed:  original API — pass corner_idxs, auto-discovers drivers.
CornerMetricComparator: generalized version accepting explicit driver dict.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.corner_utils import corner_identifier_to_index, corner_label
from f1analytics.plot_utils import assign_colors_simple, setup_dark_theme, add_branding


# ── Original CornerMinSpeed class (preserved API) ─────────────────────────────

class CornerMinSpeed:
    """
    Analyze minimum speed across all drivers for a specific corner or group of corners.

    This is the **original** interface:
        CornerMinSpeed(session, name, year, type, corner_idxs=[11], before=30, after=30, n_drivers=10)
    """

    def __init__(self, session, session_name: str, year: int, session_type: str,
                 corner_idxs, before=100, after=100, n_drivers=None):
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.before = before
        self.after = after
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        self.laps = session.laps
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()

        # Resolve corner identifiers to zero-based indices
        if isinstance(corner_idxs, (int, str)):
            self.corner_list = [corner_identifier_to_index(self.circuit_info, corner_idxs)]
        elif hasattr(corner_idxs, '__iter__'):
            self.corner_list = [corner_identifier_to_index(self.circuit_info, x) for x in corner_idxs]
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
        self.min_speed = {}
        self._compute_min_speed()

    def _compute_min_speed(self):
        corners = self.circuit_info.corners['Distance'].values
        dists = [corners[idx] for idx in self.corner_list]
        start = min(dists) - self.before
        end = max(dists) + self.after

        for d in self.drivers:
            try:
                lap = self.session.laps.pick_drivers(d).pick_fastest()
                tel = lap.get_car_data().add_distance()
                tel = interpolate_dataframe(tel)
                df = tel[(tel['Distance'] >= start) & (tel['Distance'] <= end)]
                self.min_speed[d] = df['Speed'].min() if not df.empty else None
            except Exception as e:
                logger.warning("Skipping %s: %s", d, e)
                self.min_speed[d] = None

    def plot(self, save_path=None):
        """Plot min-speed bar chart. Returns (fig, ax)."""
        vals = {d: v for d, v in self.min_speed.items() if v is not None}
        if not vals:
            logger.warning("No valid speed data to plot.")
            return None, None
        sorted_vals = dict(sorted(vals.items(), key=lambda item: item[1]))

        fig, ax = plt.subplots(figsize=(14, 7))
        setup_dark_theme(fig, [ax])

        bars = ax.bar(
            sorted_vals.keys(),
            sorted_vals.values(),
            color=[self.colors[d] for d in sorted_vals.keys()],
            edgecolor='white', linewidth=0.3
        )

        ax.set_ylabel("Minimum Speed (km/h)", color='white', fontsize=11)

        # Build corner label string
        if len(self.corner_list) == 1:
            clabel = f"Corner {corner_label(self.circuit_info, self.corner_list[0])}"
        else:
            sorted_idxs = sorted(self.corner_list)
            is_contiguous = all(sorted_idxs[i] + 1 == sorted_idxs[i + 1] for i in range(len(sorted_idxs) - 1))
            if is_contiguous:
                clabel = f"Corners {corner_label(self.circuit_info, sorted_idxs[0])}-{corner_label(self.circuit_info, sorted_idxs[-1])}"
            else:
                clabel = "Corners " + ",".join(corner_label(self.circuit_info, i) for i in sorted_idxs)

        parts = [f"{self.session_name} {self.year}"]
        if self.session_type:
            parts.append(self.session_type)
        parts.append(f"Minimum Speed — {clabel}")
        ax.set_title(" — ".join(parts), color='white', fontsize=13)
        ax.set_ylim(0, max(sorted_vals.values()) + 20)
        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f"{height:.0f} km/h", ha='center', va='bottom', color='white', fontsize=9)

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        return fig, ax


# ── Generalized CornerMetricComparator (new API) ──────────────────────────────

class CornerMetricComparator:
    """
    Compare a per-corner metric (min speed, min throttle, etc.) across explicit drivers.

    New API:
        CornerMetricComparator(session, ..., drivers={'LEC': 'fastest', 'VER': 14}, metric='Speed', mode='min')
    """

    def __init__(self, session_name, year, session_type,
                 session=None, drivers=None, laps=None,
                 metric='Speed', mode='min', margin=50):
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.metric = metric
        self.mode = mode
        self.margin = margin

        self.driver_specs = normalize_driver_specs(drivers=drivers, laps=laps, max_specs=4)
        self.drivers = [s['driver'] for s in self.driver_specs]
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.palette = assign_colors_simple(self.drivers)

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

        self.telemetry = {}
        self._load_laps()

    def _load_laps(self):
        for spec in self.driver_specs:
            d = spec['driver']
            lap_id = spec['lap']
            disp = spec['display_name']
            sess = spec.get('session') or self.session

            drv_laps = sess.laps.pick_drivers(d)
            if lap_id == 'fastest':
                lap = drv_laps.pick_fastest()
            else:
                try:
                    lap_num = int(lap_id)
                    lap = drv_laps.pick_laps(lap_num).iloc[0]
                except Exception as e:
                    raise ValueError(f"Invalid lap selection for {d}: {lap_id}") from e
            df = lap.get_car_data().add_distance()
            df = interpolate_dataframe(df)
            self.telemetry[disp] = df

    def compute(self):
        """Return DataFrame with one row per corner and one column per driver."""
        corners_df = self.circuit_info.corners
        results = {'Corner': []}
        for d in self.display_names:
            results[d] = []

        for apex_idx in range(len(corners_df)):
            apex_dist = corners_df.iloc[apex_idx]['Distance']
            results['Corner'].append(corner_label(self.circuit_info, apex_idx))
            for d in self.display_names:
                df = self.telemetry[d]
                window = df[(df['Distance'] >= apex_dist - self.margin) & (df['Distance'] <= apex_dist + self.margin)]
                if window.empty or self.metric not in window.columns:
                    results[d].append(np.nan)
                elif self.mode == 'min':
                    results[d].append(window[self.metric].min())
                else:
                    results[d].append(window[self.metric].max())

        return pd.DataFrame(results)

    def plot(self, save_path=None):
        """Plot grouped bar chart of per-corner metrics. Returns (fig, ax)."""
        df = self.compute()
        n_corners = len(df)
        n_drivers = len(self.display_names)
        x = np.arange(n_corners)
        bar_w = 0.8 / n_drivers

        fig, ax = plt.subplots(figsize=(max(12, n_corners * 0.8), 6))
        setup_dark_theme(fig, [ax])

        for i, (drv, col) in enumerate(zip(self.display_names, self.palette)):
            ax.bar(x + i * bar_w, df[drv], width=bar_w,
                   label=drv, color=col, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + bar_w * (n_drivers - 1) / 2)
        ax.set_xticklabels(df['Corner'], color='white')

        mode_label = self.mode.capitalize()
        unit = 'km/h' if self.metric == 'Speed' else '%' if self.metric == 'Throttle' else ''
        y_label = f"{mode_label} {self.metric}" + (f" ({unit})" if unit else "")

        ax.set_ylabel(y_label, color='white')
        parts = [f"{self.session_name} {self.year}"]
        if self.session_type:
            parts.append(self.session_type)
        parts.append(f"{mode_label} {self.metric} per Corner")
        ax.set_title(" — ".join(parts), color='white', fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
        ax.tick_params(colors='white')

        plt.tight_layout()
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        return fig, ax