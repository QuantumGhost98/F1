import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
import warnings
from scipy.signal import savgol_filter
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.corner_utils import (
    corner_identifier_to_index, corner_label, indices_between,
    compress_indices_to_ranges, format_corner_label_list, resolve_corner_idxs,
)
from f1analytics.plot_utils import (
    assign_colors_simple, setup_dark_theme, add_branding, finalize_plot,
)

class CornerAnalysis:
    """
    Analyze telemetry for one or more corners (or a corner range).
    Supports flexible channel selection (e.g., include/exclude 'Acc') and optional delta-time
    (default when comparing multiple drivers unless suppressed; can be triggered explicitly
    via 'Delta', 'DeltaTime', or 'Δ').
    corner_idxs now also supports ranges like "3-5" and nested lists/tuples like [3,4] or (7, 9) which are expanded inclusively.
    """

    def __init__(self, session_name: str, year: int, session_type: str,
                 session=None, drivers=None, laps=None,
                 corner_idxs=None, before=50, after=50):
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.before = before
        self.after = after

        # Normalize drivers
        self.driver_specs = normalize_driver_specs(drivers=drivers, laps=laps, max_specs=4)
        self.display_names = [s['display_name'] for s in self.driver_specs]
        self.drivers = [s['driver'] for s in self.driver_specs]

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

        # Resolve corners using the shared utility
        self.corner_list = resolve_corner_idxs(self.circuit_info, corner_idxs)
        self.start_idx = min(self.corner_list)
        self.end_idx = max(self.corner_list)

        self.telemetry = {}
        self.lap_objs = {}
        self._load_data()
        self.palette = self._assign_colors()

    # Backward-compatible wrappers delegating to shared utilities
    def _corner_identifier_to_index(self, label):
        return corner_identifier_to_index(self.circuit_info, label)

    def _corner_label(self, apex_idx):
        return corner_label(self.circuit_info, apex_idx)

    def _indices_between(self, a_idx, b_idx):
        return indices_between(a_idx, b_idx)

    def _compress_indices_to_ranges(self, idx_list):
        return compress_indices_to_ranges(idx_list)

    def _format_corner_label_list(self, idx_list):
        return format_corner_label_list(self.circuit_info, idx_list)

    def _load_data(self):
        """Load and interpolate telemetry for each driver based on lap selection."""
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
            self.lap_objs[disp] = lap

    def _assign_colors(self):
        return assign_colors_simple(self.drivers)

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

    def plot_all(self, channels=None, save_path=None):
        """
        Plot requested channels around the corner(s). Acceptable channel names include
    'Speed', 'Throttle', 'Brake', and delta aliases ('Delta', 'DeltaTime', 'Δ').
    If multiple drivers are compared, delta-time is shown by default unless channels
    are provided without a delta alias.

        Returns (fig, axes).
        """
        default_channels = ['Speed', 'Throttle', 'Brake']
        user_provided_channels = channels is not None
        channels = channels or default_channels

        # Delta-time logic
        delta_aliases = {'delta', 'deltatime', 'Δ'}
        wants_delta_token = any(str(ch).lower() in delta_aliases for ch in channels)
        channels = [ch for ch in channels if str(ch).lower() not in delta_aliases]

        # Only show delta if there are 2+ drivers
        wants_delta = (len(self.display_names) > 1) and (wants_delta_token or not user_provided_channels)

        # Do we also want the throttle scatter alt view?
        include_throttle_scatter = any(str(ch).lower() == 'throttle' for ch in channels)

        n_line = len(channels)
        n_extra = (1 if wants_delta else 0) + (1 if include_throttle_scatter else 0)
        total_plots = max(1, n_line + n_extra)

        fig, axs = plt.subplots(total_plots, 1, figsize=(14, 3.5 * total_plots), sharex=False)

        # normalize axs into a list
        if isinstance(axs, np.ndarray):
            axes_list = axs.ravel().tolist()
        else:
            axes_list = [axs]

        setup_dark_theme(fig, axes_list)

        # Build title text for single or multiple corners (compact: e.g., 1,3-4,6)
        if len(self.corner_list) == 1:
            corner_label_str = f"Corner {self._corner_label(self.corner_list[0])}"
        else:
            corner_label_str = f"Corners {self._format_corner_label_list(self.corner_list)}"

        parts = [self.session_name, str(self.year)]
        if self.session_type:
            parts.append(self.session_type)
        parts.append(corner_label_str)
        title = " — ".join([" ".join(parts[:2 + bool(self.session_type)]), corner_label_str])
        fig.suptitle(title, color='white', fontsize=14)
        fig.subplots_adjust(top=0.92)

        plot_idx = 0

        # Line plots for requested channels
        for ch in channels:
            ax = axes_list[plot_idx]
            for d, col in zip(self.display_names, self.palette):
                dfc = self.get_corner_df(d)
                if ch not in dfc.columns:
                    continue
                ax.plot(dfc['Distance'], dfc[ch], color=col, label=f"{d} {ch}")
            ax.set_ylabel(ch, color='white')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
            ax.tick_params(colors='white')
            plot_idx += 1

        # Δ time subplot (only if 2+ drivers)
        if wants_delta:
            ax_dt = axes_list[plot_idx]
            # choose baseline as the fastest lap
            lap_times = [self.lap_objs[d]['LapTime'].total_seconds() for d in self.display_names]
            baseline_idx = lap_times.index(min(lap_times))
            baseline_driver = self.display_names[baseline_idx]
            ref_lap = self.lap_objs[baseline_driver]

            corners = self.circuit_info.corners['Distance'].values
            start_dist = corners[self.start_idx] - self.before
            end_dist = corners[self.end_idx] + self.after

            for comp_driver, col in zip(self.display_names, self.palette):
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
            if wants_delta_token and len(self.display_names) == 1:
                logger.info("Delta requested but only one driver provided; skipping Δ plot.")

        # Throttle scatter (optional)
        if include_throttle_scatter:
            ax_throttle_alt = axes_list[plot_idx]
            for d, col in zip(self.display_names, self.palette):
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

        # Signature & logo
        plt.tight_layout(rect=[0, 0, 0.95, 0.94])
        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, axes_list