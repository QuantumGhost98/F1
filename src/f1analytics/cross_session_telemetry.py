import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.acceleration import compute_acceleration, compute_total_acceleration
from f1analytics.lateral_acceleration import compute_lateral_acceleration
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
import warnings
from scipy.signal import savgol_filter
from f1analytics.colors_pilots import colors_pilots
from f1analytics.config import logger
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.plot_utils import (
    assign_colors, adjust_brightness, setup_dark_theme,
    add_branding,
)


class CrossSessionTelemetry:
    """
    Compare telemetry data across different sessions (e.g. year-over-year).
    Each entry in sessions is (session_obj, session_name, year, session_type).
    """
    def __init__(self, sessions):
        """
        sessions: list of tuples (session, session_name, year, session_type)
        """
        self.sessions = []
        for item in sessions:
            session, name, year, stype = item
            self.sessions.append({
                'session': session,
                'name': name,
                'year': year,
                'type': stype,
                'laps': session.laps,
                'circuit_info': session.get_circuit_info() if hasattr(session, "get_circuit_info") else None,
            })
        self.colors_pilots = colors_pilots

    class FastestLap:
        """
        Wraps a fastest-lap record, same as Telemetry.FastestLap but also stores session_key.
        """
        def __init__(self, lap, session_key=""):
            self.name = lap['Driver']
            self.session_key = session_key
            self.s1_time = lap['Sector1Time']
            self.s2_time = lap['Sector2Time']
            df = lap.get_car_data().add_distance()
            self.df = interpolate_dataframe(df)

        @property
        def sector_distances(self):
            d1 = self.df[self.df['Time'] <= self.s1_time]['Distance'].max()
            d2 = self.df[self.df['Time'] <= (self.s1_time + self.s2_time)]['Distance'].max()
            return d1, d2

    @staticmethod
    def adjust_brightness(color, factor):
        return adjust_brightness(color, factor)

    def assign_colors(self, driver_specs, driver_color_map=None, default_colors=None, fallback_shades=None):
        return assign_colors(
            driver_specs,
            driver_color_map=driver_color_map,
            default_colors=default_colors,
            fallback_shades=fallback_shades,
        )

    def compare_laps(self, session_drivers, channels=None, driver_color_map=None, save_path=None):
        """
        Compare laps across sessions.

        Parameters
        ----------
        session_drivers : list of (session_index, driver_code_or_dict)
            e.g. [(0, 'LEC'), (1, {'VER': 5})] or [(0, 'LEC'), (1, 'LEC')]
        channels : list of str, telemetry channels to plot
        driver_color_map : optional color overrides
        save_path : optional file path to save

        Returns
        -------
        (fig, axes)
        """
        default_channels = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'Total_Acc']
        user_provided_channels = channels is not None
        channels = channels or default_channels

        delta_aliases = {'delta', 'deltatime', 'Δ'}
        wants_delta = (
            len(session_drivers) > 1 and
            (not user_provided_channels or any(str(ch).lower() in delta_aliases for ch in channels))
        )
        channels = [ch for ch in channels if str(ch).lower() not in delta_aliases]
        effective_channels = channels.copy()
        if not user_provided_channels and 'Total_Acc' not in effective_channels:
            effective_channels.append('Total_Acc')

        units = {
            'Speed': 'km/h', 'Throttle': '%', 'Brake': '%',
            'RPM': 'rpm', 'nGear': '', 'DRS': '', 'Total_Acc': 'g',
        }

        # Build driver_specs and load data
        driver_specs = []
        laps = []
        lap_objs = []

        for sess_idx, driver_input in session_drivers:
            sess_info = self.sessions[sess_idx]
            session = sess_info['session']
            transformed = sess_info['laps'].copy()
            transformed.loc[:, "LapTime (s)"] = sess_info['laps']["LapTime"].dt.total_seconds()

            if isinstance(driver_input, dict):
                specs = normalize_driver_specs(driver_input, max_specs=1)
                spec = specs[0]
            elif isinstance(driver_input, str):
                spec = {'driver': driver_input, 'lap': 'fastest', 'display_name': driver_input}
            else:
                raise ValueError(f"Invalid driver_input: {driver_input}")

            # Append session info to display name
            session_label = f"{sess_info['name']} {sess_info['year']}"
            spec['display_name'] = f"{spec['display_name']} ({session_label})"

            driver_specs.append(spec)

            drv = spec['driver']
            lap_id = spec['lap']
            if lap_id == 'fastest':
                lap = transformed.pick_drivers(drv).pick_fastest()
            else:
                try:
                    lap = transformed.pick_drivers(drv).pick_laps(int(lap_id)).iloc[0]
                except Exception as e:
                    raise ValueError(f"Invalid lap selection for {drv}: {lap_id}") from e

            session_key = f"{sess_info['name']} {sess_info['year']}"
            fl = self.FastestLap(lap, session_key=session_key)
            fl.df = compute_total_acceleration(fl.df)
            if 'Total_Acceleration' in fl.df.columns:
                fl.df = fl.df.rename(columns={'Total_Acceleration': 'Total_Acc'})
            laps.append(fl)
            lap_objs.append(lap)

        display_names = [s['display_name'] for s in driver_specs]
        palette = assign_colors(
            driver_specs,
            driver_color_map=driver_color_map,
            default_colors=colors_pilots,
        )

        lap_times = [lap_obj['LapTime'].total_seconds() for lap_obj in lap_objs]
        baseline_idx = lap_times.index(min(lap_times))
        baseline_name = display_names[baseline_idx]
        s1_dist, s2_dist = laps[baseline_idx].sector_distances

        # Use circuit info from first session
        circuit_info = self.sessions[0]['circuit_info']
        corner_df = circuit_info.corners.copy().sort_values('Distance') if circuit_info else pd.DataFrame()

        n_plots = len(effective_channels) + (1 if wants_delta else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        setup_dark_theme(fig, axes)

        plot_idx = 0
        for ch in effective_channels:
            ax = axes[plot_idx]
            if ch not in laps[baseline_idx].df.columns:
                ax.set_visible(False)
                plot_idx += 1
                continue
            for lap, col, disp_name in zip(laps, palette, display_names):
                ax.plot(lap.df['Distance'], lap.df.get(ch, np.nan),
                        color=col, linestyle='-', label=f"{disp_name} {ch}")
            unit = units.get(ch, '')
            ax.set_ylabel(f"{ch} ({unit})" if unit else ch, color='white')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.tick_params(colors='white')
            ax.axvline(s1_dist, color='white', linestyle='--')
            ax.axvline(s2_dist, color='white', linestyle='--')

            zero_based = corner_df['Number'].min() == 0 if not corner_df.empty else False
            for _, row in corner_df.iterrows():
                num = int(row['Number']) + (1 if zero_based else 0)
                letter = ''
                if 'Letter' in row and pd.notna(row['Letter']) and str(row['Letter']).strip():
                    letter = str(row['Letter']).strip()
                label = f"{num}{letter}"
                ax.text(row['Distance'], ax.get_ylim()[0], label,
                        color='white', fontsize=8, ha='center', va='bottom')
            plot_idx += 1

        if wants_delta:
            ax_dt = axes[-1]
            ref_lap = lap_objs[baseline_idx]
            for idx, comp_lap in enumerate(lap_objs):
                if idx == baseline_idx:
                    continue
                delta_series, ref_tel, comp_tel = delta_time(ref_lap, comp_lap)
                ax_dt.plot(ref_tel['Distance'], delta_series,
                        color=palette[idx], linestyle='-',
                        label=f"Δ ({display_names[idx]} - {baseline_name})")
            ax_dt.axvline(s1_dist, color='white', linestyle='--')
            ax_dt.axvline(s2_dist, color='white', linestyle='--')
            ax_dt.set_ylabel('Δ Time (s)', color='white')
            benchmark_color = palette[baseline_idx]
            ax_dt.axhline(0, color=benchmark_color, linestyle='--', linewidth=1.2)
            ax_dt.grid(True, linestyle='--', linewidth=0.5)
            ax_dt.tick_params(colors='white')
            for _, row in corner_df.iterrows():
                num = int(row['Number']) + (1 if zero_based else 0)
                letter = ''
                if 'Letter' in row and pd.notna(row['Letter']) and str(row['Letter']).strip():
                    letter = str(row['Letter']).strip()
                label = f"{num}{letter}"
                ax_dt.text(row['Distance'], ax_dt.get_ylim()[0], label,
                        color='white', fontsize=8, ha='center', va='bottom')
            ax_dt.xaxis.set_major_locator(plt.MultipleLocator(500))
            ax_dt.xaxis.set_minor_locator(plt.MultipleLocator(100))
            ax_dt.legend(loc='upper right', title=f"Benchmark: {baseline_name}")

        # Annotations
        labels = []
        for i, (name, secs) in enumerate(zip(display_names, lap_times)):
            if pd.isna(secs):
                label = f"{name}: NaN"
            else:
                mins = int(secs // 60)
                rem = secs - mins * 60
                label = f"{name}: {mins}:{rem:06.3f}"
            labels.append(label)

        fig.text(0.02, 0.98, "\n".join(labels), ha='left', va='top',
                color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=4))

        title = "Cross-Session Comparison"
        fig.suptitle(title, color='white')
        fig.subplots_adjust(top=0.92)
        plt.tight_layout(rect=[0, 0, 0.90, 0.94])

        for ax in axes:
            pos = ax.get_position()
            rect = plt.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                transform=fig.transFigure,
                facecolor='none', edgecolor='white', linewidth=1.2
            )
            fig.patches.append(rect)

        add_branding(fig, text_pos=(0.9, 0.96), logo_pos=[0.80, 0.90, 0.06, 0.06])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        plt.show()
        return fig, axes