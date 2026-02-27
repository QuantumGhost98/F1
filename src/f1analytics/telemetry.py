import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.acceleration import compute_acceleration
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
import warnings
from scipy.signal import savgol_filter
from f1analytics.palette import driver_colors as colors_pilots
from f1analytics.config import logger
from f1analytics.driver_utils import normalize_driver_specs
from f1analytics.plot_utils import (
    assign_colors, adjust_brightness, setup_dark_theme,
    add_branding, finalize_plot,
)
from f1analytics.corner_utils import corner_label as _corner_label_util



class Telemetry:
    def __init__(self, session_name: str, year: int, session_type: str, *, session=None):
        """
        session_name: e.g. "Hungary Grand Prix"
        year: e.g. 2025
        session_type: e.g. "Q", "R"
        session: loaded FastF1 session object (optional for cross-session mode)
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        if session is not None:
            self.laps = session.laps
            self.transformed_laps = self.laps.copy()
            self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()
            self.weather = session.weather_data
            self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        else:
            self.laps = None
            self.transformed_laps = None
            self.weather = None
            self.circuit_info = None
        self.colors_pilots = colors_pilots

    class FastestLap:
        """
        Wraps a fastest-lap record and provides convenience accessors.
        Expects lap to have ['Driver','Sector1Time','Sector2Time'] and get_car_data().
        """
        def __init__(self, lap):
            self.name = lap['Driver']
            self.s1_time = lap['Sector1Time']
            self.s2_time = lap['Sector2Time']
            df = lap.get_car_data().add_distance()
            self.df = interpolate_dataframe(df)

        @property
        def sector_distances(self):
            d1 = self.df[self.df['Time'] <= self.s1_time]['Distance'].max()
            d2 = self.df[self.df['Time'] <= (self.s1_time + self.s2_time)]['Distance'].max()
            return d1, d2


    # Keep as static methods for backward compat (they were class-level before)
    @staticmethod
    def adjust_brightness(color, factor):
        """
        Lighten (factor>1) or darken (factor<1) an RGB color.
        """
        return adjust_brightness(color, factor)

    def assign_colors(self, driver_specs, driver_color_map=None, default_colors=None, fallback_shades=None):
        """
        Returns a list of colors for each spec in driver_specs (order preserved).
        Priority: driver_color_map[display_name] > driver_color_map[driver] > default_colors[driver] > 'white'
        Applies fallback shades when base color repeats; if exhausted, auto-adjust brightness.
        """
        return assign_colors(
            driver_specs,
            driver_color_map=driver_color_map,
            default_colors=default_colors,
            fallback_shades=fallback_shades,
        )


    def compare_laps(self, drivers=None, laps=None, channels=None, session_label="",
                     driver_color_map=None, save_path=None):
        """
        Compare up to three laps (can be from the same or different sessions).

        Parameters:
        - drivers: flexible specification (single-session mode):
            * dict like {'LEC': 'fastest', 'VER': 5}
            * list of strings: ['VER', 'LEC'] (fastest laps)
        - laps: cross-session mode:
            * [(session1, 'LEC', 'fastest', 'Day 5'), (session2, 'LEC', 'fastest', 'Day 6')]
        - channels: list of telemetry fields to plot
        - session_label: optional label for title
        - driver_color_map: optional color override
        - save_path: optional file path to save the figure

        Returns:
        - (fig, axes) tuple
        """
        # Normalize into driver_specs using the shared utility
        driver_specs = normalize_driver_specs(drivers=drivers, laps=laps)

        display_names = [spec['display_name'] for spec in driver_specs]

        default_channels = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'Long_Acc']
        user_provided_channels = channels is not None
        channels = channels or default_channels

        delta_aliases = {'delta', 'deltatime', 'Δ'}
        wants_delta = (
            len(driver_specs) > 1 and
            (not user_provided_channels or any(str(ch).lower() in delta_aliases for ch in channels))
        )
        channels = [ch for ch in channels if str(ch).lower() not in delta_aliases]
        effective_channels = channels.copy()
        if not user_provided_channels and 'Long_Acc' not in effective_channels:
            effective_channels.append('Long_Acc')

        units = {
            'Speed': 'km/h',
            'Throttle': '%',
            'Brake': '%',
            'RPM': 'rpm',
            'nGear': '',
            'DRS': '',
            'Long_Acc': 'g',
        }

        # Load data — per-spec session for cross-session support
        loaded_laps = []
        lap_objs = []
        per_spec_weather = []  # (avg_air, avg_track) per spec
        first_session = None
        for spec in driver_specs:
            sess = spec.get('session') or self.session
            if first_session is None:
                first_session = sess
            driver = spec['driver']
            lap_id = spec['lap']

            drv_laps = sess.laps.pick_drivers(driver)
            if lap_id == 'fastest':
                lap = drv_laps.pick_fastest()
            else:
                try:
                    lap = drv_laps.pick_laps(int(lap_id)).iloc[0]
                except Exception as e:
                    raise ValueError(f"Invalid lap selection for {driver}: {lap_id}") from e

            fl = self.FastestLap(lap)
            fl.df = compute_acceleration(fl.df)
            if 'Acceleration' in fl.df.columns:
                fl.df = fl.df.rename(columns={'Acceleration': 'Long_Acc'})
            loaded_laps.append(fl)
            lap_objs.append(lap)

            # Collect per-spec weather from session
            w = self.weather
            if w is None and sess is not None:
                try:
                    w = sess.weather_data
                except Exception:
                    w = None
            if w is not None:
                per_spec_weather.append((w['AirTemp'].mean(), w['TrackTemp'].mean()))
            else:
                per_spec_weather.append((None, None))

        # Use circuit_info from self or first session
        circuit_info = self.circuit_info
        if circuit_info is None and first_session:
            circuit_info = first_session.get_circuit_info() if hasattr(first_session, 'get_circuit_info') else None

        lap_times = [lap['LapTime'].total_seconds() for lap in lap_objs]
        baseline_idx = lap_times.index(min(lap_times))
        baseline_name = display_names[baseline_idx]

        s1_dist, s2_dist = loaded_laps[baseline_idx].sector_distances
        corner_df = circuit_info.corners.copy().sort_values('Distance') if circuit_info else pd.DataFrame()

        # Color assignment via shared utility
        palette = assign_colors(
            driver_specs,
            driver_color_map=driver_color_map,
            default_colors=globals().get('colors_pilots', None)
        )

        # Plot setup
        n_plots = len(effective_channels) + (1 if wants_delta else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        setup_dark_theme(fig, axes)

        plot_idx = 0
        for ch in effective_channels:
            ax = axes[plot_idx]
            if ch not in loaded_laps[baseline_idx].df.columns:
                ax.set_visible(False)
                plot_idx += 1
                continue
            for lap, col, disp_name in zip(loaded_laps, palette, display_names):
                ax.plot(lap.df['Distance'], lap.df.get(ch, np.nan),
                        color=col, linestyle='-', label=f"{disp_name} {ch}")
            unit = units.get(ch, '')
            ax.set_ylabel(f"{ch} ({unit})" if unit else ch, color='white')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
            ax.tick_params(colors='white')
            ax.axvline(s1_dist, color='white', linestyle='--')
            ax.axvline(s2_dist, color='white', linestyle='--')

            zero_based = corner_df['Number'].min() == 0
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
            # Add sector lines
            ax_dt.axvline(s1_dist, color='white', linestyle='--')
            ax_dt.axvline(s2_dist, color='white', linestyle='--')

            ax_dt.set_ylabel('Δ Time (s)', color='white')
            benchmark_color = palette[baseline_idx]
            ax_dt.axhline(0, color=benchmark_color, linestyle='--', linewidth=1.2)
            ax_dt.grid(True, linestyle='--', linewidth=0.5)
            ax_dt.tick_params(colors='white')
            zero_based = corner_df['Number'].min() == 0
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

        # Annotations — each lap shows its own session's weather
        labels = []
        for i, (name, secs) in enumerate(zip(display_names, lap_times)):
            if pd.isna(secs):
                label = f"{name}: NaN"
            else:
                mins = int(secs // 60)
                rem = secs - mins * 60
                label = f"{name}: {mins}:{rem:06.3f}"
                air_t, track_t = per_spec_weather[i]
                if air_t is not None and track_t is not None:
                    label += f"   AIR: {air_t:.1f}°C  TRACK: {track_t:.1f}°C"
            labels.append(label)

        fig.text(0.02, 0.98, "\n".join(labels), ha='left', va='top',
                color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=4))

        title = (f"{self.session_name} {self.year} {session_label}"
                if session_label else f"{self.session_name} {self.year}")
        fig.suptitle(title, color='white', fontsize=14)
        fig.subplots_adjust(top=0.92)
        plt.tight_layout(rect=[0, 0, 0.95, 0.94])
        # Add white contour around each subplot
        for ax in axes:
            pos = ax.get_position()
            rect = plt.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                transform=fig.transFigure,
                facecolor='none',
                edgecolor='white',
                linewidth=0.8,
                alpha=0.6
            )
            fig.patches.append(rect)

        add_branding(fig, text_pos=(0.99, 0.96), logo_pos=[0.90, 0.92, 0.05, 0.05])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info("Saved plot to %s", save_path)

        return fig, axes