import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
from f1analytics.acceleration import compute_acceleration, compute_total_acceleration
from f1analytics.lateral_acceleration import compute_lateral_acceleration
from f1analytics.timedelta_to_seconds import timedelta_to_seconds
import warnings
from scipy.signal import savgol_filter
import sys
from f1analytics.colors_pilots import colors_pilots



class Telemetry:
    def __init__(self, session, session_name: str, year: int, session_type: str):
        """
        session: loaded FastF1 session object
        session_name: e.g. "Hungary Grand Prix"
        year: e.g. 2025
        session_type: e.g. "Q", "R"
        """
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.laps = session.laps
        # Transform the laps in total seconds
        self.transformed_laps = self.laps.copy()
        self.transformed_laps.loc[:, "LapTime (s)"] = self.laps["LapTime"].dt.total_seconds()
        self.weather = session.weather_data
        self.circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
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



    def adjust_brightness(color, factor):
        """
        Lighten (factor>1) or darken (factor<1) an RGB color.
        """
        try:
            rgb = np.array(mcolors.to_rgb(color))
            # scale and clip
            adjusted = np.clip(rgb * factor, 0, 1)
            return mcolors.to_hex(adjusted)
        except Exception:
            return color  # fallback

    def assign_colors(self, driver_specs, driver_color_map=None, default_colors=None, fallback_shades=None):
        """
        Returns a list of colors for each spec in driver_specs (order preserved).
        Priority: driver_color_map[display_name] > driver_color_map[driver] > default_colors[driver] > 'white'
        Applies fallback shades when base color repeats; if exhausted, auto-adjust brightness.
        """
        if default_colors is None:
            default_colors = {}
        if fallback_shades is None:
            fallback_shades = {
                'red': ['white', 'lightcoral'],
                'blue': ['cyan', 'lightblue'],
                'orange': ['white', 'wheat'],
                'grey': ['white', 'silver'],
                'green': ['lime', 'springgreen'],
                'pink': ['violet', 'lightpink'],
                'olive': ['khaki'],
                'navy': ['skyblue'],
                '#9932CC': ['plum'],
                'lime': ['yellowgreen']
            }

        used = {}
        palette = []
        for spec in driver_specs:
            driver = spec['driver']
            display = spec['display_name']

            # Determine base color
            base_color = None
            if driver_color_map:
                base_color = driver_color_map.get(display, driver_color_map.get(driver))
            if base_color is None:
                base_color = default_colors.get(driver, 'white')

            count = used.get(base_color, 0)
            if count == 0:
                color = base_color
            else:
                # try fallback shades first
                alternates = fallback_shades.get(base_color, [])
                if count - 1 < len(alternates):
                    color = alternates[count - 1]
                else:
                    # auto adjust brightness for further duplicates
                    # alternate lighten/darken
                    factor = 1 + 0.2 * ((count - len(alternates)) % 2) * (1 if ((count - len(alternates)) // 2) % 2 == 0 else -1)
                    color = self.adjust_brightness(base_color, factor)
            used[base_color] = count + 1
            palette.append(color)

        return palette


    def compare_laps(self, drivers, channels=None, session_label="", driver_color_map=None):
        """
        Compare up to three laps (can be from the same or different drivers).

        Parameters:
        - drivers: flexible specification:
            * dict like {'LEC': 'fastest', 'VER': 5}
            * dict where value is list: {'LEC': ['fastest', 4]} to get two LEC laps
            * list of strings: ['VER', 'LEC'] (fastest laps)
            * list of tuples: [('VER', 12), ('VER', 'fastest')]
            * mix of above
        - channels: list of telemetry fields to plot (can include 'Delta', 'Δ', etc.)
        - session_label: optional label for title
        - driver_color_map: optional override; keys can be driver codes or display names like "LEC_4"
        """
        # Normalize into driver_specs
        driver_specs = []  # each: {'driver':..., 'lap':..., 'display_name':...}
        if isinstance(drivers, dict):
            for drv, lap_sel in drivers.items():
                if isinstance(lap_sel, (list, tuple)):
                    for sel in lap_sel:
                        if sel == 'fastest':
                            name = drv
                        else:
                            name = f"{drv}_{sel}"
                        driver_specs.append({'driver': drv, 'lap': sel, 'display_name': name})
                else:
                    name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                    driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
        elif isinstance(drivers, (list, tuple)):
            for entry in drivers:
                if isinstance(entry, str):
                    driver_specs.append({'driver': entry, 'lap': 'fastest', 'display_name': entry})
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    drv, lap_sel = entry
                    if lap_sel == 'fastest':
                        name = drv
                    else:
                        name = f"{drv}_{lap_sel}"
                    driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
                elif isinstance(entry, dict):
                    if len(entry) != 1:
                        raise ValueError(f"Invalid driver dict entry: {entry}")
                    drv, lap_sel = next(iter(entry.items()))
                    name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                    driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
                else:
                    raise ValueError(f"Unsupported driver entry: {entry}")
        else:
            raise ValueError("drivers must be dict, list, or tuple of supported specs.")

        if not (1 <= len(driver_specs) <= 3):
            raise ValueError("Must compare between 1 and 3 laps/drivers.")

        driver_codes = [spec['driver'] for spec in driver_specs]
        lap_selections = [spec['lap'] for spec in driver_specs]
        display_names = [spec['display_name'] for spec in driver_specs]

        default_channels = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'Total_Acc']
        user_provided_channels = channels is not None
        channels = channels or default_channels

        delta_aliases = {'delta', 'deltatime', 'Δ'}
        wants_delta = (
            len(driver_specs) > 1 and
            (not user_provided_channels or any(str(ch).lower() in delta_aliases for ch in channels))
        )
        channels = [ch for ch in channels if str(ch).lower() not in delta_aliases]
        effective_channels = channels.copy()
        if not user_provided_channels and 'Total_Acc' not in effective_channels:
            effective_channels.append('Total_Acc')

        units = {
            'Speed': 'km/h',
            'Throttle': '%',
            'Brake': '%',
            'RPM': 'rpm',
            'nGear': '',
            'DRS': '',
            'Total_Acc': 'g',
        }

        # Load data
        laps = []
        lap_objs = []
        for driver, lap_id in zip(driver_codes, lap_selections):
            if lap_id == 'fastest':
                lap = self.transformed_laps.pick_drivers(driver).pick_fastest()
            else:
                try:
                    lap = self.transformed_laps.pick_drivers(driver).pick_laps(int(lap_id)).iloc[0]
                except Exception as e:
                    raise ValueError(f"Invalid lap selection for {driver}: {lap_id}") from e

            fl = self.FastestLap(lap)
            # Compute total acceleration (vector magnitude)
            fl.df = compute_total_acceleration(fl.df)
            # Rename 'Total_Acceleration' column to 'Total_Acc' for plotting compatibility
            if 'Total_Acceleration' in fl.df.columns:
                fl.df = fl.df.rename(columns={'Total_Acceleration': 'Total_Acc'})
            laps.append(fl)
            lap_objs.append(lap)

        avg_air_temp = self.weather['AirTemp'].mean()
        avg_track_temp = self.weather['TrackTemp'].mean()

        lap_times = [lap['LapTime'].total_seconds() for lap in lap_objs]
        baseline_idx = lap_times.index(min(lap_times))
        baseline_name = display_names[baseline_idx]

        s1_dist, s2_dist = laps[baseline_idx].sector_distances
        corner_df = self.circuit_info.corners.copy().sort_values('Distance')

        # Color assignment
        palette = self.assign_colors(
            driver_specs,
            driver_color_map=driver_color_map,
            default_colors=globals().get('colors_pilots', None)
        )

        # Plot setup
        n_plots = len(effective_channels) + (1 if wants_delta else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        plt.style.use('dark_background')
        fig.patch.set_facecolor('black')
        for ax in axes:
            ax.set_facecolor('black')

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
            # Add white border contour to subplot
            
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

        # Annotations
        labels = []
        for i, (name, secs) in enumerate(zip(display_names, lap_times)):
            if pd.isna(secs):
                label = f"{name}: NaN"
            else:
                mins = int(secs // 60)
                rem = secs - mins * 60
                label = f"{name}: {mins}:{rem:06.3f}"
                if i == baseline_idx:
                    label += f"   AIR: {avg_air_temp:.1f}°C  TRACK: {avg_track_temp:.1f}°C"
            labels.append(label)

        fig.text(0.02, 0.98, "\n".join(labels), ha='left', va='top',
                color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=4))

        fig.text(0.9, 0.96, "Provided by: Pietro Paolo Melella",
                ha='right', va='bottom', color='white', fontsize=15)
        title = (f"{self.session.event['EventName']} {self.session.event.year} {session_label}"
                if session_label else f"{self.session.event['EventName']} {self.session.event.year}")
        fig.suptitle(title, color='white')
        fig.subplots_adjust(top=0.92)
        plt.tight_layout(rect=[0, 0, 0.90, 0.94])
        # ➕ Add white contour around each subplot
        for ax in axes:
            pos = ax.get_position()
            rect = plt.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                transform=fig.transFigure,
                facecolor='none',
                edgecolor='white',
                linewidth=1.2
            )
            fig.patches.append(rect)
            # Add logo below the "Provided by" text
        sys.path.append('/Users/PietroPaolo/Desktop/GitHub/F1/')
        logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')  # or .jpg etc.

        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            # [left, bottom, width, height] — values are in 0–1 relative figure coords
            logo_ax = fig.add_axes([0.80, 0.90, 0.06, 0.06], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")
        plt.show()