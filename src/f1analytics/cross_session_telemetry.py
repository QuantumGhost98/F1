import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os
from f1analytics.delta_time_sector_constrained import delta_time
from f1analytics.interpolate_df import interpolate_dataframe
import sys
from f1analytics.colors_pilots import colors_pilots


class CrossSessionTelemetry:
    """
    A class for comparing telemetry data between laps from different F1 sessions.
    Similar interface to the regular Telemetry class but handles multiple sessions.
    """
    
    def __init__(self, sessions_data):
        """
        Initialize with multiple sessions data.
        
        Parameters:
        sessions_data: dict with format:
        {
            'session_key': {
                'session': fastf1_session_object,
                'session_name': str,
                'year': int,
                'session_type': str
            }
        }
        
        Example:
        sessions_data = {
            'fp2': {
                'session': fp2_session,
                'session_name': 'Baku Grand Prix',
                'year': 2025,
                'session_type': 'FP2'
            },
            'race': {
                'session': race_session,
                'session_name': 'Baku Grand Prix', 
                'year': 2025,
                'session_type': 'R'
            }
        }
        """
        self.sessions_data = sessions_data
        self.sessions = {}
        self.colors_pilots = colors_pilots
        
        # Process each session
        for session_key, data in sessions_data.items():
            session = data['session']
            self.sessions[session_key] = {
                'session': session,
                'session_name': data['session_name'],
                'year': data['year'],
                'session_type': data['session_type'],
                'laps': session.laps,
                'weather': session.weather_data,
                'circuit_info': session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
            }
            
            # Transform laps to total seconds
            transformed_laps = session.laps.copy()
            transformed_laps.loc[:, "LapTime (s)"] = session.laps["LapTime"].dt.total_seconds()
            self.sessions[session_key]['transformed_laps'] = transformed_laps

    class FastestLap:
        """
        Wraps a fastest-lap record and provides convenience accessors.
        Enhanced to include session information.
        """
        def __init__(self, lap, session_key, session_type):
            self.name = lap['Driver']
            self.session_key = session_key
            self.session_type = session_type
            self.s1_time = lap['Sector1Time']
            self.s2_time = lap['Sector2Time']
            df = lap.get_car_data().add_distance()
            self.df = interpolate_dataframe(df)

        @property
        def sector_distances(self):
            d1 = self.df[self.df['Time'] <= self.s1_time]['Distance'].max()
            d2 = self.df[self.df['Time'] <= (self.s1_time + self.s2_time)]['Distance'].max()
            return d1, d2

    def adjust_brightness(self, color, factor):
        """
        Lighten (factor>1) or darken (factor<1) an RGB color.
        """
        try:
            rgb = np.array(mcolors.to_rgb(color))
            adjusted = np.clip(rgb * factor, 0, 1)
            return mcolors.to_hex(adjusted)
        except Exception:
            return color

    def assign_colors(self, driver_specs, driver_color_map=None, default_colors=None, fallback_shades=None):
        """
        Returns a list of colors for each spec in driver_specs (order preserved).
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

            base_color = None
            if driver_color_map:
                base_color = driver_color_map.get(display, driver_color_map.get(driver))
            if base_color is None:
                base_color = default_colors.get(driver, 'white')

            count = used.get(base_color, 0)
            if count == 0:
                color = base_color
            else:
                alternates = fallback_shades.get(base_color, [])
                if count - 1 < len(alternates):
                    color = alternates[count - 1]
                else:
                    factor = 1 + 0.2 * ((count - len(alternates)) % 2) * (1 if ((count - len(alternates)) // 2) % 2 == 0 else -1)
                    color = self.adjust_brightness(base_color, factor)
            used[base_color] = count + 1
            palette.append(color)

        return palette

    def compare_laps(self, lap_specs, channels=None, session_label="Cross-Session", driver_color_map=None):
        """
        Compare laps from different sessions.
        
        Parameters:
        - lap_specs: list of tuples (session_key, driver, lap_selection)
          Example: [('fp2', 'LEC', 'fastest'), ('race', 'LEC', 15)]
        - channels: list of telemetry fields to plot
        - session_label: label for the comparison
        - driver_color_map: optional color mapping
        """
        if not lap_specs:
            raise ValueError("lap_specs cannot be empty")
        
        if len(lap_specs) > 3:
            raise ValueError("Maximum 3 laps can be compared")

        # Normalize lap specs
        driver_specs = []
        for spec in lap_specs:
            if len(spec) == 3:
                session_key, driver, lap_sel = spec
            else:
                raise ValueError("Each lap_spec must be a tuple of (session_key, driver, lap_selection)")
            
            if session_key not in self.sessions:
                raise ValueError(f"Session '{session_key}' not found. Available: {list(self.sessions.keys())}")
            
            session_type = self.sessions[session_key]['session_type']
            
            # Create display name
            if lap_sel == 'fastest':
                display_name = f"{driver}_{session_type}"
            else:
                display_name = f"{driver}_{session_type}_L{lap_sel}"
            
            driver_specs.append({
                'session_key': session_key,
                'driver': driver,
                'lap': lap_sel,
                'display_name': display_name
            })

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

        # Load data from different sessions
        laps = []
        lap_objs = []
        session_info = []
        
        for spec in driver_specs:
            session_key = spec['session_key']
            driver = spec['driver']
            lap_id = spec['lap']
            
            session_data = self.sessions[session_key]
            transformed_laps = session_data['transformed_laps']
            
            if lap_id == 'fastest':
                lap = transformed_laps.pick_drivers(driver).pick_fastest()
            else:
                try:
                    lap = transformed_laps.pick_drivers(driver).pick_laps(int(lap_id)).iloc[0]
                except Exception as e:
                    raise ValueError(f"Invalid lap selection for {driver} in session {session_key}: {lap_id}") from e

            fl = self.FastestLap(lap, session_key, session_data['session_type'])
            
            laps.append(fl)
            lap_objs.append(lap)
            session_info.append(session_data)

        # Use the first session for circuit info
        primary_session = session_info[0]

        # Find baseline (fastest lap)
        lap_times = [lap['LapTime'].total_seconds() for lap in lap_objs]
        baseline_idx = lap_times.index(min(lap_times))
        baseline_name = driver_specs[baseline_idx]['display_name']

        s1_dist, s2_dist = laps[baseline_idx].sector_distances
        corner_df = primary_session['circuit_info'].corners.copy().sort_values('Distance')

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

        # Plot telemetry channels
        plot_idx = 0
        for ch in effective_channels:
            ax = axes[plot_idx]
            if ch not in laps[baseline_idx].df.columns:
                ax.set_visible(False)
                plot_idx += 1
                continue
            for lap, col, spec in zip(laps, palette, driver_specs):
                ax.plot(lap.df['Distance'], lap.df.get(ch, np.nan),
                        color=col, linestyle='-', label=f"{spec['display_name']} {ch}")
            unit = units.get(ch, '')
            ax.set_ylabel(f"{ch} ({unit})" if unit else ch, color='white')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.tick_params(colors='white')
            ax.axvline(s1_dist, color='white', linestyle='--')
            ax.axvline(s2_dist, color='white', linestyle='--')

            # Add corner markers
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

        # Delta time plot
        if wants_delta:
            ax_dt = axes[-1]
            ref_lap = lap_objs[baseline_idx]
            for idx, comp_lap in enumerate(lap_objs):
                if idx == baseline_idx:
                    continue
                delta_series, ref_tel, comp_tel = delta_time(ref_lap, comp_lap)
                ax_dt.plot(ref_tel['Distance'], delta_series,
                        color=palette[idx], linestyle='-',
                        label=f"Δ ({driver_specs[idx]['display_name']} - {baseline_name})")
            
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

        # Create annotations with session info and weather data
        labels = []
        for i, (spec, secs) in enumerate(zip(driver_specs, lap_times)):
            session_type = spec['session_key'].upper()
            name = spec['display_name']
            
            # Get weather data for this specific session
            session_data = session_info[i]
            avg_air_temp = session_data['weather']['AirTemp'].mean()
            avg_track_temp = session_data['weather']['TrackTemp'].mean()
            
            if pd.isna(secs):
                label = f"{name} ({session_type}): NaN"
            else:
                mins = int(secs // 60)
                rem = secs - mins * 60
                label = f"{name} ({session_type}): {mins}:{rem:06.3f}"
            
            # Add weather data for each session
            label += f"   AIR: {avg_air_temp:.1f}°C  TRACK: {avg_track_temp:.1f}°C"
            labels.append(label)

        fig.text(0.02, 0.98, "\n".join(labels), ha='left', va='top',
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=4))

        fig.text(0.75, 0.92, "Provided by: Pietro Paolo Melella",
                ha='right', va='bottom', color='white', fontsize=10)
        
        # Create title
        event_name = primary_session['session'].event.EventName
        sessions_str = " vs ".join([spec['session_key'].upper() for spec in driver_specs])
        title = f"{event_name} - {session_label} ({sessions_str})"
        
        fig.suptitle(title, color='white')
        fig.subplots_adjust(top=0.92)
        plt.tight_layout(rect=[0, 0, 0.90, 0.94])
        
        # Add white contour around each subplot
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
        
        # Add logo
        sys.path.append('/Users/PietroPaolo/Desktop/GitHub/F1/')
        logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')

        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            logo_ax = fig.add_axes([0.80, 0.90, 0.06, 0.06], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")
        
        plt.show()        