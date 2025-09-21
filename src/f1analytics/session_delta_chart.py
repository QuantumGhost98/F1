import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os
import sys
from f1analytics.colors_pilots import colors_pilots


def create_session_delta_chart(session1, session2, title="Session Comparison"):
    """
    Create a simple chart showing which drivers improved (+) or got worse (-) between sessions.
    
    Parameters:
    - session1: First FastF1 session (baseline)
    - session2: Second FastF1 session (comparison)
    - title: Title for the chart
    
    Example:
    create_session_delta_chart(fp2_session, q_session, "FP2 vs Qualifying")
    """
    
    # Get all drivers that appear in both sessions
    drivers1 = set(session1.laps['Driver'].unique())
    drivers2 = set(session2.laps['Driver'].unique())
    common_drivers = list(drivers1.intersection(drivers2))
    
    if not common_drivers:
        print("No common drivers found between sessions!")
        return
    
    # Get fastest laps for each driver
    driver_deltas = []
    for driver in common_drivers:
        try:
            lap1 = session1.laps.pick_drivers(driver).pick_fastest()
            lap2 = session2.laps.pick_drivers(driver).pick_fastest()
            
            time1 = lap1['LapTime'].total_seconds()
            time2 = lap2['LapTime'].total_seconds()
            
            delta = time2 - time1  # Positive means session2 is slower (worse)
            
            driver_deltas.append({
                'driver': driver,
                'time1': time1,
                'time2': time2,
                'delta': delta,
                'improved': delta < 0  # True if session2 is faster
            })
        except:
            continue
    
    if not driver_deltas:
        print("No valid fastest laps found!")
        return
    
    # Sort by delta (best improvement first)
    driver_deltas.sort(key=lambda x: x['delta'])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Prepare data for plotting
    drivers = [d['driver'] for d in driver_deltas]
    deltas = [d['delta'] for d in driver_deltas]
    colors = []
    
    for d in driver_deltas:
        if d['improved']:
            colors.append('green')  # Improved (faster in session2)
        else:
            colors.append('red')    # Got worse (slower in session2)
    
    # Create horizontal bar chart
    bars = ax.barh(drivers, deltas, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        width = bar.get_width()
        label_x = width + (0.01 if width >= 0 else -0.01)
        ha = 'left' if width >= 0 else 'right'
        
        # Format delta time
        sign = '+' if delta >= 0 else ''
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
               f'{sign}{delta:.3f}s', 
               ha=ha, va='center', color='white', fontweight='bold', fontsize=10)
    
    # Add vertical line at zero
    ax.axvline(0, color='white', linestyle='-', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time Delta (seconds)', color='white', fontsize=12)
    ax.set_ylabel('Drivers', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Improved (Session 2 faster)'),
        Patch(facecolor='red', alpha=0.8, label='Got worse (Session 2 slower)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
             facecolor='black', edgecolor='white')
    
    # Title and attribution
    try:
        event_name = session1.event.EventName
        year = session1.event.year
        full_title = f"{event_name} {year} - {title}\nFastest Lap Comparison"
    except:
        full_title = f"{title}\nFastest Lap Comparison"
    
    ax.set_title(full_title, color='white', fontsize=14, pad=20)
    
    # Add summary text
    improved_count = sum(1 for d in driver_deltas if d['improved'])
    worse_count = len(driver_deltas) - improved_count
    
    summary_text = f"Improved: {improved_count} drivers | Got worse: {worse_count} drivers"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
           color='white', fontsize=10, va='top',
           bbox=dict(facecolor='black', alpha=0.7, pad=4))
    
    # Attribution
    fig.text(0.98, 0.02, "Provided by: Pietro Paolo Melella",
            ha='right', va='bottom', color='white', fontsize=8)
    
    plt.tight_layout()
    
    # Add logo
    logo_path = os.path.join('/Users/PietroPaolo/Desktop/GitHub/F1/', 'logo-square.png')
    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        logo_ax = fig.add_axes([0.88, 0.91, 0.08, 0.08], anchor='NE', zorder=10)
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')
    
    plt.show()
    
