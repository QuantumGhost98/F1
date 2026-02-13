"""
Session delta chart — compare lap times between two sessions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f1analytics.config import logger
from f1analytics.plot_utils import setup_dark_theme, add_branding


def create_session_delta_chart(session1, label1, session2, label2, save_path=None):
    """
    Compare fastest lap-times between two sessions and plot a delta chart.

    Parameters
    ----------
    session1 : FastF1 session
    label1   : display label for session1 (e.g. "2024 Qualifying")
    session2 : FastF1 session
    label2   : display label for session2
    save_path: optional path to save figure

    Returns
    -------
    (fig, ax)
    """
    laps1 = session1.laps.copy()
    laps2 = session2.laps.copy()

    laps1.loc[:, 'LapTime_s'] = laps1['LapTime'].dt.total_seconds()
    laps2.loc[:, 'LapTime_s'] = laps2['LapTime'].dt.total_seconds()

    best1 = laps1.groupby('Driver')['LapTime_s'].min()
    best2 = laps2.groupby('Driver')['LapTime_s'].min()

    common = sorted(set(best1.index) & set(best2.index))
    if not common:
        raise ValueError("No common drivers found between sessions")

    deltas = pd.DataFrame({
        'Driver': common,
        label1: [best1[d] for d in common],
        label2: [best2[d] for d in common],
    })
    deltas['Delta'] = deltas[label2] - deltas[label1]
    deltas = deltas.sort_values('Delta')

    n = len(deltas)
    x = np.arange(n)
    colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in deltas['Delta']]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 6))
    setup_dark_theme(fig, [ax])

    ax.bar(x, deltas['Delta'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(deltas['Driver'], color='white')
    ax.axhline(0, color='white', linestyle='--', linewidth=0.8)

    ax.set_ylabel('Δ Time (s)', color='white')
    ax.set_title(f'{label2} vs {label1} — Lap Time Delta', color='white', fontsize=14)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='white')

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='white', label=f'{label2} faster'),
        Patch(facecolor='#e74c3c', edgecolor='white', label=f'{label1} faster'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.95, 0.93])
    add_branding(fig, text_pos=(0.95, 0.91), logo_pos=[0.80, 0.91, 0.08, 0.08])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        logger.info("Saved plot to %s", save_path)

    plt.show()
    return fig, ax
