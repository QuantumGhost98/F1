"""
Shared plotting utilities for f1analytics.

Replaces duplicated branding, color assignment, and dark-theme
setup code found across 10+ modules.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from f1analytics.config import LOGO_PATH, ATTRIBUTION, logger
from f1analytics.palette import driver_colors as colors_pilots  # compat alias


# ── Default fallback shades when same base color is used for multiple drivers ──
# Keys match hex values in palette.py so cross-session comparisons (same driver
# appearing twice) get visually distinct colors.

DEFAULT_FALLBACK_SHADES = {
    # Red Bull
    '#3671C6': ['#A3C4F3', '#D6E6FF'],
    '#6CA0E8': ['#A3C4F3'],
    # Ferrari
    '#E80020': ['#FFB347', '#FFDDA6'],   # orange/gold — clearly distinct from red
    '#FF6F61': ['#FFB347'],
    # Mercedes
    '#27F4D2': ['#B8FFE8', '#E0FFF5'],
    '#86F9E8': ['#E0FFF5'],
    # McLaren
    '#FF8700': ['#FFD966', '#FFF2CC'],
    '#FFB347': ['#FFD966'],
    # Aston Martin
    '#229971': ['#80DEAC', '#C8F7DC'],
    '#5CBFA0': ['#C8F7DC'],
    # Alpine
    '#FF87BC': ['#E0AAFF', '#FFD6EB'],
    '#FFB8D9': ['#FFD6EB'],
    # Williams
    '#64C4FF': ['#C8E6FF', '#E8F4FF'],
    '#A3DDFF': ['#E8F4FF'],
    # Racing Bulls
    '#6692FF': ['#B3CCFF', '#D9E6FF'],
    '#99B8FF': ['#D9E6FF'],
    # Haas
    '#B6BABD': ['#FFFFFF', '#E8E8E8'],
    '#D9DCDE': ['#FFFFFF'],
    # Audi
    '#00E701': ['#99FF99', '#CCFFCC'],
    '#66F34D': ['#CCFFCC'],
    # Cadillac
    '#C0C0C0': ['#FFFFFF', '#F0F0F0'],
    '#E0E0E0': ['#FFFFFF'],
    # Legacy names (backward compat)
    'red': ['#FFB347', 'lightcoral'],
    'blue': ['cyan', 'lightblue'],
    'orange': ['gold', 'wheat'],
    'grey': ['white', 'silver'],
    'green': ['lime', 'springgreen'],
    'pink': ['violet', 'lightpink'],
}


# ── Color helpers ──────────────────────────────────────────────────────────────

def adjust_brightness(color: str, factor: float) -> str:
    """Lighten (factor>1) or darken (factor<1) an RGB color."""
    try:
        rgb = np.array(mcolors.to_rgb(color))
        adjusted = np.clip(rgb * factor, 0, 1)
        return mcolors.to_hex(adjusted)
    except Exception:
        return color


def assign_colors(
    driver_specs: List[Dict[str, Any]],
    driver_color_map: Optional[Dict[str, str]] = None,
    default_colors: Optional[Dict[str, str]] = None,
    fallback_shades: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Return a list of colors for each spec in *driver_specs* (order preserved).

    Priority per spec:
        driver_color_map[display_name] > driver_color_map[driver]
        > default_colors[driver] > colors_pilots[driver] > 'white'

    When the same base color repeats it cycles through *fallback_shades*,
    then auto-adjusts brightness.
    """
    if default_colors is None:
        default_colors = {}
    if fallback_shades is None:
        fallback_shades = DEFAULT_FALLBACK_SHADES

    used: Dict[str, int] = {}
    palette: List[str] = []

    for spec in driver_specs:
        driver = spec['driver']
        display = spec['display_name']

        # Determine base color
        base_color = None
        if driver_color_map:
            base_color = driver_color_map.get(display, driver_color_map.get(driver))
        if base_color is None:
            base_color = default_colors.get(driver, colors_pilots.get(driver, 'white'))

        count = used.get(base_color, 0)
        if count == 0:
            color = base_color
        else:
            alternates = fallback_shades.get(base_color, [])
            if count - 1 < len(alternates):
                color = alternates[count - 1]
            else:
                # Auto adjust brightness for further duplicates
                factor = 1 + 0.2 * ((count - len(alternates)) % 2) * (
                    1 if ((count - len(alternates)) // 2) % 2 == 0 else -1
                )
                color = adjust_brightness(base_color, factor)

        used[base_color] = count + 1
        palette.append(color)

    return palette


def assign_colors_simple(
    drivers: List[str],
    fallback_shades: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Simpler color assignment keyed by driver code only (no display_name logic).
    Used by CornerAnalysis and similar classes that pass raw driver lists.
    """
    if fallback_shades is None:
        fallback_shades = DEFAULT_FALLBACK_SHADES

    used: Dict[str, int] = {}
    palette: List[str] = []

    for driver in drivers:
        base_color = colors_pilots.get(driver, 'white')
        count = used.get(base_color, 0)
        if count == 0:
            palette.append(base_color)
        else:
            alternates = fallback_shades.get(base_color, ['white'])
            alt_color = alternates[count - 1] if count - 1 < len(alternates) else 'white'
            palette.append(alt_color)
        used[base_color] = count + 1

    return palette


# ── Plot setup & branding ─────────────────────────────────────────────────────

def setup_dark_theme(fig, axes) -> None:
    """Apply the standard f1analytics dark theme to a figure."""
    plt.style.use('dark_background')
    fig.patch.set_facecolor('black')
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('black')


def add_branding(
    fig,
    text_pos: Tuple[float, float] = (0.9, 0.96),
    logo_pos: Optional[List[float]] = None,
    text_fontsize: int = 15,
) -> None:
    """
    Add the standard attribution text and logo to a figure.

    Parameters
    ----------
    fig : matplotlib Figure
    text_pos : (x, y) in figure coordinates for the attribution text
    logo_pos : [left, bottom, width, height] for logo axes, or None for auto
    text_fontsize : font size for the attribution text
    """
    fig.text(
        text_pos[0], text_pos[1], ATTRIBUTION,
        ha='right', va='bottom', color='white', fontsize=text_fontsize,
    )

    if logo_pos is None:
        logo_pos = [0.80, 0.90, 0.08, 0.08]

    logo_str = str(LOGO_PATH)
    if os.path.exists(logo_str):
        logo_img = mpimg.imread(logo_str)
        logo_ax = fig.add_axes(logo_pos, anchor='NE', zorder=10)
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')
    else:
        logger.warning("Logo file not found at: %s", logo_str)


def finalize_plot(
    fig,
    axes=None,
    save_path: Optional[str] = None,
    tight_layout_rect: Optional[List[float]] = None,
    show: bool = False,
):
    """
    Finalize a plot: apply tight_layout, optionally save, return (fig, axes).

    Parameters
    ----------
    fig : matplotlib Figure
    axes : axes or list of axes (returned as-is)
    save_path : if provided, save the figure to this path
    tight_layout_rect : rect argument for plt.tight_layout
    show : if True, also call plt.show() (for backward compat in interactive use)

    Returns
    -------
    (fig, axes)
    """
    if tight_layout_rect:
        plt.tight_layout(rect=tight_layout_rect)
    else:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        logger.info("Saved plot to %s", save_path)

    if show:
        plt.show()

    return fig, axes
