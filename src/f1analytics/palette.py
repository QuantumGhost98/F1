"""
Unified color palette for f1analytics.

Provides:
    - driver_colors: driver code → hex color
    - team_colors:   team name   → hex color

All colors are hex-based for consistent rendering on dark backgrounds.
"""

# ── Team colors (official 2025/2026 hex values) ──────────────────────────────

team_colors = {
    "Red Bull Racing":   "#3671C6",
    "Ferrari":           "#E80020",
    "Mercedes":          "#27F4D2",
    "McLaren":           "#FF8700",
    "Aston Martin":      "#229971",
    "Alpine":            "#FF87BC",
    "Williams":          "#64C4FF",
    "Racing Bulls":      "#6692FF",
    "Haas F1 Team":      "#B6BABD",
    "Audi":              "#00E701",
    "Cadillac":          "#C0C0C0",
}


# ── Driver colors (hex, derived from team colors) ────────────────────────────
# First-seat driver gets the team color, second-seat gets a lighter variant.

driver_colors = {
    # Red Bull Racing
    "VER": "#3671C6",
    "HAD": "#3671C6",

    # Ferrari
    "LEC": "#E80020",
    "HAM": "#E80020",

    # Mercedes
    "RUS": "#27F4D2",
    "ANT": "#27F4D2",

    # McLaren
    "NOR": "#FF8700",
    "PIA": "#FF8700",

    # Aston Martin
    "ALO": "#229971",
    "STR": "#229971",

    # Alpine
    "GAS": "#FF87BC",
    "COL": "#FF87BC",

    # Williams
    "SAI": "#64C4FF",
    "ALB": "#64C4FF",

    # Racing Bulls
    "LAW": "#6692FF",
    "LIN": "#6692FF",

    # Haas
    "OCO": "#B6BABD",
    "BEA": "#B6BABD",

    # Audi
    "HUL": "#00E701",
    "BOR": "#00E701",

    # Cadillac
    "PER": "#C0C0C0",
    "BOT": "#C0C0C0",
}


# Backward compatibility alias
colors_pilots = driver_colors
