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
    "HAD": "#6CA0E8",

    # Ferrari
    "LEC": "#E80020",
    "HAM": "#FF6F61",

    # Mercedes
    "RUS": "#27F4D2",
    "ANT": "#86F9E8",

    # McLaren
    "NOR": "#FF8700",
    "PIA": "#FFB347",

    # Aston Martin
    "ALO": "#229971",
    "STR": "#5CBFA0",

    # Alpine
    "GAS": "#FF87BC",
    "COL": "#FFB8D9",

    # Williams
    "SAI": "#64C4FF",
    "ALB": "#A3DDFF",

    # Racing Bulls
    "LAW": "#6692FF",
    "LIN": "#99B8FF",

    # Haas
    "OCO": "#B6BABD",
    "BEA": "#D9DCDE",

    # Audi
    "HUL": "#00E701",
    "BOR": "#66F34D",

    # Cadillac
    "PER": "#C0C0C0",
    "BOT": "#E0E0E0",
}


# Backward compatibility alias
colors_pilots = driver_colors
