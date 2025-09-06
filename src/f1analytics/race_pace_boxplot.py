# race_pace_boxplot.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


class RacePaceBoxplot:
    """
    Create a race pace boxplot (FP2, long runs) with per-team stats.
    """

    def __init__(self, race_pace_laps: pd.DataFrame, session, session_name: str, year: int, session_type: str,
                 team_palette: dict, ylim: tuple = None):
        """
        :param race_pace_laps: DataFrame with ['Team', 'LapTime (s)']
        :param session: FastF1 session object (for laps filtering)
        :param session_name: e.g. "Hungary GP"
        :param year: e.g. 2025
        :param session_type: e.g. "FP2"
        :param team_palette: dict mapping team → color
        :param ylim: optional (min, max) y-axis limits for LapTime (s)
        """
        self.race_pace_laps = race_pace_laps.copy().reset_index(drop=True)
        self.session = session
        self.session_name = session_name
        self.year = year
        self.session_type = session_type
        self.team_palette = team_palette
        self.ylim = ylim

        # Pre-compute stats
        self.team_stats = self.race_pace_laps.groupby("Team")["LapTime (s)"].agg(
            ["mean", "median", "min", "max"]
        )
        self.team_order = self.team_stats["median"].sort_values().index

    def plot(self, figsize=(20, 10), box_width=0.5):
        """
        Render the race pace boxplot.
        :param figsize: tuple, figure size
        :param box_width: float, width of the boxes
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=self.race_pace_laps,
            x="Team",
            y="LapTime (s)",
            hue="Team",
            order=self.team_order,
            palette=self.team_palette,
            whiskerprops=dict(color="white"),
            boxprops=dict(edgecolor="white"),
            medianprops=dict(color="grey"),
            capprops=dict(color="white"),
            width=box_width,
            dodge=False,
            ax=ax
        )

        # Annotate stats under whiskers
        for i, team in enumerate(self.team_order):
            whisker_position = self.team_stats.loc[team, "min"]
            mean_time = self.team_stats.loc[team, "mean"]
            median_time = self.team_stats.loc[team, "median"]
            std_time = self.race_pace_laps[self.race_pace_laps["Team"] == team]["LapTime (s)"].std()

            ax.text(
                i, whisker_position - 0.5,
                f"Mean: {mean_time:.2f}\nMedian: {median_time:.2f}\nStd: {std_time:.2f}",
                ha="center", color="white", fontsize=10
            )

        # Title uses manual session metadata
        ax.set_title(
            f"{self.session_name} {self.year} {self.session_type} — Race Pace",
            color="white"
        )
        ax.grid(False)
        ax.set(xlabel=None)

        if self.ylim:
            ax.set_ylim(*self.ylim)

        plt.text(
            0.95, 0.1, "Provided by:\nPietro Paolo Melella",
            va="bottom", ha="right", transform=ax.transAxes,
            color="white", fontsize=12
        )

        plt.tight_layout()

        # Insert logo at specified coordinates
        logo_path = "/Users/PietroPaolo/Desktop/GitHub/F1/logo-square.png"
        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            logo_ax = fig.add_axes([0.73, 0.84, 0.12, 0.12], anchor='NE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
        else:
            print(f"[WARN] Logo file not found at: {logo_path}")

        plt.show()

    def table(self):
        """Return the computed stats (mean, median, min, max, std) per team."""
        stats = self.team_stats.copy()
        stats["std"] = self.race_pace_laps.groupby("Team")["LapTime (s)"].std()
        return stats