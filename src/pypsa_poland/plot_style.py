# plot_style.py
#
# Shared plot styling and chart primitives for the pypsa-poland analysis suite.
#
# Designed for NZP Phase 2 outputs at paper/report quality.
# All plotting scripts import from here so that figures are visually consistent
# across single-run plots, cross-year comparisons, and map outputs.
#
# Public API:
#   apply_style()       — call once before creating any figure
#   CARRIER_COLORS      — fixed colour map keyed by carrier/sector name
#   bar()               — single-series bar chart
#   stacked_bar()       — stacked bar chart (carrier breakdown)
#   line()              — multi-line time-series chart
#   scatter_annotated() — scatter with point labels and optional trend line
#   heatmap()           — annotated heatmap (e.g. year × season)
#   duration_curve()    — overlapping load/generation duration curves
#   ranking_table()     — DataFrame rendered as a styled figure table
#   savefig()           — save with consistent dpi/bbox settings

from __future__ import annotations
 
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
 
 
# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
 
_RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.facecolor":     "#F8F9FA",
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.color":         "#E0E0E0",
    "grid.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": True,
    "axes.edgecolor":     "#CCCCCC",
    "xtick.color":        "#444444",
    "ytick.color":        "#444444",
    "axes.labelcolor":    "#444444",
    "text.color":         "#1A1A2E",
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#DDDDDD",
    "legend.fancybox":    False,
    "legend.fontsize":    8.5,
    "lines.linewidth":    1.8,
    "lines.markersize":   4.5,
    "figure.dpi":         150,
    "savefig.dpi":        220,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
}
 
# Accent colours used for single-series plots
BLUE   = "#2E86AB"
RED    = "#E84855"
AMBER  = "#F4B400"
GREY   = "#607D8B"
 
# Title / label colours
TITLE_COLOR = "#1A1A2E"
LABEL_COLOR = "#444444"
 
 
def apply_style() -> None:
    """Call once at script startup to apply all rcParams."""
    matplotlib.rcParams.update(_RC)
 
 
# ---------------------------------------------------------------------------
# Fixed carrier colour map
# ---------------------------------------------------------------------------
# Colours are intentionally consistent with energy-system convention:
#   solar → amber/yellow, onshore wind → sky blue, offshore → deep navy,
#   nuclear → purple, gas → orange, biomass → lime,
#   hydrogen → cyan family, storage → greens/teals,
#   heat → reds, transport → brown.
 
CARRIER_COLORS: dict[str, str] = {
    # ---- Generation ----
    "PV ground":               "#F4B400",   # amber
    "wind":                    "#2196F3",   # sky blue
    "wind offshore":           "#0D47A1",   # deep navy
    "nuclear":                 "#7B1FA2",   # purple
    "Natural gas":             "#FF7043",   # burnt orange
    "Biogas plant":            "#8BC34A",   # lime green
    # ---- Hydrogen sector ----
    "hydrogen":                "#00BCD4",   # cyan
    "hydrogen storage":        "#006064",   # dark teal
    "hydrogen storage other":  "#80DEEA",   # light teal
    # ---- Electricity storage ----
    "battery":                 "#43A047",   # mid green
    "flow":                    "#1B5E20",   # dark green
    "PSH":                     "#1565C0",   # deep blue
    # ---- Heat ----
    "hot_water":               "#EF5350",   # coral red  (thermal storage)
    "heat_pump":               "#E91E63",   # magenta pink
    "high_temp_heat":          "#BF360C",   # dark terracotta
    # ---- Transport ----
    "transport":               "#795548",   # brown
    # ---- Transmission ----
    "DC":                      "#607D8B",   # blue-grey
    # ---- Load sector classes ----
    "electricity_or_other":    "#455A64",   # dark blue-grey
    "heat":                    "#EF5350",   # coral (same as hot_water)
    "high_temp_heat_load":     "#BF360C",
    "hydrogen_load":           "#00BCD4",
    "transport_load":          "#795548",
    # ---- Fallback ----
    "other":                   "#B0BEC5",   # light grey
}
 
 
def carrier_color(name: str) -> str:
    """Return the fixed colour for a carrier, falling back to grey."""
    return CARRIER_COLORS.get(str(name), CARRIER_COLORS["other"])
 
 
def carrier_colors_for(names: list[str]) -> list[str]:
    """Return a list of colours for a list of carrier names."""
    return [carrier_color(n) for n in names]
 
 
# ---------------------------------------------------------------------------
# Unit formatting helpers
# ---------------------------------------------------------------------------
 
def _auto_unit_formatter(unit: str):
    """
    Return a matplotlib FuncFormatter that formats axis tick values with
    the given unit string appended.
    Examples: 'GW', 'TWh', '%', 'p.u.', 'bn €'
    """
    if unit in ("GW", "TWh", "bn €"):
        fmt = lambda v, _: f"{v:,.1f}"
    elif unit in ("%",):
        fmt = lambda v, _: f"{v:.1f}"
    elif unit in ("p.u.", "CF"):
        fmt = lambda v, _: f"{v:.2f}"
    else:
        fmt = lambda v, _: f"{v:,.0f}"
    return mticker.FuncFormatter(fmt)
 
 
def _format_value_label(v: float, unit: str) -> str:
    """Format a single value for a bar-top annotation."""
    if unit in ("GW", "TWh", "bn €"):
        return f"{v:,.1f}"
    elif unit == "%":
        return f"{v:.1f}%"
    elif unit in ("p.u.", "CF"):
        return f"{v:.3f}"
    elif unit == "hours":
        return f"{int(round(v)):,}"
    else:
        return f"{v:,.0f}"
 
 
# ---------------------------------------------------------------------------
# Figure / axis helpers
# ---------------------------------------------------------------------------
 
def _looks_like_year_index(idx) -> tuple[bool, list[int] | None]:
    """
    Detect whether an index is a list of plausible calendar years (e.g. 1940..2025).

    Returns (True, list_of_int_years) if yes, (False, None) otherwise.
    Used by bar() and stacked_bar() to switch from "label every bar" to
    "label every Nth year" automatically.
    """
    try:
        years = []
        for v in idx:
            iv = int(v)
            if iv != float(v):  # rejects non-integers like 2020.5
                return False, None
            if iv < 1900 or iv > 2100:
                return False, None
            years.append(iv)
        if len(years) < 2:
            return False, None
        return True, years
    except (TypeError, ValueError):
        return False, None


def _apply_ax_style(ax: plt.Axes, ylabel: str, unit: str, title: str,
                    subtitle: str | None = None) -> None:
    """Apply consistent axis styling."""
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
 
    ax.set_ylabel(f"{ylabel} ({unit})" if ylabel and unit else (ylabel or unit),
                  fontsize=10, color=LABEL_COLOR)
 
    full_title = title
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=11, fontweight="bold",
                 color=TITLE_COLOR, pad=10, loc="left")
 
 
def savefig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    """Save figure with consistent settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight" if tight else None,
                facecolor="white")
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
 
def bar(
    series: pd.Series,
    out_path: Path,
    title: str,
    ylabel: str,
    unit: str = "",
    color: str = BLUE,
    subtitle: str | None = None,
    value_labels: bool = False,
    figsize: tuple[float, float] = (11, 4.5),
    rotate_xticks: bool = True,
    ref_line: float | None = None,
    ref_label: str | None = None,
    year_tick_step: int = 5,
) -> None:
    """
    Single-series bar chart.

    Parameters
    ----------
    series          : pd.Series — index becomes x-axis labels
    value_labels    : default False. Flip to True only for short summary series
                      (solve status, short rankings) where the per-bar number
                      is the headline. Long weather-year series are easier to
                      read without per-bar text.
    year_tick_step  : when the index looks like a list of integer years AND
                      the series is long (>30 bars), only show every Nth year
                      label (default 5: 1940, 1945, 1950, ...). All bars are
                      still drawn — only the labels are sparser.
    ref_line        : draw a horizontal dashed reference line at this y-value
    ref_label       : legend label for the reference line
    """
    series = series.dropna()
    if series.empty:
        return
 
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(series))
 
    bars = ax.bar(x, series.values, color=color,
                  edgecolor="white", linewidth=0.5, zorder=3, width=0.65)
 
    if value_labels:
        # When there are many bars and labels were explicitly requested,
        # rotate them 90° so they don't overlap horizontally.
        label_rot = 90 if (rotate_xticks and len(series) > 30) else 0

        # Extra headroom so rotated labels don't clip against the top axis.
        if label_rot == 90 and len(series) >= 2:
            finite = series.values[np.isfinite(series.values)]
            if finite.size:
                ymax = float(finite.max())
                ymin = float(finite.min())
                yspan = max(ymax - ymin, abs(ymax), 1.0)
                ax.set_ylim(min(0.0, ymin), ymax + 0.22 * yspan)

        for b, v in zip(bars, series.values):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + np.ptp(series.values) * 0.01,
                    _format_value_label(v, unit),
                    ha="center", va="bottom",
                    rotation=label_rot,
                    fontsize=7.5, color=LABEL_COLOR,
                )
 
    if ref_line is not None:
        ax.axhline(ref_line, color=RED, linewidth=1.2, linestyle="--",
                   label=ref_label or f"Reference: {ref_line}", zorder=4)
        ax.legend(fontsize=8)
 
    # ---- Decide the x-tick label policy --------------------------------
    # If the index looks like a list of years AND there are many bars, show
    # only multiples of `year_tick_step` so the axis isn't a wall of text.
    is_year_index, year_values = _looks_like_year_index(series.index)

    if is_year_index and rotate_xticks and len(series) > 30:
        tick_positions = [i for i, y in enumerate(year_values)
                          if int(y) % year_tick_step == 0]
        tick_labels = [str(int(year_values[i])) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=9)
    else:
        ax.set_xticks(x)
        if rotate_xticks and len(series) > 30:
            rot = 90
        elif rotate_xticks and len(series) > 6:
            rot = 45
        else:
            rot = 0
        ax.set_xticklabels(series.index.astype(str), rotation=rot,
                           ha="right" if rot == 45 else "center", fontsize=9)

    # Auto-label the x-axis as 'Weather Years' whenever the index is a list
    # of calendar years. This catches every weather-year bar across all the
    # plotting scripts without requiring per-call edits, and never fires on
    # short categorical series like solve-status counts.
    if is_year_index:
        ax.set_xlabel("Weather Years", fontsize=10, color=LABEL_COLOR)

    ax.yaxis.set_major_formatter(_auto_unit_formatter(unit))
 
    _apply_ax_style(ax, ylabel, unit, title, subtitle)
    fig.tight_layout()
    savefig(fig, out_path)
 
 
# ---------------------------------------------------------------------------
# Stacked bar chart
# ---------------------------------------------------------------------------
 
def stacked_bar(
    wide: pd.DataFrame,
    out_path: Path,
    title: str,
    ylabel: str,
    unit: str = "",
    top_n: int = 10,
    subtitle: str | None = None,
    figsize: tuple[float, float] = (12, 5),
    legend_cols: int = 2,
    color_map: dict[str, str] | None = None,
    rotate_xticks: bool = True,
    value_labels: bool = False,
    year_tick_step: int = 5,
    stack_order_by_variability: bool = False,
) -> None:
    """
    Stacked bar chart — year (or any category) on x-axis, carriers stacked.
    Uses CARRIER_COLORS by default; pass color_map to override.

    Parameters
    ----------
    value_labels                 : default False. Long weather-year stacks are
                                   easier to read without per-bar totals on
                                   top. Flip to True for short summary stacks.
    year_tick_step               : when the index is a list of years AND the
                                   series is long (>30 bars), only label every
                                   Nth year on the x-axis (default 5).
    stack_order_by_variability   : default False (legacy ordering: largest
                                   carrier on the bottom). When True, carriers
                                   are stacked from LOWEST coefficient of
                                   variation upward, so the rock-stable
                                   technologies form a flat foundation and the
                                   weather-sensitive ones float visibly on top.
                                   Use this for the installed-capacity figure.
    """
    if wide.empty:
        return
 
    wide = wide.fillna(0.0).sort_index()
 
    # ---- Decide which carriers stay vs. fold into "other" --------------
    # Always select on absolute size (you don't want to drop a tiny but
    # variable carrier into "other" just to highlight variability — that
    # would defeat the purpose).
    totals = wide.sum(axis=0).sort_values(ascending=False)
    top_cols = list(totals.head(top_n).index)
    rest = wide.drop(columns=top_cols, errors="ignore")
    plot = wide[top_cols].copy()
    if rest.shape[1] > 0:
        plot["other"] = rest.sum(axis=1)

    # ---- Decide stack ORDER inside the bars ----------------------------
    # Two modes:
    #   - default: largest carrier first, so totals look smooth visually.
    #   - stack_order_by_variability: ascending CV first, so the stable
    #     foundation sits at the bottom and any year-to-year wiggle is
    #     concentrated in the top stacks where it's easy to see.
    if stack_order_by_variability:
        means = plot.mean(axis=0).replace(0, np.nan)
        cvs = (plot.std(axis=0, ddof=0) / means.abs()).fillna(0.0)
        # Keep "other" at the very top (it's a heterogeneous bucket and its
        # CV isn't meaningful).
        if "other" in plot.columns:
            non_other = [c for c in plot.columns if c != "other"]
            non_other_sorted = sorted(non_other, key=lambda c: cvs.get(c, 0.0))
            ordered_cols = non_other_sorted + ["other"]
        else:
            ordered_cols = sorted(plot.columns, key=lambda c: cvs.get(c, 0.0))
        plot = plot[ordered_cols]
 
    cmap = color_map or CARRIER_COLORS
    colors = [cmap.get(str(c), CARRIER_COLORS["other"]) for c in plot.columns]
 
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(plot))
    x = np.arange(len(plot))
 
    for col, color in zip(plot.columns, colors):
        ax.bar(x, plot[col].values, bottom=bottom, label=str(col),
               color=color, edgecolor="white", linewidth=0.4, zorder=3, width=0.65)
        bottom += plot[col].values

    # Total at the top of each stack — only when explicitly requested,
    # because long stacks look much cleaner without per-bar text.
    if value_labels:
        totals_per_x = plot.sum(axis=1).values
        label_rot = 90 if (rotate_xticks and len(plot) > 30) else 0

        if label_rot == 90 and len(plot) >= 2:
            finite = totals_per_x[np.isfinite(totals_per_x)]
            if finite.size:
                ymax = float(finite.max())
                yspan = max(ymax, 1.0)
                ax.set_ylim(0.0, ymax + 0.28 * yspan)

        if np.any(np.isfinite(totals_per_x)):
            offset = np.ptp(totals_per_x) * 0.01
            for xi, total in zip(x, totals_per_x):
                if np.isfinite(total) and total > 0:
                    ax.text(
                        xi, total + offset,
                        _format_value_label(total, unit),
                        ha="center", va="bottom",
                        rotation=label_rot,
                        fontsize=7.5, color=LABEL_COLOR,
                    )

    # ---- X-tick label policy: same year-aware logic as bar() -----------
    is_year_index, year_values = _looks_like_year_index(plot.index)

    if is_year_index and rotate_xticks and len(plot) > 30:
        tick_positions = [i for i, y in enumerate(year_values)
                          if int(y) % year_tick_step == 0]
        tick_labels = [str(int(year_values[i])) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=9)
    else:
        ax.set_xticks(x)
        if rotate_xticks and len(plot) > 30:
            rot = 90
        elif rotate_xticks and len(plot) > 6:
            rot = 45
        else:
            rot = 0
        ax.set_xticklabels(plot.index.astype(str), rotation=rot,
                           ha="right" if rot == 45 else "center", fontsize=9)

    # Auto-label the x-axis as 'Weather Years' whenever the index is a list
    # of calendar years. Same policy as bar() — fires for every cross-year
    # stack across all plotting scripts, never for short categorical stacks.
    if is_year_index:
        ax.set_xlabel("Weather Years", fontsize=10, color=LABEL_COLOR)

    ax.yaxis.set_major_formatter(_auto_unit_formatter(unit))
 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              loc="upper right", ncols=legend_cols,
              fontsize=8.5, framealpha=0.92, edgecolor="#DDDDDD",
              fancybox=False)
 
    _apply_ax_style(ax, ylabel, unit, title, subtitle)
    fig.tight_layout()
    savefig(fig, out_path)
 
 
# ---------------------------------------------------------------------------
# Line chart
# ---------------------------------------------------------------------------
 
def line(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    ylabel: str,
    unit: str = "",
    cols: list[str] | None = None,
    subtitle: str | None = None,
    figsize: tuple[float, float] = (11, 4.5),
    color_map: dict[str, str] | None = None,
    markers: bool = True,
    legend_cols: int = 2,
    shade_range: tuple[float, float] | None = None,
    markers_only: bool = False,
) -> None:
    """
    Multi-line plot over a numeric or datetime index.

    Parameters
    ----------
    shade_range  : (ymin, ymax) — shade a horizontal band (e.g. normal range)
    markers_only : if True, draw markers without connecting lines. Use this
                   when the index is categorical-like (weather years) and a
                   connecting line would imply a temporal trend that doesn't
                   exist between adjacent years. The result is effectively a
                   multi-series scatter on a shared y-axis.
    """
    if df.empty:
        return
    plot_cols = [c for c in (cols or df.columns) if c in df.columns]
    if not plot_cols:
        return
 
    cmap = color_map or CARRIER_COLORS
 
    fig, ax = plt.subplots(figsize=figsize)
 
    if shade_range is not None:
        ax.axhspan(shade_range[0], shade_range[1],
                   color="#E0E0E0", alpha=0.4, zorder=0, label="Normal range")
 
    for col in plot_cols:
        color = cmap.get(str(col), BLUE)
        if markers_only:
            # Pure scatter: no line, just markers. Slightly larger so the
            # plot still reads cleanly with 86 weather years.
            ax.scatter(df.index, df[col], color=color, label=str(col),
                       s=22, alpha=0.85, edgecolors="white", linewidths=0.5,
                       zorder=3)
        else:
            mk = "o" if markers else None
            ax.plot(df.index, df[col], marker=mk, color=color,
                    label=str(col), zorder=3, linewidth=1.8, markersize=4.5)
 
    is_year_index, _ = _looks_like_year_index(df.index)
    if is_year_index:
        ax.set_xlabel("Weather Years", fontsize=10, color=LABEL_COLOR)

    ax.yaxis.set_major_formatter(_auto_unit_formatter(unit))
    ax.legend(loc="best", ncols=legend_cols,
              fontsize=8.5, framealpha=0.92, edgecolor="#DDDDDD", fancybox=False)
 
    _apply_ax_style(ax, ylabel, unit, title, subtitle)
    fig.tight_layout()
    savefig(fig, out_path)
 
 
# ---------------------------------------------------------------------------
# Scatter with year annotations + optional trend line
# ---------------------------------------------------------------------------
 
def scatter_annotated(
    x: pd.Series,
    y: pd.Series,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    xunit: str = "",
    yunit: str = "",
    color: str = BLUE,
    trend: bool = True,
    figsize: tuple[float, float] = (8, 6),
    highlight: list | None = None,
    highlight_color: str = RED,
    # -- Clustering controls --------------------------------------------------
    cluster: bool = False,
    cluster_x_tol: float | None = None,
    cluster_y_tol: float | None = None,
    cluster_frac: float = 0.03,
    max_cluster_label_years: int = 4,
    # -- Label-policy controls ------------------------------------------------
    label_extrema_singletons: bool = True,
    n_extra_singleton_labels: int = 12,
) -> None:
    """
    Year-labelled scatter with optional spatial clustering and a readable,
    non-redundant label policy.

    Labelling policy (when clustering is active):
      - Every cluster with >= 2 years is labelled (that is the point of
        clustering: group what the reader would otherwise have to decode).
      - Singletons are only labelled if they are *extrema* on the data: the
        years at min/max of x and min/max of y plus the top
        `n_extra_singleton_labels` off-trend singletons with the largest
        residuals from the linear fit. Every other singleton becomes an
        unlabelled dot.
      - Default n_extra_singleton_labels = 12, chosen to land around 20-30
        labelled years in an 86-point weather ensemble: more than the bare
        extrema so the reader learns which years drive the trend, but
        fewer than all 86 so the figure stays readable. Bump it up for
        sparse scatters and down for tight ones.
      - This keeps the figure readable without losing the informative years.

    Statistical policy:
      - The Pearson r and the OLS trend line are ALWAYS computed on the raw
        (un-clustered) data, so the statistical content of the figure is
        identical to the un-clustered version. Clustering is purely visual.

    Clustering tolerances:
      - If cluster_x_tol / cluster_y_tol are given, they are used as absolute
        bin widths (backward-compatible with earlier callers).
      - Otherwise, if cluster=True, tolerances are derived automatically as
        `cluster_frac` of the data range on each axis (default 3%). This lets
        callers just pass `cluster=True` without knowing axis units.
      - If neither is given and cluster=False, no clustering happens and every
        point is plotted individually (no labels except extrema, if enabled).
    """
    # ------------------------------------------------------------------
    # 1. Merge inputs and identify the year index
    # ------------------------------------------------------------------
    merged = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if merged.empty:
        return

    merged = merged.reset_index()
    merged = merged.rename(columns={merged.columns[0]: "year"})

    highlight = set(highlight or [])

    def format_years(years):
        years = sorted(int(v) for v in years)
        if len(years) == 1:
            return str(years[0])
        if len(years) <= max_cluster_label_years:
            return ", ".join(str(v) for v in years)
        return f"{years[0]}–{years[-1]} ({len(years)} yrs)"

    # ------------------------------------------------------------------
    # 2. Resolve clustering tolerances
    #
    # Precedence: explicit tolerances > cluster=True (auto) > no clustering.
    # ------------------------------------------------------------------
    x_range = float(merged["x"].max() - merged["x"].min())
    y_range = float(merged["y"].max() - merged["y"].min())

    if cluster_x_tol is None and cluster and x_range > 0:
        cluster_x_tol = cluster_frac * x_range
    if cluster_y_tol is None and cluster and y_range > 0:
        cluster_y_tol = cluster_frac * y_range

    do_cluster = (cluster_x_tol is not None) and (cluster_y_tol is not None)

    # ------------------------------------------------------------------
    # 3. Build the grouped (plotted) dataframe
    # ------------------------------------------------------------------
    if do_cluster:
        x0 = merged["x"].min()
        y0 = merged["y"].min()

        merged["x_bin"] = np.floor((merged["x"] - x0) / cluster_x_tol).astype(int)
        merged["y_bin"] = np.floor((merged["y"] - y0) / cluster_y_tol).astype(int)

        grouped = (
            merged.groupby(["x_bin", "y_bin"], dropna=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                years=("year", lambda s: list(s)),
                n_years=("year", "count"),
            )
            .reset_index(drop=True)
        )
    else:
        grouped = merged[["x", "y", "year"]].copy()
        grouped["years"] = grouped["year"].apply(lambda v: [v])
        grouped["n_years"] = 1
        grouped = grouped[["x", "y", "years", "n_years"]]

    grouped["label"] = grouped["years"].apply(format_years)
    grouped["is_highlight"] = grouped["years"].apply(
        lambda vals: any(v in highlight for v in vals)
    )

    # ------------------------------------------------------------------
    # 4. Decide which rows get a text label
    #
    # - Multi-year clusters always get a label.
    # - Singletons get a label only if they are extrema on the data, so the
    #   figure still tells the reader which years are the outliers without
    #   drowning the plot in 86 overlapping numbers.
    # ------------------------------------------------------------------
    grouped["show_label"] = grouped["n_years"] >= 2

    if label_extrema_singletons:
        singleton_mask = grouped["n_years"] == 1
        if singleton_mask.any():
            # Axis extrema: min/max on x and min/max on y (singletons only,
            # because a multi-cluster sitting at an extremum is already labelled).
            extrema_idx = set()
            singles = grouped.loc[singleton_mask]
            extrema_idx.update([
                singles["x"].idxmin(),
                singles["x"].idxmax(),
                singles["y"].idxmin(),
                singles["y"].idxmax(),
            ])

            # A small number of extra "most off-trend" singletons.
            if n_extra_singleton_labels > 0 and len(merged) >= 3:
                coeffs = np.polyfit(merged["x"], merged["y"], 1)
                y_hat = np.polyval(coeffs, singles["x"])
                residuals = np.abs(singles["y"].values - y_hat)
                order = np.argsort(residuals)[::-1]
                for k in order[:n_extra_singleton_labels]:
                    extrema_idx.add(singles.index[int(k)])

            grouped.loc[list(extrema_idx), "show_label"] = True

    # Highlighted years always get a label too.
    grouped.loc[grouped["is_highlight"], "show_label"] = True

    # ------------------------------------------------------------------
    # 5. Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    point_colors = [
        highlight_color if is_h else color
        for is_h in grouped["is_highlight"]
    ]
    # Cluster size scales with the number of years it merges, so the reader
    # can see at a glance where the ensemble is dense.
    point_sizes = 55 + 18 * (grouped["n_years"] - 1)

    ax.scatter(
        grouped["x"],
        grouped["y"],
        c=point_colors,
        s=point_sizes,
        zorder=4,
        edgecolors="white",
        linewidths=0.8,
    )

    # Rotating offsets avoid stacked labels when two clusters sit close.
    offsets = [(5, 4), (6, -10), (-18, 5), (-18, -10), (8, 12), (-12, 12)]

    label_counter = 0
    for _, row in grouped.iterrows():
        if not row["show_label"]:
            continue
        dx, dy = offsets[label_counter % len(offsets)]
        label_counter += 1
        fs = 8 if row["n_years"] == 1 else 7.3
        ax.annotate(
            row["label"],
            (row["x"], row["y"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=fs,
            color=LABEL_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

    # ------------------------------------------------------------------
    # 6. Trend and Pearson r — always computed on the RAW (un-clustered) data
    #    so the statistical content matches the un-clustered version.
    # ------------------------------------------------------------------
    if trend and len(merged) >= 3:
        z = np.polyfit(merged["x"], merged["y"], 1)
        p = np.poly1d(z)
        xline = np.linspace(merged["x"].min(), merged["x"].max(), 200)
        ax.plot(
            xline, p(xline), "--",
            color=RED, linewidth=1.2, alpha=0.65, zorder=2, label="Linear trend"
        )

        r = np.corrcoef(merged["x"], merged["y"])[0, 1]
        ax.text(
            0.05, 0.95, f"r = {r:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            color=LABEL_COLOR,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#DDDDDD",
                alpha=0.9,
            ),
        )

    ax.set_xlabel(f"{xlabel} ({xunit})" if xunit else xlabel, fontsize=10, color=LABEL_COLOR)
    ax.set_ylabel(f"{ylabel} ({yunit})" if yunit else ylabel, fontsize=10, color=LABEL_COLOR)

    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(
        full_title,
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=10,
        loc="left",
    )

    if trend:
        ax.legend(fontsize=8, framealpha=0.92, edgecolor="#DDDDDD", fancybox=False)

    # Footer note only makes sense when clustering is actually active.
    if do_cluster and (grouped["n_years"] > 1).any():
        ax.text(
            0.01, 0.01,
            "Nearby years are grouped into one marker (size ∝ n_years). "
            "Only clusters and axis extrema are labelled.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="dimgray",
        )

    fig.tight_layout()
    savefig(fig, out_path)
 
# ---------------------------------------------------------------------------
# Heatmap (year × season or year × region)
# ---------------------------------------------------------------------------
 
def heatmap(
    data: pd.DataFrame,
    out_path: Path,
    title: str,
    cbar_label: str = "",
    subtitle: str | None = None,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    fmt: str = ".2f",
    figsize: tuple[float, float] | None = None,
    row_label: str = "",
    col_label: str = "",
) -> None:
    """
    Annotated heatmap — rows on y-axis, columns on x-axis.
    Typical uses: year × season CF, year × region stress.
    """
    if data.empty:
        return
 
    nrows, ncols = data.shape
    if figsize is None:
        figsize = (max(5, ncols * 1.1), max(3, nrows * 0.38))
 
    fig, ax = plt.subplots(figsize=figsize)
 
    arr = data.values.astype(float)
    vmin_ = vmin if vmin is not None else np.nanmin(arr)
    vmax_ = vmax if vmax is not None else np.nanmax(arr)
 
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin_, vmax=vmax_)
 
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(data.columns.astype(str), fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(data.index.astype(str), fontsize=8.5)
 
    if col_label:
        ax.set_xlabel(col_label, fontsize=10, color=LABEL_COLOR)
    if row_label:
        ax.set_ylabel(row_label, fontsize=10, color=LABEL_COLOR)
 
    if annotate:
        mid = (vmin_ + vmax_) / 2
        for i in range(nrows):
            for j in range(ncols):
                v = arr[i, j]
                if np.isfinite(v):
                    text_color = "#1A1A2E" if v > mid else "white"
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=7.5, color=text_color)
 
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9, color=LABEL_COLOR)
    cbar.ax.tick_params(labelsize=8, colors=LABEL_COLOR)
 
    ax.spines[:].set_visible(False)
    ax.grid(False)
    ax.tick_params(colors=LABEL_COLOR)
 
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=11, fontweight="bold",
                 color=TITLE_COLOR, pad=10, loc="left")
 
    fig.tight_layout()
    savefig(fig, out_path)
 
 
# ---------------------------------------------------------------------------
# Duration curve
# ---------------------------------------------------------------------------
 
def duration_curve(
    series_dict: dict[str, pd.Series],
    out_path: Path,
    title: str,
    ylabel: str,
    unit: str = "",
    subtitle: str | None = None,
    figsize: tuple[float, float] = (11, 4.5),
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Overlapping duration curves for one or more series.
    Each series is sorted descending; x-axis shows fraction of hours.
    """
    cmap = color_map or CARRIER_COLORS
    fig, ax = plt.subplots(figsize=figsize)
 
    for label, s in series_dict.items():
        s_sorted = s.dropna().sort_values(ascending=False).reset_index(drop=True)
        x_frac = np.linspace(0, 1, len(s_sorted))
        color = cmap.get(str(label), BLUE)
        ax.plot(x_frac * 100, s_sorted.values, label=str(label),
                color=color, linewidth=1.8, zorder=3)
 
    ax.set_xlabel("% of hours", fontsize=10, color=LABEL_COLOR)
    ax.yaxis.set_major_formatter(_auto_unit_formatter(unit))
    ax.legend(loc="upper right", ncols=1,
              fontsize=8.5, framealpha=0.92, edgecolor="#DDDDDD", fancybox=False)
 
    _apply_ax_style(ax, ylabel, unit, title, subtitle)
    fig.tight_layout()
    savefig(fig, out_path)
 
 
# ---------------------------------------------------------------------------
# Ranking table figure (text-based)
# ---------------------------------------------------------------------------
 
def ranking_table(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    subtitle: str | None = None,
    figsize: tuple[float, float] | None = None,
    highlight_top: int = 3,
    highlight_bottom: int = 3,
) -> None:
    """
    Render a DataFrame as a styled figure table.
    Top rows highlighted in light red (most stressed), bottom in light green (least).
    """
    if df.empty:
        return
 
    n_rows, n_cols = df.shape
    if figsize is None:
        figsize = (max(7, n_cols * 1.6), max(3, n_rows * 0.38 + 1.2))
 
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
 
    col_labels = list(df.columns)
    row_labels  = [str(i) for i in df.index]
    cell_text   = []
    for _, row in df.iterrows():
        cell_text.append([
            f"{v:.3f}" if isinstance(v, float) else str(v)
            for v in row.values
        ])
 
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
 
    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#1A1A2E")
        cell.set_text_props(color="white", fontweight="bold")
 
    # Style row labels column
    for i in range(1, n_rows + 1):
        cell = table[i, -1]
        cell.set_facecolor("#F0F0F0")
 
    # Highlight top N (most stressful)
    for i in range(1, min(highlight_top + 1, n_rows + 1)):
        for j in range(n_cols):
            table[i, j].set_facecolor("#FFEBEE")
 
    # Highlight bottom N (least stressful)
    for i in range(max(1, n_rows - highlight_bottom + 1), n_rows + 1):
        for j in range(n_cols):
            table[i, j].set_facecolor("#E8F5E9")
 
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=11, fontweight="bold",
                 color=TITLE_COLOR, pad=12, loc="left",
                 transform=ax.transAxes)
 
    fig.tight_layout()
    savefig(fig, out_path)