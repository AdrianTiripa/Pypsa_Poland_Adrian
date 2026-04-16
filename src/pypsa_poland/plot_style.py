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
    value_labels: bool = True,
    figsize: tuple[float, float] = (11, 4.5),
    rotate_xticks: bool = True,
    ref_line: float | None = None,
    ref_label: str | None = None,
) -> None:
    """
    Single-series bar chart with optional value labels above bars.

    Parameters
    ----------
    series      : pd.Series — index becomes x-axis labels
    ref_line    : draw a horizontal dashed reference line at this y-value
    ref_label   : legend label for the reference line
    """
    series = series.dropna()
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(series))

    bars = ax.bar(x, series.values, color=color,
                  edgecolor="white", linewidth=0.5, zorder=3, width=0.65)

    if value_labels:
        for b, v in zip(bars, series.values):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + np.ptp(series.values) * 0.01,
                    _format_value_label(v, unit),
                    ha="center", va="bottom", fontsize=7.5, color=LABEL_COLOR,
                )

    if ref_line is not None:
        ax.axhline(ref_line, color=RED, linewidth=1.2, linestyle="--",
                   label=ref_label or f"Reference: {ref_line}", zorder=4)
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    rot = 45 if rotate_xticks and len(series) > 6 else 0
    ax.set_xticklabels(series.index.astype(str), rotation=rot,
                       ha="right" if rot else "center", fontsize=9)
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
) -> None:
    """
    Stacked bar chart — year (or any category) on x-axis, carriers stacked.

    Automatically keeps the top_n carriers by total and bundles the rest as
    "other". Uses CARRIER_COLORS by default; pass color_map to override.
    """
    if wide.empty:
        return

    wide = wide.fillna(0.0).sort_index()

    # Select top_n columns by total, bundle rest as "other"
    totals = wide.sum(axis=0).sort_values(ascending=False)
    top_cols = list(totals.head(top_n).index)
    rest = wide.drop(columns=top_cols, errors="ignore")
    plot = wide[top_cols].copy()
    if rest.shape[1] > 0:
        plot["other"] = rest.sum(axis=1)

    cmap = color_map or CARRIER_COLORS
    colors = [cmap.get(str(c), CARRIER_COLORS["other"]) for c in plot.columns]

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(plot))
    x = np.arange(len(plot))

    for col, color in zip(plot.columns, colors):
        ax.bar(x, plot[col].values, bottom=bottom, label=str(col),
               color=color, edgecolor="white", linewidth=0.4, zorder=3, width=0.65)
        bottom += plot[col].values

    ax.set_xticks(x)
    rot = 45 if rotate_xticks and len(plot) > 6 else 0
    ax.set_xticklabels(plot.index.astype(str), rotation=rot,
                       ha="right" if rot else "center", fontsize=9)
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
) -> None:
    """
    Multi-line plot over a numeric or datetime index.

    Parameters
    ----------
    shade_range : (ymin, ymax) — shade a horizontal band (e.g. normal range)
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
        mk = "o" if markers else None
        ax.plot(df.index, df[col], marker=mk, color=color,
                label=str(col), zorder=3, linewidth=1.8, markersize=4.5)

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
) -> None:
    """
    Scatter with point labels (year or index value) and optional OLS trend line.

    Parameters
    ----------
    highlight : list of index values to highlight in a different colour
    """
    merged = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if merged.empty:
        return

    highlight = highlight or []

    fig, ax = plt.subplots(figsize=figsize)

    colors_pts = [
        highlight_color if idx in highlight else color
        for idx in merged.index
    ]
    ax.scatter(merged["x"], merged["y"], c=colors_pts, zorder=4,
               s=55, edgecolors="white", linewidths=0.8)

    for xi, yi, lab in zip(merged["x"], merged["y"], merged.index):
        ax.annotate(
            str(lab), (xi, yi),
            textcoords="offset points", xytext=(5, 4),
            fontsize=8, color=LABEL_COLOR,
        )

    if trend and len(merged) >= 3:
        z = np.polyfit(merged["x"], merged["y"], 1)
        p = np.poly1d(z)
        xline = np.linspace(merged["x"].min(), merged["x"].max(), 200)
        ax.plot(xline, p(xline), "--", color=RED,
                linewidth=1.2, alpha=0.65, zorder=2, label="Linear trend")

        # Pearson r annotation
        r = np.corrcoef(merged["x"], merged["y"])[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.2f}",
                transform=ax.transAxes, fontsize=9,
                color=LABEL_COLOR, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#DDDDDD", alpha=0.9))

    xfmt = xunit if xunit else ""
    yfmt = yunit if yunit else ""
    ax.set_xlabel(f"{xlabel} ({xfmt})" if xfmt else xlabel,
                  fontsize=10, color=LABEL_COLOR)
    ax.set_ylabel(f"{ylabel} ({yfmt})" if yfmt else ylabel,
                  fontsize=10, color=LABEL_COLOR)

    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=11, fontweight="bold",
                 color=TITLE_COLOR, pad=10, loc="left")

    if trend:
        ax.legend(fontsize=8, framealpha=0.92, edgecolor="#DDDDDD", fancybox=False)

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
