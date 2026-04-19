from __future__ import annotations

"""
investment_weather_summary.py

Focused executive-summary analysis for NZP weather-year runs.

Purpose
-------
Keep the existing detailed plots from weather_year_compare.py, but add a smaller
set of clearer, decision-oriented figures that answer three questions:

1. What is the system cost range across weather years?
2. How does the capacity mix / investment pattern change with weather?
3. Which weather characteristics appear to matter most for those changes?

Main inputs
-----------
- all_runs_summary.csv from weather_year_compare.py
- weather_year_inputs_summary.csv from weather_year_inputs.py

Main outputs
------------
- cost_range_summary.csv
- capacity_sensitivity_summary.csv
- clustered VRES-vs-heating-demand scatter plots coloured by selected outputs
- capacity mix plots for the most variable technologies
- input-output correlation summary for selected investment metrics

Usage example
-------------
python investment_weather_summary.py \
    --runs_summary C:/Users/adria/MODEL_PyPSA/Core/runs/weather_year_comparison/all_runs_summary.csv \
    --inputs_csv   C:/Users/adria/MODEL_PyPSA/Core/runs/weather_year_inputs/weather_year_inputs_summary.csv \
    --out_dir      C:/Users/adria/MODEL_PyPSA/Core/runs/weather_year_comparison/investment_weather_summary
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        converted = pd.to_numeric(out[c], errors="coerce")
        if converted.notna().sum() > 0:
            out[c] = converted
    return out


def scale_series(s: pd.Series, col: str) -> tuple[pd.Series, str]:
    s = pd.to_numeric(s, errors="coerce")
    if col == "objective":
        return s / 1e9, "bn €"
    if col.endswith("_mw"):
        return s / 1e3, "GW"
    if col.endswith("_mwh"):
        return s / 1e6, "TWh"
    if "utilisation" in col or col.endswith("_pu"):
        return s * 100, "%"
    return s, ""


def pretty_name(col: str) -> str:
    name = col
    for a, b in [
        ("cap_gen_", "Capacity: "),
        ("storage_energy_", "Storage energy: "),
        ("storage_power_", "Storage power: "),
        ("gen_", "Generation: "),
        ("_mw", " (MW)"),
        ("_mwh", " (MWh)"),
        ("_pu", " (p.u.)"),
        ("_", " "),
    ]:
        name = name.replace(a, b)
    return name.strip()


def coefficient_of_variation(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    mean = float(s.mean())
    if abs(mean) < 1e-12:
        return np.nan
    return float(s.std(ddof=0)) / abs(mean)


def choose_default_color_metrics(df: pd.DataFrame) -> list[str]:
    candidates = [
        "objective",
        "storage_energy_hydrogen_storage_mwh",
        "cap_gen_PV ground_mw",
        "cap_gen_wind_mw",
        "cap_gen_wind offshore_mw",
        "heat_pump_total_mw",
        "electrolyser_total_mw",
        "h2_pipeline_total_mw",
    ]
    return [c for c in candidates if c in df.columns]


def choose_variable_capacity_metrics(df: pd.DataFrame, max_n: int = 8) -> list[str]:
    cap_cols = [c for c in df.columns if c.startswith("cap_gen_") or c.startswith("storage_power_") or c.startswith("storage_energy_")]
    rows = []
    for c in cap_cols:
        cv = coefficient_of_variation(df[c])
        if pd.notna(cv):
            rows.append((c, cv))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return [c for c, _ in rows[:max_n]]


def clustered_scatter_table(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    x_bin: float,
    y_bin: float,
) -> pd.DataFrame:
    work = df[["year", x_col, y_col, color_col]].copy()
    work = work.dropna()
    if work.empty:
        return work

    x = pd.to_numeric(work[x_col], errors="coerce")
    y = pd.to_numeric(work[y_col], errors="coerce")
    c = pd.to_numeric(work[color_col], errors="coerce")
    work = work.loc[x.notna() & y.notna() & c.notna()].copy()
    if work.empty:
        return work

    work["x_plot"] = x.loc[work.index]
    work["y_plot"] = y.loc[work.index] / 1e6 if y_col.endswith("_mwh") else y.loc[work.index]
    work["color_plot"], color_unit = scale_series(work[color_col], color_col)

    work["x_bin"] = (work["x_plot"] / x_bin).round().astype(int)
    work["y_bin"] = (work["y_plot"] / y_bin).round().astype(int)

    grouped = (
        work.groupby(["x_bin", "y_bin"], dropna=False)
        .agg(
            x=("x_plot", "mean"),
            y=("y_plot", "mean"),
            color_value=("color_plot", "mean"),
            n_years=("year", "count"),
            years=("year", lambda s: ", ".join(map(str, sorted(s.tolist())))),
        )
        .reset_index(drop=True)
    )
    grouped.attrs["color_unit"] = color_unit
    return grouped



def plot_clustered_scatter(clustered: pd.DataFrame, title: str, color_label: str, out_path: Path) -> None:
    if clustered.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sizes = 40 + 35 * clustered["n_years"].values
    sc = ax.scatter(
        clustered["x"],
        clustered["y"],
        c=clustered["color_value"],
        s=sizes,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )

    # label many more points:
    # - always label multi-year clusters
    # - also label single-year points with a small alternating offset
    for i, (_, r) in enumerate(clustered.iterrows()):
        years_txt = str(r["years"])

        if int(r["n_years"]) >= 3:
            ax.annotate(
                years_txt,
                (r["x"], r["y"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        else:
            dx = 0.0006 if i % 2 == 0 else -0.0006
            dy = 0.22 if i % 3 == 0 else (-0.22 if i % 3 == 1 else 0.10)
            ax.annotate(
                years_txt,
                (r["x"], r["y"]),
                fontsize=7,
                alpha=0.9,
                xytext=(r["x"] + dx, r["y"] + dy),
                textcoords="data",
            )

    ax.set_xlabel("Combined VRES capacity factor (p.u.)")
    ax.set_ylabel("Electricity for heat (TWh)")
    ax.set_title(title)
    ax.text(
        0.01,
        0.99,
        "Upper-left = harder weather years (low VRES, high electricity-for-heat)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="dimgray",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_capacity_trajectories(df: pd.DataFrame, metrics: list[str], out_path: Path) -> None:
    if not metrics:
        return

    plt.figure(figsize=(10.5, 5.8))
    year_idx = pd.to_numeric(df["year"], errors="coerce")
    for col in metrics:
        s, unit = scale_series(df[col], col)
        plt.plot(year_idx, s, marker="o", linewidth=1.2, label=pretty_name(col))
    plt.title("Selected capacity and storage metrics across weather years")
    plt.xlabel("Weather year")
    plt.ylabel("Scaled value")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_capacity_variability_bar(summary_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    top = summary_df.sort_values("cv", ascending=False).head(top_n)
    if top.empty:
        return
    plt.figure(figsize=(9.5, 5.8))
    y = np.arange(len(top))
    plt.barh(y, top["cv"].values)
    plt.yticks(y, top["pretty_name"].values, fontsize=8)
    plt.xlabel("Coefficient of variation")
    plt.title("Most weather-sensitive capacity / investment metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_cost_range(df: pd.DataFrame, out_path: Path) -> None:
    if "objective" not in df.columns:
        return
    s = pd.to_numeric(df["objective"], errors="coerce") / 1e9
    year = pd.to_numeric(df["year"], errors="coerce")
    keep = s.notna() & year.notna()
    s = s[keep]
    year = year[keep]
    if s.empty:
        return

    plt.figure(figsize=(10.5, 5.4))
    plt.plot(year, s, marker="o", linewidth=1.3)
    plt.axhline(float(s.mean()), linestyle="--", linewidth=1.0, color="dimgray", label=f"Mean: {s.mean():.1f} bn €")
    plt.fill_between(year, float(s.min()), float(s.max()), alpha=0.08)
    plt.title("System cost range across weather years")
    plt.xlabel("Weather year")
    plt.ylabel("System cost (bn €)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_summary", required=True, type=str)
    ap.add_argument("--inputs_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--x_bin", default=0.0040, type=float,
                    help="Clustering width for VRES CF axis")
    ap.add_argument("--y_bin", default=0.8, type=float,
                    help="Clustering width for electricity-for-heat axis in TWh")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = safe_numeric(pd.read_csv(args.runs_summary))
    inp_df = safe_numeric(pd.read_csv(args.inputs_csv))

    if "year" not in runs_df.columns or "year" not in inp_df.columns:
        raise ValueError("Both inputs must contain a 'year' column.")

    runs_df["year"] = pd.to_numeric(runs_df["year"], errors="coerce")
    inp_df["year"] = pd.to_numeric(inp_df["year"], errors="coerce")

    merged = runs_df.merge(inp_df, on="year", how="inner")
    merged = merged.dropna(subset=["year"]).copy()

    # ------------------------------------------------------------------
    # Cost range summary
    # ------------------------------------------------------------------
    cost_rows = []
    if "objective" in merged.columns:
        cost = pd.to_numeric(merged["objective"], errors="coerce")
        cost_rows.append({
            "metric": "objective",
            "min_bn_eur": float(cost.min() / 1e9),
            "max_bn_eur": float(cost.max() / 1e9),
            "mean_bn_eur": float(cost.mean() / 1e9),
            "range_bn_eur": float((cost.max() - cost.min()) / 1e9),
            "cv": coefficient_of_variation(cost),
        })
    pd.DataFrame(cost_rows).to_csv(out_dir / "cost_range_summary.csv", index=False)
    plot_cost_range(merged, out_dir / "cost_range_by_year.png")

    # ------------------------------------------------------------------
    # Capacity / investment sensitivity summary
    # ------------------------------------------------------------------
    candidate_cols = [
        c for c in merged.columns
        if (
            c.startswith("cap_gen_")
            or c.startswith("storage_power_")
            or c.startswith("storage_energy_")
            or c in {
                "electrolyser_total_mw",
                "heat_pump_total_mw",
                "h2_pipeline_total_mw",
                "chp_h2_total_mw",
                "elec_transmission_total_mw",
            }
        )
    ]

    cap_rows = []
    for c in candidate_cols:
        s = pd.to_numeric(merged[c], errors="coerce").dropna()
        if len(s) < 2:
            continue
        cap_rows.append({
            "metric": c,
            "pretty_name": pretty_name(c),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
            "range": float(s.max() - s.min()),
            "cv": coefficient_of_variation(s),
        })

    cap_df = pd.DataFrame(cap_rows).sort_values("cv", ascending=False)
    cap_df.to_csv(out_dir / "capacity_sensitivity_summary.csv", index=False)
    plot_capacity_variability_bar(cap_df, out_dir / "capacity_sensitivity_ranking.png", top_n=12)

    variable_caps = choose_variable_capacity_metrics(merged, max_n=8)
    plot_capacity_trajectories(merged, variable_caps, out_dir / "selected_capacity_trajectories.png")

    # ------------------------------------------------------------------
    # Clustered weather scatter plots
    # ------------------------------------------------------------------
    base_needed = {"year", "cf_vres_combined_annual", "elec_for_heat_annual_mwh"}
    if base_needed.issubset(set(merged.columns)):
        color_metrics = choose_default_color_metrics(merged)
        scatter_rows = []
        for c in color_metrics:
            clustered = clustered_scatter_table(
                merged,
                x_col="cf_vres_combined_annual",
                y_col="elec_for_heat_annual_mwh",
                color_col=c,
                x_bin=args.x_bin,
                y_bin=args.y_bin,
            )
            if clustered.empty:
                continue

            _, color_unit = scale_series(pd.to_numeric(merged[c], errors="coerce"), c)
            label = pretty_name(c)
            if color_unit:
                label = f"{label.replace(' (MW)', '').replace(' (MWh)', '').replace(' (p.u.)', '')} [{color_unit}]"

            out_name = f"clustered_scatter_{c.replace('/', '_')}.png"
            plot_clustered_scatter(
                clustered,
                title=f"Weather-stress map coloured by {pretty_name(c)}",
                color_label=label,
                out_path=out_dir / out_name,
            )
            clustered.assign(color_metric=c).to_csv(out_dir / f"clustered_points_{c}.csv", index=False)
            scatter_rows.append({"color_metric": c, "n_clusters": len(clustered), "label": label})

        pd.DataFrame(scatter_rows).to_csv(out_dir / "clustered_scatter_inventory.csv", index=False)

    # ------------------------------------------------------------------
    # Focused input-output correlations for investment metrics
    # ------------------------------------------------------------------
    focus_inputs = [
        c for c in [
            "cf_vres_combined_annual",
            "heat_demand_peak_mw",
            "elec_for_heat_annual_mwh",
            "elec_for_heat_peak_mw",
            "cop_winter_mean",
            "wind_drought_max_hours",
            "dark_doldrums_hours",
            "cold_stress_hours_cop_lt2",
            "stress_energy_block",
            "stress_peak_block",
            "stress_persistence_block",
            "stress_score",
        ] if c in merged.columns
    ]

    focus_outputs = [
        c for c in [
            "objective",
            "heat_pump_total_mw",
            "electrolyser_total_mw",
            "h2_pipeline_total_mw",
            "storage_energy_hydrogen_storage_mwh",
            "cap_gen_PV ground_mw",
            "cap_gen_wind_mw",
            "cap_gen_wind offshore_mw",
            "curtailment_share_pu",
            "elec_transmission_total_mw",
        ] if c in merged.columns
    ]

    corr_rows = []
    for i_col in focus_inputs:
        x = pd.to_numeric(merged[i_col], errors="coerce")
        for o_col in focus_outputs:
            y = pd.to_numeric(merged[o_col], errors="coerce")
            both = pd.concat([x, y], axis=1).dropna()
            if len(both) < 5:
                continue
            x_valid = both.iloc[:, 0]
            y_valid = both.iloc[:, 1]
            if float(x_valid.std(ddof=0)) == 0.0:
                continue
            if float(y_valid.std(ddof=0)) == 0.0:
                continue
            corr = x_valid.corr(y_valid)
            if pd.notna(corr):
                corr_rows.append({
                    "input_metric": i_col,
                    "output_metric": o_col,
                    "corr": float(corr),
                    "abs_corr": float(abs(corr)),
                })

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)
    corr_df.to_csv(out_dir / "focused_input_output_correlations.csv", index=False)

    print("\nTop capacity / investment sensitivity metrics:")
    if not cap_df.empty:
        print(cap_df[["pretty_name", "cv", "min", "max"]].head(10).to_string(index=False))

    print("\nTop focused input-output correlations:")
    if not corr_df.empty:
        print(corr_df[["input_metric", "output_metric", "corr"]].head(15).to_string(index=False))

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()