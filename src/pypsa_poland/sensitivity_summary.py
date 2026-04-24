# sensitivity_summary.py
#
# Weather sensitivity and robustness summary for pypsa-poland output metrics.
#
# Part of the weather-robustness workflow (step 4 of 4):
#   1. weather_year_inputs.py  — characterise input-side meteorological years
#   2. results_to_csv.py       — export structured CSVs from each run
#   3. weather_year_compare.py — aggregate outputs across runs into a summary CSV
#   4. THIS SCRIPT             — quantify sensitivity and compute input-output correlations
#
# Reads:
#   - all_runs_summary.csv from weather_year_compare.py
#   - weather_year_inputs_summary.csv from weather_year_inputs.py
#
# Outputs (all written to --out_dir):
#   - output_sensitivity_summary.csv   — all outputs ranked by CV (most sensitive first)
#   - output_robustness_summary.csv    — same, ranked ascending (most robust first)
#   - output_input_correlations.csv    — Pearson correlations between every input metric
#                                        and every output metric across weather years
#   - top_20_weather_sensitive_outputs.csv
#   - top_20_robust_outputs.csv
#   - top_30_input_output_links.csv
#   - top_weather_sensitive_outputs.png
#   - selected_sensitive_output_trajectories.png
#   - output_variability_heatmap.png
#   - top_input_output_relationships.png
#
# Usage:
#   python sensitivity_summary.py \
#       --runs_summary <all_runs_summary.csv> \
#       --inputs_csv   <weather_year_inputs_summary.csv> \
#       --out_dir      <output_folder>
#& C:/Users/adria/anaconda3/envs/pypsa-env/python.exe c:/Users/adria/MODEL_PyPSA/Core/pypsa-poland_ADRIAN/src/pypsa_poland/investment_weather_summary.py  --runs_summary C:\Users\adria\MODEL_PyPSA\Core\weather_year_comparison\all_runs_summary.csv --inputs_csv C:\Users\adria\MODEL_PyPSA\Core\weather_year_inputs\weather_year_inputs_summary.csv --out_dir C:\Users\adria\MODEL_PyPSA\Core\weather_year_comparison\investment_weather_summary  

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants — input and output metric lists
# ---------------------------------------------------------------------------

# Input-side weather metrics produced by weather_year_inputs.py.
INPUT_METRICS = [
    "cf_vres_combined_annual",
    "cf_vres_combined_winter",
    "heat_demand_annual_mwh",
    "heat_demand_peak_mw",
    "cop_annual_mean",
    "cop_winter_mean",
    "elec_for_heat_annual_mwh",
    "elec_for_heat_peak_mw",
    "wind_drought_max_hours",
    "dark_doldrums_hours",
    "cold_stress_hours_cop_lt2",
    "stress_energy_block",
    "stress_peak_block",
    "stress_persistence_block",
    "stress_score",
]

# Output metrics matched against these prefixes are prioritised for reporting.
OUTPUT_PRIORITY_PATTERNS = [
    "objective",
    "cap_gen_",
    "gen_",
    "curtailment_",
    "storage_power_",
    # storage_energy_ intentionally excluded from headline ranking
    # because in this model several storage energy metrics are mechanically
    # linked to storage power through fixed max_hours, so including both
    # double counts the same sensitivity story.
    "electrolyser_total_mw",
    "heat_pump_total_mw",
    "h2_pipeline_total_mw",
    "chp_h2_total_mw",
    "h2_network_abs_flow_mwh",
    "elec_transmission_total_mw",
    "transmission_mean_utilisation",
    "transmission_peak_utilisation",
    "peak_load_mw",
    "total_annual_load_mwh",
    "load_heat_mwh",
    "load_hydrogen_mwh",
    "load_transport_mwh",
]

EXCLUDE_OUTPUTS = {
    "year",
    "step_hr",
    "runtime_s",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_priority_output(col: str) -> bool:
    """Return True if col matches a recognised output metric pattern."""
    if col in EXCLUDE_OUTPUTS:
        return False
    if col.startswith("storage_energy_"):
        return False
    return any(p in col for p in OUTPUT_PRIORITY_PATTERNS)


def pretty_name(col: str) -> str:
    """Convert a snake_case metric column name to a human-readable label."""
    name = col
    replacements = [
        ("cap_gen_", "Capacity: "),
        ("gen_", "Generation: "),
        ("curtailment_", "Curtailment: "),
        ("storage_power_", "Storage power: "),
        ("storage_energy_", "Storage energy: "),
        ("_mw", " (MW)"),
        ("_mwh", " (MWh)"),
        ("_pu", " (p.u.)"),
        ("_", " "),
    ]
    for a, b in replacements:
        name = name.replace(a, b)
    return name.strip()


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all DataFrame columns to numeric, turning non-parseable values to NaN."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def scaled_series(s: pd.Series, col: str) -> tuple[pd.Series, str]:
    """
    Scale a raw metric series to a human-readable unit for plotting.

    Returns the scaled series and a unit label string.
    """
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


def coefficient_of_variation(s: pd.Series) -> float:
    """
    Return the population coefficient of variation (std / |mean|) for a series.

    Returns NaN if fewer than 2 non-null values are present or if the mean is
    effectively zero (to avoid division-by-zero on near-constant outputs).
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    mean = float(s.mean())
    std  = float(s.std(ddof=0))
    if abs(mean) < 1e-12:
        return np.nan
    return std / abs(mean)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_top_sensitive(summary_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path, top_n: int = 12) -> None:
    """
    Produce a horizontal bar chart of the top_n most weather-sensitive outputs
    (ranked by CV) and a line chart of their cross-year trajectories.
    """
    top = summary_df.sort_values("cv", ascending=False).dropna(subset=["cv"]).head(top_n)

    plt.figure(figsize=(9.5, 5.8))
    y = np.arange(len(top))
    plt.barh(y, top["cv"].values)
    plt.yticks(y, top["pretty_name"].values, fontsize=8)
    plt.xlabel("Coefficient of variation")
    plt.title("Most weather-sensitive outputs")
    plt.tight_layout()
    plt.savefig(out_dir / "top_weather_sensitive_outputs.png", dpi=220)
    plt.close()

    top_cols = top["metric"].tolist()
    if top_cols:
        plt.figure(figsize=(11.5, 5.8))
        for c in top_cols[:8]:
            s, _ = scaled_series(runs_df.set_index("year")[c], c)
            plt.plot(s.index, s.values, marker="o", linewidth=1.2, label=pretty_name(c))
        plt.title("Trajectories of selected weather-sensitive outputs")
        plt.xlabel("Weather Years")
        plt.ylabel("Scaled value")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "selected_sensitive_output_trajectories.png", dpi=220)
        plt.close()


def plot_output_variability_heatmap(summary_df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    """
    Produce a z-score heatmap of mean, std, range, and CV for the top_n most
    sensitive outputs. Each column is standardised so values are directly comparable.
    """
    top = summary_df.sort_values("cv", ascending=False).dropna(subset=["cv"]).head(top_n).copy()
    if top.empty:
        return

    plot = top[["mean", "std", "range", "cv"]].copy()
    plot.index = top["pretty_name"]

    z = plot.copy()
    for c in z.columns:
        s = z[c]
        std = float(s.std(ddof=0))
        if std > 0:
            z[c] = (s - s.mean()) / std
        else:
            z[c] = 0.0

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.33 * len(z))))
    im = ax.imshow(z.values, aspect="auto")
    ax.set_xticks(np.arange(len(z.columns)))
    ax.set_xticklabels(z.columns)
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(z.index, fontsize=8)
    ax.set_title("Output variability summary heatmap")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_dir / "output_variability_heatmap.png", dpi=220)
    plt.close()


def plot_top_input_output_links(corr_df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    """
    Produce a horizontal bar chart of the strongest input-output Pearson
    correlations, sorted by absolute correlation strength.
    """
    if corr_df.empty:
        return

    top = corr_df.reindex(corr_df["abs_corr"].sort_values(ascending=False).index).head(top_n).copy()
    top["label"] = top["input_metric"].map(pretty_name) + " \u2192 " + top["output_metric"].map(pretty_name)

    plt.figure(figsize=(10, max(5.0, 0.32 * len(top))))
    y = np.arange(len(top))
    plt.barh(y, top["corr"].values)
    plt.yticks(y, top["label"].values, fontsize=8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Pearson correlation")
    plt.title("Strongest input-output relationships")
    plt.tight_layout()
    plt.savefig(out_dir / "top_input_output_relationships.png", dpi=220)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load run summaries and input characterisations, compute sensitivity metrics,
    write CSVs, and produce variability and correlation figures.
    """
    ap = argparse.ArgumentParser(
        description="Weather sensitivity and robustness summary for pypsa-poland."
    )
    ap.add_argument("--runs_summary", required=True, type=str,
                    help="Path to all_runs_summary.csv from weather_year_compare.py")
    ap.add_argument("--inputs_csv", required=True, type=str,
                    help="Path to weather_year_inputs_summary.csv from weather_year_inputs.py")
    ap.add_argument("--out_dir", required=True, type=str)
    args = ap.parse_args()

    runs_path = Path(args.runs_summary)
    inp_path = Path(args.inputs_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_csv(runs_path)
    inp_df = pd.read_csv(inp_path)

    if "year" not in runs_df.columns or "year" not in inp_df.columns:
        raise ValueError("Both input files must contain a 'year' column.")

    runs_df = safe_numeric(runs_df)
    runs_df["year"] = pd.to_numeric(runs_df["year"], errors="coerce").astype("Int64")
    inp_df = safe_numeric(inp_df)
    inp_df["year"] = pd.to_numeric(inp_df["year"], errors="coerce").astype("Int64")

    merged = runs_df.merge(inp_df, on="year", how="inner", suffixes=("_out", "_in"))
    merged = merged.dropna(subset=["year"]).copy()

    output_cols = [c for c in runs_df.columns if is_priority_output(c)]

    summary_rows = []
    for col in output_cols:
        s = pd.to_numeric(runs_df[col], errors="coerce").dropna()
        if len(s) < 2:
            continue

        summary_rows.append({
            "metric": col,
            "pretty_name": pretty_name(col),
            "n_years": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
            "range": float(s.max() - s.min()),
            "cv": coefficient_of_variation(s),
            "min_year": int(runs_df.loc[s.index, "year"].iloc[s.argmin()]) if len(s) else np.nan,
            "max_year": int(runs_df.loc[s.index, "year"].iloc[s.argmax()]) if len(s) else np.nan,
        })

    sens_df = pd.DataFrame(summary_rows)
    sens_df = sens_df.sort_values("cv", ascending=False)
    sens_df.to_csv(out_dir / "output_sensitivity_summary.csv", index=False)

    robust_df = sens_df.sort_values("cv", ascending=True).copy()
    robust_df.to_csv(out_dir / "output_robustness_summary.csv", index=False)

    corr_rows = []
    output_cols_corr = [c for c in output_cols if c in merged.columns]
    input_cols_corr = [c for c in INPUT_METRICS if c in merged.columns]

    for in_col in input_cols_corr:
        x = pd.to_numeric(merged[in_col], errors="coerce")
        for out_col in output_cols_corr:
            y = pd.to_numeric(merged[out_col], errors="coerce")
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
                    "input_metric": in_col,
                    "output_metric": out_col,
                    "corr": float(corr),
                    "abs_corr": float(abs(corr)),
                    "n_obs": int(len(both)),
                })

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)
    corr_df.to_csv(out_dir / "output_input_correlations.csv", index=False)

    top_sensitive = sens_df.head(20).copy()
    top_robust = robust_df.head(20).copy()
    top_corr = corr_df.head(30).copy()

    top_sensitive.to_csv(out_dir / "top_20_weather_sensitive_outputs.csv", index=False)
    top_robust.to_csv(out_dir / "top_20_robust_outputs.csv", index=False)
    top_corr.to_csv(out_dir / "top_30_input_output_links.csv", index=False)

    plot_top_sensitive(sens_df, runs_df, out_dir, top_n=12)
    plot_output_variability_heatmap(sens_df, out_dir, top_n=20)
    plot_top_input_output_links(corr_df, out_dir, top_n=20)

    print("\nTop 10 weather-sensitive outputs:")
    cols_show = ["pretty_name", "cv", "min", "max"]
    print(sens_df[cols_show].head(10).to_string(index=False))

    print("\nTop 10 robust outputs:")
    print(robust_df[cols_show].head(10).to_string(index=False))

    print("\nTop 15 input-output relationships:")
    print(corr_df[["input_metric", "output_metric", "corr"]].head(15).to_string(index=False))

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()