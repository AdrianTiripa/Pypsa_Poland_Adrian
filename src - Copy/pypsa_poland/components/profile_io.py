from __future__ import annotations

from pathlib import Path
import calendar
import logging
import pandas as pd


logger = logging.getLogger(__name__)


PROFILE_KEY_TO_DYNAMIC_KIND = {
    "cop_multi": "cop",
    "heat_demand_multi": "heat",
}


def _looks_numeric_index(idx) -> bool:
    if isinstance(idx, pd.RangeIndex):
        return True

    try:
        if pd.api.types.is_numeric_dtype(idx):
            return True
    except Exception:
        pass

    s = pd.Index(idx).astype(str)
    sample = s[: min(len(s), 50)]
    numericish = sample.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False).mean()
    return numericish > 0.8


def _try_datetime_index(idx) -> pd.DatetimeIndex | None:
    if _looks_numeric_index(idx):
        return None

    dt = pd.to_datetime(idx, errors="coerce")
    if dt.isna().mean() > 0.1:
        return None

    return pd.DatetimeIndex(dt)


def _full_year_hours(year: int, keep_feb29: bool) -> int:
    if calendar.isleap(int(year)) and keep_feb29:
        return 8784
    return 8760


def _slice_to_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Return a one-year hourly table for the requested meteorological year.

    Supports either:
    - a datetime index spanning multiple years
    - a raw one-year table with no datetime index
    """
    dt_index = _try_datetime_index(df.index)

    if dt_index is not None:
        out = df.copy()
        out.index = dt_index
        out = out[out.index.year == int(year)]

        if out.empty:
            years = pd.Index(dt_index.year).unique().sort_values()
            raise ValueError(
                f"No rows found for year={year}. Available years in index: {list(years[:10])}"
                + (" ..." if len(years) > 10 else "")
            )
        return out.copy()

    return df.copy()


def _drop_last_24_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:-24].copy()


def _drop_feb29_slot_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the 24 hourly rows occupying the Feb-29 slot position
    (hours 1416..1439 in a 0-based hourly year table).
    """
    start = 59 * 24
    end = 60 * 24
    return pd.concat([df.iloc[:start], df.iloc[end:]], axis=0)


def _last_24_rows_are_all_zero(df: pd.DataFrame, tol: float = 1e-12) -> bool:
    """
    Detect non-leap files padded to 8784 rows by appending 24 trailing zero rows.
    """
    tail = df.iloc[-24:]
    vals = pd.to_numeric(tail.stack(), errors="coerce").dropna()
    if vals.empty:
        return False
    return (vals.abs() <= tol).all()


def _conform_to_full_year(df: pd.DataFrame, year: int, keep_feb29: bool) -> pd.DataFrame:
    """
    Make the table match the full hourly length for the requested year
    before any downsampling is applied.
    """
    out = _slice_to_year(df, year)
    expected_full = _full_year_hours(year, keep_feb29)

    if len(out) == expected_full:
        return out.copy()

    if len(out) == 8784 and expected_full == 8760:
        # Case 1: non-leap profile padded with 24 trailing zero rows
        if _last_24_rows_are_all_zero(out):
            logger.info(
                "Detected 8784-row padded profile for year=%s; dropping last 24 zero rows.",
                year,
            )
            out2 = _drop_last_24_rows(out)
            if len(out2) == 8760:
                return out2

        # Case 2: real leap-style profile -> remove Feb-29 slot
        logger.info(
            "Detected 8784-row leap-style profile for year=%s; dropping Feb-29 slot rows.",
            year,
        )
        out2 = _drop_feb29_slot_rows(out)
        if len(out2) == 8760:
            return out2

    raise ValueError(
        f"Full-year time series length is {len(out)}, expected {expected_full} for year={year} "
        f"(keep_feb29={keep_feb29})."
    )


def _apply_stepsize(df: pd.DataFrame, step: int) -> pd.DataFrame:
    step = int(step)
    if step <= 1:
        return df.copy()
    return df.iloc[::step].copy()


def conform_timeseries(df: pd.DataFrame, cfg: dict, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    year = int(cfg["snapshots"]["year"])
    keep_feb29 = bool(cfg["snapshots"].get("keep_feb29", False))
    step = int(cfg["snapshots"].get("stepsize", 1))

    out = _conform_to_full_year(df, year, keep_feb29)
    out = _apply_stepsize(out, step)

    if len(out) != len(snapshots):
        raise ValueError(
            f"Timeseries length after year selection and downsampling is {len(out)}, "
            f"but network snapshots length is {len(snapshots)} "
            f"(year={year}, keep_feb29={keep_feb29}, stepsize={step})."
        )

    out = out.copy()
    out.index = snapshots
    return out


def _resolve_dynamic_profile_path(cfg: dict, kind: str) -> Path:
    profiles_cfg = cfg.get("profiles", {})

    scenario = profiles_cfg.get("scenario")
    if not scenario:
        raise KeyError("cfg['profiles']['scenario'] is required for dynamic COP/heat files.")

    system_year = int(profiles_cfg.get("system_year", 2050))
    meteo_year = int(cfg["snapshots"]["year"])

    root = Path(cfg["paths"]["scenario_profiles_root"])
    kind_cfg = profiles_cfg.get(kind, {})
    subfolder = str(kind_cfg.get("folder", "")).strip()
    prefix = str(kind_cfg.get("prefix", "")).strip()

    base = root / scenario
    if subfolder:
        base = base / subfolder

    if not base.exists():
        raise FileNotFoundError(f"Dynamic profile folder does not exist: {base}")

    candidates = []

    if prefix:
        candidates.extend(base.glob(f"{prefix}_{meteo_year}_{system_year}.csv"))
        candidates.extend(base.glob(f"{prefix}*_{meteo_year}_{system_year}.csv"))

    candidates.extend(base.glob(f"*_{meteo_year}_{system_year}.csv"))

    unique = []
    seen = set()
    for p in candidates:
        if not p.is_file():
            continue
        if p.name.startswith("Neli2_sum_"):
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if not unique:
        raise FileNotFoundError(
            f"No {kind} file found in {base} for meteo_year={meteo_year}, system_year={system_year}."
        )

    exact = []
    if prefix:
        exact_name = f"{prefix}_{meteo_year}_{system_year}.csv"
        exact = [p for p in unique if p.name == exact_name]

    matches = exact or unique

    if len(matches) > 1:
        names = ", ".join(p.name for p in matches[:5])
        raise ValueError(
            f"Ambiguous {kind} match in {base} for meteo_year={meteo_year}, "
            f"system_year={system_year}: {names}"
        )

    return matches[0]


def _resolve_profile_path(cfg: dict, file_key: str) -> Path:
    files = cfg.get("files", {})
    dynamic_kind = PROFILE_KEY_TO_DYNAMIC_KIND.get(file_key)
    use_dynamic_first = (
        bool(cfg.get("profiles", {}).get("use_dynamic_heat_cop", False))
        and dynamic_kind is not None
    )

    static_path = None
    if file_key in files:
        static_path = Path(cfg["paths"]["profiles_folder"]) / files[file_key]

    if use_dynamic_first:
        return _resolve_dynamic_profile_path(cfg, dynamic_kind)

    if static_path is not None and static_path.exists():
        return static_path

    if dynamic_kind:
        return _resolve_dynamic_profile_path(cfg, dynamic_kind)

    expected = files.get(file_key, "<missing>")
    raise FileNotFoundError(
        f"Could not resolve file_key='{file_key}'. Checked static path "
        f"{Path(cfg['paths']['profiles_folder']) / str(expected)}"
        + (" and dynamic scenario-based resolution." if dynamic_kind else ".")
    )


def read_profile_csv(cfg: dict, file_key: str, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    path = _resolve_profile_path(cfg, file_key)

    dynamic_kind = PROFILE_KEY_TO_DYNAMIC_KIND.get(file_key)
    use_dynamic = (
        bool(cfg.get("profiles", {}).get("use_dynamic_heat_cop", False))
        and dynamic_kind is not None
    )

    if use_dynamic:
        # New COP / heat files are plain numeric matrices with no datetime index
        # and no useful header row.
        df = pd.read_csv(path, header=None)
    else:
        # Old files keep the old behavior
        df = pd.read_csv(path, index_col=0)

    return conform_timeseries(df, cfg, snapshots)


def read_excel_timeseries(
    path: str | Path,
    cfg: dict,
    snapshots: pd.DatetimeIndex,
    **read_kwargs,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, index_col=0, **read_kwargs)
    return conform_timeseries(df, cfg, snapshots)