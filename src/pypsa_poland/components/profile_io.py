from __future__ import annotations

from pathlib import Path
import pandas as pd


def _looks_numeric_index(idx) -> bool:
    """
    Return True if the index is mostly numeric-like (int/float or strings of digits).
    This catches common "1..8760" or "0..8759" hour counters that should NOT be parsed as datetimes.
    """
    if isinstance(idx, pd.RangeIndex):
        return True

    # Fast path: if dtype is numeric
    try:
        if pd.api.types.is_numeric_dtype(idx):
            return True
    except Exception:
        pass

    # If object dtype, test a sample for numeric strings
    s = pd.Index(idx).astype(str)
    sample = s[: min(len(s), 50)]
    numericish = sample.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False).mean()
    return numericish > 0.8


def _try_datetime_index(idx) -> pd.DatetimeIndex | None:
    """
    Attempt datetime parsing only if the index does NOT look numeric.
    Return None if parsing is not meaningful.
    """
    if _looks_numeric_index(idx):
        return None

    dt = pd.to_datetime(idx, errors="coerce", infer_datetime_format=True)

    # If most are NaT, it's not usable
    if dt.isna().mean() > 0.1:
        return None

    return pd.DatetimeIndex(dt)


def _ensure_one_year_8760(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Return exactly one year of hourly data (8760 rows, Feb 29 dropped if needed).

    Supports two cases:
      1) datetime index spanning multiple years -> select matching year.
      2) non-datetime index, but already a single-year hourly table length 8760 or 8784 -> use as-is.
    """
    dt_index = _try_datetime_index(df.index)

    if dt_index is not None:
        df2 = df.copy()
        df2.index = dt_index

        out = df2[df2.index.year == int(year)]
        if out.empty:
            years = pd.Index(df2.index.year).unique().sort_values()
            raise ValueError(
                f"No rows found for year={year}. Available years in index: {list(years[:10])}"
                + (" ..." if len(years) > 10 else "")
            )

        # Drop Feb 29 if present
        out = out.loc[~((out.index.month == 2) & (out.index.day == 29))]

        if len(out) != 8760:
            raise ValueError(f"Year slice length is {len(out)}, expected 8760 for year={year}.")
        return out

    # Non-datetime index: assume already one-year hourly data
    if len(df) == 8784:
        # Without a datetime index we can't locate Feb 29 precisely.
        # Fallback: truncate to 8760.
        df = df.iloc[:8760].copy()

    if len(df) != 8760:
        raise ValueError(
            f"Time series has non-datetime (numeric-like) index and length={len(df)}; "
            f"expected 8760 (or 8784). Fix the file to have a datetime index, or provide a 8760-row table."
        )

    return df.copy()


def _apply_stepsize(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    step = int(cfg["snapshots"].get("stepsize", 1))
    if step <= 1:
        return df
    return df.iloc[::step].copy()


def conform_timeseries(df: pd.DataFrame, cfg: dict, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    year = int(cfg["snapshots"]["year"])
    out = _ensure_one_year_8760(df, year)
    out = _apply_stepsize(out, cfg)

    if len(out) != len(snapshots):
        step = int(cfg["snapshots"].get("stepsize", 1))
        raise ValueError(
            f"Timeseries length {len(out)} != snapshots length {len(snapshots)} "
            f"(year={year}, stepsize={step})."
        )

    out = out.copy()
    out.index = snapshots
    return out


def read_profile_csv(cfg: dict, file_key: str, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    profiles_folder = Path(cfg["paths"]["profiles_folder"])
    fname = cfg["files"][file_key]
    path = profiles_folder / fname

    df = pd.read_csv(path, index_col=0)
    return conform_timeseries(df, cfg, snapshots)


def read_excel_timeseries(path: str | Path, cfg: dict, snapshots: pd.DatetimeIndex, **read_kwargs) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, index_col=0, **read_kwargs)
    return conform_timeseries(df, cfg, snapshots)