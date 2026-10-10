"""Phase 1 - Data acquisition and cleaning.

Downloads OHLCV history from Yahoo Finance with ``yfinance``, caches it as a CSV
so repeated runs work offline, and cleans it into a tidy, gap-free DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .config import DATA_DIR

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, start: str, end: str | None):
    end_label = end or date.today().isoformat()
    return DATA_DIR / f"{ticker.upper()}_{start}_{end_label}.csv"


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the rest of the package works offline

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # prices adjusted for splits & dividends
        progress=False,
        threads=False,
    )
    if raw is None:
        return pd.DataFrame()
    # Newer yfinance versions return (Price, Ticker) MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def fetch_data(ticker: str, start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return raw OHLCV data, using the CSV cache when available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end)

    if path.exists() and not refresh:
        print(f"[data] Loaded cached data: {path.name}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print(f"[data] Downloading {ticker.upper()} from Yahoo Finance ({start} -> {end or 'today'}) ...")
    error: Exception | None = None
    try:
        raw = _download(ticker, start, end)
    except Exception as exc:  # network problems, API changes, etc.
        raw, error = pd.DataFrame(), exc

    if raw.empty:
        # Fall back to the most recent cache for this ticker, if any.
        cached = sorted(DATA_DIR.glob(f"{ticker.upper()}_*.csv"), key=lambda p: p.stat().st_mtime)
        if cached:
            print(f"[data] Download failed ({error or 'empty response'}); using cache {cached[-1].name}")
            return pd.read_csv(cached[-1], index_col=0, parse_dates=True)
        raise RuntimeError(
            f"Could not download data for '{ticker}'. Check the ticker symbol and your "
            f"internet connection. Details: {error or 'Yahoo Finance returned no rows.'}"
        )

    # Fetch Macro Economic Benchmark (Phase 1 Upgrade)
    benchmark_ticker = "^NSEI" if ticker.endswith(".NS") or ticker.endswith(".BO") else "^GSPC"
    try:
        bench_raw = _download(benchmark_ticker, start, end)
        if not bench_raw.empty:
            raw["Macro_Close"] = bench_raw["Close"]
    except Exception:
        pass  # If macro fails, we just won't have it

    raw.to_csv(path)
    print(f"[data] Saved {len(raw):,} rows to {path.name}")
    return raw


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_prices_fixed: int = 0
    missing_values_filled: int = 0
    rows_dropped: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"rows in / out          : {self.rows_in:,} / {self.rows_out:,}",
            f"duplicate dates removed: {self.duplicates_removed}",
            f"invalid prices (<=0)   : {self.invalid_prices_fixed}",
            f"missing values filled  : {self.missing_values_filled}",
            f"unrecoverable rows     : {self.rows_dropped}",
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join("    " + line for line in lines)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and repair raw OHLCV data."""
    report = CleaningReport(rows_in=len(df))
    missing_cols = [c for c in OHLCV if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing columns: {missing_cols}")

    out = df[OHLCV].copy()

    # Index: datetime, timezone-naive, sorted, unique.
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "Date"
    dupes = out.index.duplicated(keep="last")
    report.duplicates_removed = int(dupes.sum())
    out = out[~dupes].sort_index()

    # Types: force numeric, anything unparsable becomes NaN.
    out = out.apply(pd.to_numeric, errors="coerce")

    # Prices must be strictly positive; volume must be non-negative.
    bad_prices = out[PRICE_COLUMNS] <= 0
    report.invalid_prices_fixed = int(bad_prices.to_numpy().sum())
    out[PRICE_COLUMNS] = out[PRICE_COLUMNS].mask(bad_prices)
    out.loc[out["Volume"] < 0, "Volume"] = np.nan

    # Fill gaps: carry the last known price forward (no look-ahead).
    n_missing = int(out.isna().to_numpy().sum())
    out[PRICE_COLUMNS] = out[PRICE_COLUMNS].ffill()
    out["Volume"] = out["Volume"].fillna(0)
    unfillable = out.isna().any(axis=1)  # e.g. leading rows with no earlier price
    report.missing_values_filled = n_missing - int(out[unfillable].isna().to_numpy().sum())
    report.rows_dropped = int(unfillable.sum())
    out = out[~unfillable]

    # Make High/Low consistent with Open/Close.
    inconsistent = (out["High"] < out[["Open", "Close"]].max(axis=1)) | (
        out["Low"] > out[["Open", "Close"]].min(axis=1)
    )
    if inconsistent.any():
        report.notes.append(f"repaired {int(inconsistent.sum())} rows with inconsistent High/Low")
        out["High"] = out[PRICE_COLUMNS].max(axis=1)
        out["Low"] = out[PRICE_COLUMNS].min(axis=1)

    report.rows_out = len(out)
    if report.rows_out == 0:
        raise ValueError("No usable rows remain after cleaning.")
    return out, report

def get_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data like P/E, EPS, Market Cap."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "PE": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "MarketCap": info.get("marketCap", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Name": info.get("longName", ticker)
        }
    except Exception:
        return {}
        
def get_vix() -> float:
    """Fetch current India VIX (or US VIX) to measure Market Fear."""
    import yfinance as yf
    try:
        vix = yf.download("^INDIAVIX", period="1d", progress=False)
        if vix.empty:
            vix = yf.download("^VIX", period="1d", progress=False) # Fallback to US
        return float(vix["Close"].iloc[-1])
    except Exception:
        return 0.0
