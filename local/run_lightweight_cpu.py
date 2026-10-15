#!/usr/bin/env python3
"""T-aligned CPU KMeans stat-arbitrage benchmark.

Run this same program twice for the controlled comparison:

    # market-only residuals: [intercept, market]
    python local/run_lightweight_cpu.py ... --out .../kmeans_market

    # market + own industry residuals: [intercept, market, own SIC2 industry]
    python local/run_lightweight_cpu.py ... --include-industry --out .../kmeans_market_industry

``--include-industry`` is the only model switch.  Both variants use the same
Top-500 universe, t-close data convention, 60-day window, daily KMeans (K=30),
seed, direct labels, five-day trading signal, and unmodified ``naive_stream``.

At row t, the input return, DlyCap, SICCD and industry return all refer to t
close.  A position formed at t is paid on the next return by ``naive_stream``.
The t-aligned SIC cache is compact: it contains only securities that ever
enter Top-500, and is built by ``build_t_aligned_sic_cache.py``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "local"))
from kmeans_engine import naive_stream  # noqa: E402

SPY = "SPY"


def log(message: str) -> None:
    print(f"[cpu-replication] {message}", flush=True)


def digest(values: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in values.astype(str):
        h.update(value.encode())
        h.update(b"\0")
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True, type=Path, help="compatibility panel with prices.pkl")
    p.add_argument("--sic-cache", required=True, type=Path, help="compact cache made by build_t_aligned_sic_cache.py")
    p.add_argument("--out", required=True, type=Path, help="new result directory")
    p.add_argument("--include-industry", action="store_true", help="add one own-SIC2 industry return regressor")
    p.add_argument("--start", default="1997-05-23")
    p.add_argument("--end", default="2024-12-30")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--clusters", type=int, default=30)
    p.add_argument("--feature-window", type=int, default=60)
    p.add_argument("--beta-window", type=int, default=60)
    p.add_argument("--signal-window", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-init", type=int, default=4)
    p.add_argument("--payoff", choices=["raw", "residual"], default="raw")
    return p.parse_args()


def history_rows(a: argparse.Namespace) -> int:
    return a.beta_window + max(a.feature_window, 21, a.signal_window) + 5


def load_cache(root: Path, a: argparse.Namespace) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    required = ["metadata.json", "dates.npy", "columns.npy", "member.npy", "cap_t.npy", "sic2.npy"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete SIC cache {root}: missing {', '.join(missing)}")
    meta = json.loads((root / "metadata.json").read_text())
    if not meta.get("complete"):
        raise ValueError("SIC cache was not completed")
    expected = {"start": a.start, "end": a.end, "top_k": a.top_k}
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"SIC cache {field}={meta.get(field)!r}, expected {value!r}")
    if int(meta.get("history_rows", -1)) < history_rows(a):
        raise ValueError(
            f"SIC cache has {meta.get('history_rows')} history rows; runner requires at least {history_rows(a)}"
        )
    dates = pd.DatetimeIndex(np.load(root / "dates.npy", mmap_mode="r"))
    columns = np.load(root / "columns.npy", allow_pickle=True).astype(str)
    member = np.load(root / "member.npy", mmap_mode="r")
    cap_t = np.load(root / "cap_t.npy", mmap_mode="r")
    sic2 = np.load(root / "sic2.npy", mmap_mode="r")
    expected_shape = (len(dates), len(columns))
    if member.shape != expected_shape or cap_t.shape != expected_shape or sic2.shape != expected_shape:
        raise ValueError("SIC cache arrays do not share a date x security shape")
    if meta.get("columns_sha256") != digest(columns):
        raise ValueError("SIC cache column digest mismatch")
    return dates, columns, member, cap_t, sic2, meta


def t_aligned_returns(panel: Path, dates: pd.DatetimeIndex, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CIZ DlyRet[t] represented by compatibility prices.pct_change()[t+1]."""
    prices_path = panel / "prices.pkl"
    if not prices_path.is_file():
        raise FileNotFoundError(prices_path)
    log("loading only Top-500-union prices plus SPY")
    prices = pd.read_pickle(prices_path)
    grid = pd.DatetimeIndex(prices.index)
    first = int(grid.get_indexer([dates[0]])[0])
    last = int(grid.get_indexer([dates[-1]])[0])
    if first < 0 or last < 0 or last + 1 >= len(grid):
        raise ValueError("panel dates do not support t-aligned return extraction")
    wanted = list(columns) + [SPY]
    if any(name not in prices.columns for name in wanted):
        missing = [name for name in wanted if name not in prices.columns]
        raise ValueError(f"prices panel misses cache columns, e.g. {missing[:3]}")
    # pct_change at panel row t+1 is the underlying CIZ DlyRet[t].
    px = prices.iloc[first : last + 2].loc[:, wanted]
    del prices
    pct = px.pct_change(fill_method=None).iloc[1:]
    if len(pct) != len(dates):
        raise ValueError("t-aligned return length mismatch")
    values = pct.to_numpy(dtype=np.float64, copy=True)
    return values[:, :-1], values[:, -1]


def industry_panel(raw: np.ndarray, cap_t: np.ndarray, member: np.ndarray, sic2: np.ndarray) -> np.ndarray:
    """Each stock receives its SIC2's cap-weighted return inside Top-500[t]."""
    n_dates, n_columns = raw.shape
    result = np.full((n_dates, n_columns), np.nan, dtype=np.float64)
    for t in range(n_dates):
        code = np.asarray(sic2[t], dtype=np.int16)
        valid = member[t] & np.isfinite(raw[t]) & np.isfinite(cap_t[t]) & (cap_t[t] > 0) & (code > 0)
        if not valid.any():
            continue
        numerator = np.bincount(code[valid], weights=cap_t[t, valid] * raw[t, valid], minlength=100)
        denominator = np.bincount(code[valid], weights=cap_t[t, valid], minlength=100)
        factor = np.full(100, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=factor, where=denominator > 0)
        has_code = (code > 0) & (code < 100)
        result[t, has_code] = factor[code[has_code]]
    return result


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    total = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    total[0] = 0.0
    np.cumsum(values, axis=0, dtype=np.float64, out=total[1:])
    return total[window:] - total[:-window]


def residuals_with_intercept(raw: np.ndarray, market: np.ndarray, window: int, industry: np.ndarray | None) -> np.ndarray:
    """Rolling OLS slopes with intercept fitted, then omitted from stored residuals."""
    if raw.ndim != 2 or len(market) != raw.shape[0]:
        raise ValueError("invalid return matrix / market vector")
    market_grid = market[:, None]
    valid = np.isfinite(raw) & np.isfinite(market_grid)
    if industry is not None:
        if industry.shape != raw.shape:
            raise ValueError("industry panel shape mismatch")
        valid &= np.isfinite(industry)
    y = np.where(valid, raw, 0.0)
    m = np.where(valid, market_grid, 0.0)
    count = rolling_sum(valid.astype(np.float64), window)
    sy, sm = rolling_sum(y, window), rolling_sum(m, window)
    syy, smm, sym = rolling_sum(y * y, window), rolling_sum(m * m, window), rolling_sum(y * m, window)
    offset = window - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        cmm = smm - sm * sm / count
        cmy = sym - sm * sy / count
        if industry is None:
            beta_m = cmy / cmm
            # alpha is deliberately estimated but not subtracted from residuals.
            _alpha = (sy - beta_m * sm) / count
            out = raw[offset:] - beta_m * market_grid[offset:]
            ok = (count == window) & np.isfinite(cmm) & (np.abs(cmm) > 1e-16)
        else:
            i = np.where(valid, industry, 0.0)
            si = rolling_sum(i, window)
            sii, smi, siy = rolling_sum(i * i, window), rolling_sum(m * i, window), rolling_sum(y * i, window)
            cii = sii - si * si / count
            cmi = smi - sm * si / count
            ciy = siy - sy * si / count
            determinant = cmm * cii - cmi * cmi
            beta_m = (cmy * cii - ciy * cmi) / determinant
            beta_i = (ciy * cmm - cmy * cmi) / determinant
            _alpha = (sy - beta_m * sm - beta_i * si) / count
            out = raw[offset:] - beta_m * market_grid[offset:] - beta_i * industry[offset:]
            ok = (count == window) & np.isfinite(determinant) & (np.abs(determinant) > 1e-16)
    result = np.full(raw.shape, np.nan, dtype=np.float64)
    result[offset:] = out
    result[offset:][~ok] = np.nan
    return result


def build_kmeans_cache(
    dates: pd.DatetimeIndex, residuals: np.ndarray, cap_t: np.ndarray, member: np.ndarray,
    columns: np.ndarray, a: argparse.Namespace, cache_start: pd.Timestamp, cache_end: pd.Timestamp,
) -> dict[pd.Timestamp, dict[str, object]]:
    """Build only one bounded cache segment; daily refits make segments composable."""
    cache: dict[pd.Timestamp, dict[str, object]] = {}
    selected_days = np.flatnonzero((dates >= cache_start) & (dates <= cache_end))
    selected_days = selected_days[selected_days >= a.feature_window]
    started = time.monotonic()
    for count, t in enumerate(selected_days, start=1):
        candidates = np.flatnonzero(member[t])
        finite = np.isfinite(residuals[t - a.feature_window : t, candidates]).all(axis=0)
        candidates = candidates[finite]
        if len(candidates) < a.clusters + 5:
            continue
        candidates = candidates[np.argsort(-cap_t[t, candidates], kind="stable")]
        features = residuals[t - a.feature_window : t, candidates]
        labels = KMeans(a.clusters, random_state=a.seed, n_init=a.n_init).fit_predict(features.T)
        cache[pd.Timestamp(dates[t])] = {"universe": list(columns[candidates]), "labels": np.asarray(labels, dtype=np.int32)}
        if count % 250 == 0 or count == len(selected_days):
            log(f"partitions {count}/{len(selected_days)} through {dates[t].date()} ({time.monotonic()-started:.0f}s)")
    return cache


def performance_metrics(pnl: pd.Series) -> dict[str, object]:
    pnl = pnl.dropna().astype(float)
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    vol = float(pnl.std(ddof=1) * np.sqrt(252))
    ann_mean = float(pnl.mean() * 252)
    return {
        "days": int(len(pnl)), "start": str(pnl.index.min().date()), "end": str(pnl.index.max().date()),
        "ann_mean_return": ann_mean, "ann_volatility": vol,
        "sharpe": float(ann_mean / vol) if vol else np.nan,
        "cumulative_simple_return": float(cumulative.iloc[-1]), "max_drawdown": float(drawdown.min()),
    }


def main() -> None:
    a = parse_args()
    if a.top_k != 500:
        raise ValueError("naive_stream's source trading engine is fixed at Top-500")
    if a.clusters < 1 or a.top_k <= a.clusters:
        raise ValueError("require top-k > clusters >= 1")
    if a.out.exists():
        raise FileExistsError(f"output already exists: {a.out}")
    a.out.mkdir(parents=True)
    dates, columns, member, cap_t, sic2, cache_metadata = load_cache(a.sic_cache, a)
    raw, market = t_aligned_returns(a.panel, dates, columns)
    log(f"input: {len(dates):,} t-close dates x {len(columns):,} ever-Top-500 securities")
    industry = None
    if a.include_industry:
        log("forming own-SIC2 returns from each t-day Top-500 cross-section")
        industry = industry_panel(raw, cap_t, member, sic2)
    model = "kmeans_market_industry" if a.include_industry else "kmeans_market"
    log(f"estimating 60-day OLS with intercept: {model}")
    residuals = residuals_with_intercept(raw, market, a.beta_window, industry)
    # naive_stream selects caps[d-1].  Shift cap_t forward one row so its date d
    # selects exactly Top-500 by DlyCap[d], while its unchanged payoff stays d+1.
    engine_caps = np.full_like(cap_t, np.nan)
    engine_caps[:-1] = cap_t[1:]
    positions = {str(name): i for i, name in enumerate(columns)}
    partition_dir = a.out / "partitions"
    partition_dir.mkdir()
    pnl_parts: list[pd.Series] = []
    total_snapshots = 0
    target_start, target_end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    target_years = sorted(set(dates[(dates >= target_start) & (dates <= target_end)].year))
    for year in target_years:
        in_year = np.flatnonzero((dates.year == year) & (dates >= target_start) & (dates <= target_end))
        if not len(in_year):
            continue
        segment_start, segment_end = pd.Timestamp(dates[in_year[0]]), pd.Timestamp(dates[in_year[-1]])
        # A missing daily cache may fall back at most ten calendar days.  Include
        # that short overlap so year boundaries are bit-identical to one full cache.
        cache_start = segment_start - pd.Timedelta(days=10)
        cache = build_kmeans_cache(dates, residuals, cap_t, member, columns, a, cache_start, segment_end)
        total_snapshots += len(cache)
        with (partition_dir / f"year={year}.pkl").open("wb") as handle:
            pickle.dump(cache, handle, protocol=4)
        pnl_part = naive_stream(
            cache, "labels", dates, positions, engine_caps, columns,
            R_trade=residuals, R_affil=residuals,
            R_pay=raw if a.payoff == "raw" else None,
            clusters=a.clusters, beta_window=a.beta_window, signal_window=a.signal_window,
            date_start=segment_start, date_end=segment_end,
        )
        pnl_parts.append(pnl_part)
        log(f"{year}: {len(cache)} cached partitions; {len(pnl_part)} PnL days")
        del cache, pnl_part
        gc.collect()
    if not pnl_parts:
        raise ValueError("no yearly PnL segments were produced")
    pnl = pd.concat(pnl_parts).sort_index()
    pnl.rename("daily_pnl").to_frame().to_csv(a.out / f"{model}_daily_pnl.csv", index_label="signal_date")
    metrics = performance_metrics(pnl)
    metrics.update({"model": model, "snapshots": total_snapshots})
    pd.DataFrame([metrics]).set_index("model").to_csv(a.out / "summary.csv")
    metadata = {
        "model": model,
        "include_industry": bool(a.include_industry),
        "regression": "rolling OLS [intercept, SPY]" + (" + [own SIC2 industry]" if a.include_industry else ""),
        "stored_residual": "r - beta_market*market" + (" - beta_industry*own_industry" if a.include_industry else "") + "; fitted intercept omitted",
        "t_alignment": "return[t] = CIZ DlyRet[t]; cap[t] = CIZ DlyCap[t]; SIC[t] = CIZ SICCD[t]",
        "industry_return": "within Top-500[t], own SIC2, DlyCap[t]-weighted DlyRet[t], inclusive of security",
        "trading": "signal and universe use t-close data; unmodified naive_stream pays raw return[t+1]",
        "execution_memory": "daily partitions are cached by year with a 10-calendar-day overlap; this is equivalent to the full daily cache while bounding RAM",
        "top_k": a.top_k, "clusters": a.clusters, "feature_window": a.feature_window,
        "beta_window": a.beta_window, "signal_window": a.signal_window, "seed": a.seed,
        "n_init": a.n_init, "payoff": a.payoff, "sic_cache": str(a.sic_cache.resolve()),
        "sic_cache_metadata": cache_metadata,
    }
    (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log(f"{model}: {metrics['days']} return days; Sharpe={metrics['sharpe']:.4f}; wrote {a.out}")


if __name__ == "__main__":
    main()
