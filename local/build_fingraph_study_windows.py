#!/usr/bin/env python3
"""Materialize reusable raw-log-return windows for the FinGraph comparison.

At every refit date the universe is the largest ``top_k`` eligible stocks by
*previous-day* market capitalisation.  A stock is eligible only when its full
lookback window of CRSP log returns is finite and in the source tradability
mask.  Windows end on t-1, so labels refreshed on t are available before the
strategy earns its t-to-t+1 payoff.

The output is deliberately method-agnostic: both FinGraph and KMeans read the
same 504 x 500 CSV input files.  Re-running either estimator never reloads or
rescans the full CIZ panel.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start", required=True, help="first refit date, inclusive")
    parser.add_argument("--end", required=True, help="last refit date, inclusive")
    parser.add_argument("--lookback", type=int, default=504)
    parser.add_argument("--refit-every", type=int, default=63)
    parser.add_argument("--top-k", type=int, default=500)
    return parser.parse_args()


def require_panel(root: Path) -> None:
    needed = [root / name for name in ("prices.pkl", "caps.pkl", "mask.pkl")]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing panel inputs: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    if args.lookback < 2 or args.refit_every < 1 or args.top_k < 2:
        raise ValueError("lookback >= 2, refit-every >= 1, and top-k >= 2 are required")
    require_panel(args.panel)
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}; choose a new run directory")

    args.out.mkdir(parents=True)
    inputs = args.out / "inputs"
    inputs.mkdir()

    # Keep only one float panel plus the cap panel in memory.  The original
    # source panel is far wider than the 500-name windows written below.
    print("loading prices and forming raw log returns", flush=True)
    prices = pd.read_pickle(args.panel / "prices.pkl")
    if (prices.to_numpy(dtype=np.float64, copy=False)[np.isfinite(prices.to_numpy(dtype=np.float64, copy=False))] <= 0).any():
        raise ValueError("prices contain non-positive finite values; cannot form log returns")
    log_returns = np.log(prices).diff()
    dates = pd.DatetimeIndex(log_returns.index)
    columns = log_returns.columns.astype(str)
    returns_np = np.ascontiguousarray(log_returns.to_numpy(dtype=np.float64, copy=False))
    del prices, log_returns
    gc.collect()

    print("loading market caps and tradability mask", flush=True)
    caps = pd.read_pickle(args.panel / "caps.pkl")
    mask = pd.read_pickle(args.panel / "mask.pkl")
    if not (dates.equals(pd.DatetimeIndex(caps.index)) and dates.equals(pd.DatetimeIndex(mask.index))):
        raise ValueError("prices, caps, and mask must share an identical calendar")
    if not (columns.equals(caps.columns.astype(str)) and columns.equals(mask.columns.astype(str))):
        raise ValueError("prices, caps, and mask must share identical columns")
    caps_np = caps.to_numpy(dtype=np.float64, copy=False)
    mask_np = mask.to_numpy(dtype=bool, copy=False)
    del caps, mask
    gc.collect()

    requested_start = pd.Timestamp(args.start)
    requested_end = pd.Timestamp(args.end)
    if requested_end < requested_start:
        raise ValueError("end must be on or after start")
    candidate_idx = np.flatnonzero((dates >= requested_start) & (dates <= requested_end))
    candidate_idx = candidate_idx[candidate_idx >= args.lookback + 1]
    refit_idx = candidate_idx[:: args.refit_every]
    if not len(refit_idx):
        raise ValueError("requested period has no date with a full lookback window")

    records: list[dict[str, object]] = []
    for count, t in enumerate(refit_idx, start=1):
        # Refit at date t from information through t-1.  The caps row is also
        # t-1, matching the existing source engine's prior-day universe rule.
        hist = returns_np[t - args.lookback : t]
        history_ok = np.isfinite(hist).all(axis=0) & mask_np[t - args.lookback : t].all(axis=0)
        prior_cap = caps_np[t - 1]
        eligible = np.flatnonzero(history_ok & np.isfinite(prior_cap) & (prior_cap > 0))
        if len(eligible) < args.top_k:
            raise RuntimeError(
                f"{dates[t].date()}: only {len(eligible)} eligible names, need {args.top_k}"
            )
        order = eligible[np.argsort(-prior_cap[eligible], kind="stable")[: args.top_k]]
        window = pd.DataFrame(hist[:, order], index=dates[t - args.lookback : t], columns=columns[order])
        refit_date = pd.Timestamp(dates[t])
        filename = f"{refit_date:%Y%m%d}.csv"
        window.to_csv(inputs / filename, index_label="date", float_format="%.12g")
        records.append(
            {
                "refit_date": refit_date.date().isoformat(),
                "input_file": f"inputs/{filename}",
                "window_start": pd.Timestamp(dates[t - args.lookback]).date().isoformat(),
                "window_end": pd.Timestamp(dates[t - 1]).date().isoformat(),
                "n_observations": len(window),
                "n_assets": len(order),
                "universe_cap_date": pd.Timestamp(dates[t - 1]).date().isoformat(),
            }
        )
        print(f"window {count}/{len(refit_idx)}: {refit_date.date()} ({len(order)} names)", flush=True)

    manifest = pd.DataFrame.from_records(records)
    manifest.to_csv(args.out / "windows.csv", index=False)
    metadata = {
        "source_panel": str(args.panel.resolve()),
        "input": "raw CRSP close-to-close log returns",
        "lookback_trading_days": args.lookback,
        "refit_every_trading_days": args.refit_every,
        "universe": f"top {args.top_k} eligible stocks by prior-day market capitalization",
        "information_cutoff": "window ends at t-1; label refreshed at t",
        "n_windows": len(manifest),
        "date_start": str(manifest.refit_date.iloc[0]),
        "date_end": str(manifest.refit_date.iloc[-1]),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(manifest)} reusable windows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
