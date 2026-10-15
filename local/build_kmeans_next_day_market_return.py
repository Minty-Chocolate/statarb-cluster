#!/usr/bin/env python3
"""Build the KMeans signal-date -> next-day Top-500 market-return series.

At a signal date t, KMeans selects the DlyCap[t] Top-500 universe.  The next
trading day's raw return is therefore benchmarked with those t-close capital
weights.  This is the market leg to subtract from each saved KMeans t+1 payoff
when displaying market-excess deciles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
from run_lightweight_cpu import t_aligned_returns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--sic-cache", required=True, type=Path)
    parser.add_argument("--books", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cache = args.sic_cache
    dates = pd.DatetimeIndex(np.load(cache / "dates.npy", mmap_mode="r"))
    columns = np.load(cache / "columns.npy", allow_pickle=True).astype(str)
    member = np.load(cache / "member.npy", mmap_mode="r")
    cap_t = np.load(cache / "cap_t.npy", mmap_mode="r")
    raw, _ = t_aligned_returns(args.panel, dates, columns)

    market = np.full(len(dates), np.nan, dtype=float)
    for t in range(len(dates) - 1):
        valid = member[t] & np.isfinite(cap_t[t]) & (cap_t[t] > 0) & np.isfinite(raw[t + 1])
        if valid.any():
            market[t] = np.average(raw[t + 1, valid], weights=cap_t[t, valid])
    frame = pd.DataFrame({
        "signal_date": dates[:-1],
        "payoff_date": dates[1:],
        "market_return": market[:-1],
    }).dropna()

    books = pd.read_csv(args.books, compression="infer", parse_dates=["signal_date", "payoff_date"])
    checks = books.loc[:, ["signal_date", "payoff_date"]].drop_duplicates().merge(
        frame, on=["signal_date", "payoff_date"], how="left", validate="one_to_one"
    )
    if checks.market_return.isna().any():
        raise RuntimeError("some saved KMeans books have no matching next-day market return")
    frame.to_csv(args.out, index=False)
    args.out.with_suffix(".metadata.json").write_text(json.dumps({
        "timing": "weights=DlyCap[t] within Top-500[t]; market_return=raw return[t+1]",
        "purpose": "subtract from each KMeans raw payoff for market-excess deciles",
        "rows": int(len(frame)),
    }, indent=2) + "\n")
    print(f"wrote {args.out}: {len(frame)} signal-date market returns")


if __name__ == "__main__":
    main()
