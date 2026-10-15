#!/usr/bin/env python3
"""Attach realised t+1 market-excess returns to saved graph-local weights.

The graph and weights are already saved.  This reads the compatible panel once
to recover each security's realised payoff; it neither refits a graph nor
rebuilds the graph-local signal.  The final check requires the security-level
book to reproduce the preserved graph-local PnL exactly.
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
from backtest_fingraph_signals import market_return_from_prices, top_universe_from_caps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--backtest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.backtest / "daily_weights.csv.gz"
    daily_path = args.backtest / "graph_local_daily.csv"
    output = args.out or args.backtest / "graph_local_daily_books.csv.gz"
    if output.exists():
        raise FileExistsError(output)

    books = pd.read_csv(
        weights_path, compression="gzip", dtype={"permno": str}, parse_dates=["date", "payoff_date"]
    )
    books = books.loc[
        books.method.eq("graph_local"), ["date", "payoff_date", "permno", "raw_signal", "weight"]
    ].copy()
    names = pd.Index(books.permno.unique(), dtype=str)

    # This mirrors backtest_fingraph_signals.py but retains only graph nodes once
    # the top-500 market-excess payoff has been calculated.
    caps = pd.read_pickle(args.panel / "caps.pkl")
    positions, cap_weights = top_universe_from_caps(caps, top_k=500)
    columns, calendar = caps.columns, pd.DatetimeIndex(caps.index)
    del caps
    prices = pd.read_pickle(args.panel / "prices.pkl")
    if not (prices.index.equals(calendar) and prices.columns.equals(columns)):
        raise ValueError("price and cap panels do not share the same calendar/columns")
    market = market_return_from_prices(prices, positions, cap_weights)
    compact_prices = prices.loc[:, names]
    del prices, positions, cap_weights
    excess = compact_prices.pct_change(fill_method=None).to_numpy(dtype=np.float64, copy=False) - market[:, None]

    asset_position = {name: i for i, name in enumerate(names)}
    date_position = {date: i for i, date in enumerate(calendar)}
    signal_rows = books.date.map(date_position).to_numpy(dtype=np.int64)
    asset_rows = books.permno.map(asset_position).to_numpy(dtype=np.int64)
    payoff_rows = signal_rows + 1
    if np.any(payoff_rows >= len(calendar)):
        raise ValueError("saved book has no next-day payoff")
    if not np.array_equal(calendar.take(payoff_rows).to_numpy(), books.payoff_date.to_numpy()):
        raise ValueError("saved payoff dates do not match the panel calendar")
    books["payoff_return"] = excess[payoff_rows, asset_rows]
    if books.payoff_return.isna().any():
        raise ValueError("missing reconstructed graph-local payoff")

    recreated = (books.weight * books.payoff_return).groupby(books.date, sort=True).sum()
    saved = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").daily_excess_return
    error = recreated.reindex(saved.index) - saved
    if error.isna().any() or not np.allclose(error.to_numpy(), 0.0, atol=1e-14, rtol=0.0):
        raise RuntimeError("security-level payoff book does not reproduce graph-local PnL")

    books.rename(columns={"date": "signal_date"}).to_csv(output, index=False, compression="gzip")
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "timing": "signal_date=t close; payoff_return=market-excess return[t+1]",
        "pnl_check_max_abs_error": float(np.abs(error).max()),
        "weighting": "preserved graph-local 1% winsorized z-score, gross 1/net 0",
    }, indent=2) + "\n")
    print(f"wrote {output}; PnL check max error={np.abs(error).max():.3g}")


if __name__ == "__main__":
    main()
