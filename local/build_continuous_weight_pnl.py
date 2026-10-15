#!/usr/bin/env python3
"""Evaluate a daily 1%-winsorized z-score, gross-one long/short book."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
from decile_analysis import standardized_long_short_weights, weighted_daily_returns  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--books", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--score-column", default="raw_signal")
    p.add_argument("--winsor", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    books = pd.read_csv(args.books, compression="infer", parse_dates=["signal_date", "payoff_date"], dtype={"permno": str})
    weighted = standardized_long_short_weights(books, score_col=args.score_column, winsor=args.winsor)
    daily = weighted_daily_returns(weighted).to_frame()
    daily.to_csv(args.out, index_label="signal_date")
    exposure = weighted.groupby("signal_date", sort=True).weight.agg(net="sum", gross=lambda x: x.abs().sum())
    if not np.allclose(exposure.net, 0.0, atol=1e-12) or not np.allclose(exposure.gross, 1.0, atol=1e-12):
        raise RuntimeError("continuous book failed net/gross invariants")
    metadata_path = args.out.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps({
        "books": str(args.books.resolve()),
        "score_column": args.score_column,
        "weighting": f"{args.winsor:.1%} two-sided winsorization; cross-sectional z-score; de-mean; L1 normalize",
        "net_exposure": 0.0,
        "gross_exposure": 1.0,
        "trading_lag": "signal_date=t close; payoff_return=t+1 realised return",
        "days": int(len(daily)),
    }, indent=2) + "\n")
    print(f"wrote {args.out}: {len(daily)} days")


if __name__ == "__main__":
    main()
