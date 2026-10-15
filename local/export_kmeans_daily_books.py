#!/usr/bin/env python3
"""Export realised daily KMeans books from saved partitions without reclustering.

The saved annual partition files are the already-computed KMeans output.  This
script only reruns the deterministic trading engine to write each t-close book
and its t+1 raw return; it never refits KMeans or overwrites a benchmark.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
from kmeans_engine import naive_stream  # noqa: E402
from run_lightweight_cpu import (  # noqa: E402
    industry_panel,
    load_cache,
    residuals_with_intercept,
    t_aligned_returns,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--sic-cache", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path, help="saved kmeans_market or kmeans_market_industry output")
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/<model>_daily_books.csv.gz")
    return p.parse_args()


def run_args(metadata: dict) -> SimpleNamespace:
    fields = ("start", "end", "top_k", "clusters", "feature_window", "beta_window", "signal_window", "payoff")
    cache_metadata = metadata.get("sic_cache_metadata", {})
    resolved = {name: metadata.get(name, cache_metadata.get(name)) for name in fields}
    missing = [name for name in fields if resolved[name] is None]
    if missing:
        raise ValueError(f"run metadata is missing {missing}")
    return SimpleNamespace(**resolved)


def main() -> None:
    args = parse_args()
    metadata_path = args.run / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    config = run_args(metadata)
    model = str(metadata["model"])
    output = args.out or args.run / f"{model}_daily_books.csv.gz"
    if output.exists():
        raise FileExistsError(f"output exists: {output}")

    dates, columns, member, cap_t, sic2, _ = load_cache(args.sic_cache, config)
    raw, market = t_aligned_returns(args.panel, dates, columns)
    industry = industry_panel(raw, cap_t, member, sic2) if metadata.get("include_industry") else None
    residuals = residuals_with_intercept(raw, market, config.beta_window, industry)
    engine_caps = np.full_like(cap_t, np.nan)
    engine_caps[:-1] = cap_t[1:]
    positions = {str(name): i for i, name in enumerate(columns)}
    partition_dir = args.run / "partitions"
    partitions = sorted(partition_dir.glob("year=*.pkl"))
    if not partitions:
        raise FileNotFoundError(f"no saved partitions in {partition_dir}")

    daily_pnl: list[pd.Series] = []
    written_days = 0
    with gzip.open(output, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_date", "payoff_date", "permno", "raw_signal", "weight", "payoff_return"])

        def write_book(signal_date, payoff_date, tickers, raw_signals, weights, realised):
            nonlocal written_days
            writer.writerows(
                (signal_date.date(), payoff_date.date(), ticker, float(signal), float(weight), float(payoff))
                for ticker, signal, weight, payoff in zip(tickers, raw_signals, weights, realised, strict=True)
            )
            written_days += 1

        for path in partitions:
            year = int(path.stem.split("=")[1])
            with path.open("rb") as handle:
                partitions_for_year = pickle.load(handle)
            in_year = np.flatnonzero((dates.year == year) & (dates >= pd.Timestamp(config.start)) & (dates <= pd.Timestamp(config.end)))
            if not len(in_year):
                continue
            pnl = naive_stream(
                partitions_for_year, "labels", dates, positions, engine_caps, columns,
                R_trade=residuals, R_affil=residuals,
                R_pay=raw if config.payoff == "raw" else None,
                clusters=config.clusters, beta_window=config.beta_window, signal_window=config.signal_window,
                date_start=pd.Timestamp(dates[in_year[0]]), date_end=pd.Timestamp(dates[in_year[-1]]),
                book_callback=write_book,
            )
            daily_pnl.append(pnl)
            print(f"exported {year}: {len(pnl)} books", flush=True)

    exported = pd.concat(daily_pnl).sort_index()
    pnl_path = args.run / f"{model}_daily_pnl.csv"
    reference = pd.read_csv(pnl_path, parse_dates=["signal_date"]).set_index("signal_date").iloc[:, 0]
    comparison = exported.reindex(reference.index) - reference
    if exported.index.has_duplicates or comparison.isna().any() or not np.allclose(comparison.to_numpy(), 0.0, atol=1e-14, rtol=0.0):
        raise RuntimeError("exported book PnL does not exactly match the saved strategy PnL")
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "model": model,
        "source_run": str(args.run.resolve()),
        "timing": "signal_date=t close; payoff_return=raw return[t+1]",
        "book": "unaltered naive_stream weights; source engine has gross exposure 2 when both sides are populated",
        "days": written_days,
        "pnl_check_max_abs_error": float(np.abs(comparison).max()),
    }, indent=2) + "\n")
    print(f"wrote {output} ({written_days} daily books)")


if __name__ == "__main__":
    main()
