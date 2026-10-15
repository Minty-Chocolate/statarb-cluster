#!/usr/bin/env python3
"""Backtest the two approved FinGraph mean-reversion signals.

The graph is estimated outside this program on a rolling historical window.
This program never refits it: it selects the applicable cached graph on each
signal date, forms either a graph-local or graph-community relative-return
signal, and earns the following day's market-excess return.

``graph_local``
    Negative of a stock's trailing excess return less the adjacency-weighted
    mean of the trailing excess returns of its graph neighbours.

``graph_community``
    Negative of a stock's trailing excess return less the equal-weighted mean
    of its Fast-Greedy community.  Communities are the exact second-stage
    operation in FinGraph's crypto example.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("graph_local", "graph_community")


@dataclass(frozen=True)
class GraphSnapshot:
    refit_date: pd.Timestamp
    adjacency_file: Path
    community_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path, help="compatibility panel with prices.pkl and caps.pkl")
    parser.add_argument("--graphs", required=True, type=Path, help="output of export_fingraph_artifacts.R")
    parser.add_argument("--out", required=True, type=Path, help="new result directory")
    parser.add_argument("--start", required=True, help="first signal date, inclusive")
    parser.add_argument("--end", required=True, help="last signal date, inclusive")
    parser.add_argument("--signal-window", type=int, default=5, help="trailing daily excess-return observations")
    parser.add_argument("--top-k", type=int, default=500, help="prior-day market-cap universe")
    parser.add_argument("--winsor", type=float, default=0.01, help="two-sided winsorization fraction before z-scoring")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    return parser.parse_args()


def load_snapshots(graph_root: Path) -> tuple[list[GraphSnapshot], dict[pd.Timestamp, np.ndarray]]:
    manifest_path = graph_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, parse_dates=["refit_date"]).sort_values("refit_date")
    required = {"refit_date", "adjacency_file", "community_file"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"graph manifest is missing columns: {sorted(missing)}")
    if manifest.empty or manifest.refit_date.duplicated().any():
        raise ValueError("graph manifest must have one or more unique refit dates")
    snapshots: list[GraphSnapshot] = []
    names: dict[pd.Timestamp, np.ndarray] = {}
    for row in manifest.itertuples(index=False):
        refit_date = pd.Timestamp(row.refit_date)
        adjacency_file = graph_root / row.adjacency_file
        community_file = graph_root / row.community_file
        if not adjacency_file.is_file() or not community_file.is_file():
            raise FileNotFoundError(f"missing graph export for {refit_date.date()}")
        community = pd.read_csv(community_file, dtype={"permno": str})
        if community.permno.duplicated().any() or not {"permno", "community_id"}.issubset(community):
            raise ValueError(f"invalid community export: {community_file}")
        snapshots.append(GraphSnapshot(refit_date, adjacency_file, community_file))
        names[refit_date] = community.permno.to_numpy(dtype=str)
    return snapshots, names


def top_universe_from_caps(caps: pd.DataFrame, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the prior-day top-K universe without retaining prices in memory.

    ``cap_weights`` holds the selected prior-day capitalizations only.  It is
    sufficient to compute the market return after the caps panel is released.
    """
    cap_values = caps.to_numpy(copy=False)
    n_dates, n_assets = cap_values.shape
    if n_assets < top_k:
        raise ValueError(f"panel has {n_assets} assets, fewer than top-k={top_k}")
    positions = np.full((n_dates, top_k), -1, dtype=np.int32)
    cap_weights = np.full((n_dates, top_k), np.nan, dtype=np.float32)
    is_stock = caps.columns.astype(str).to_numpy() != "SPY"

    for t in range(1, n_dates):
        prior_cap = cap_values[t - 1]
        eligible = np.flatnonzero(np.isfinite(prior_cap) & (prior_cap > 0) & is_stock)
        if len(eligible) < top_k:
            continue
        candidate = eligible[np.argpartition(prior_cap[eligible], -top_k)[-top_k:]]
        candidate = candidate[np.argsort(-prior_cap[candidate], kind="stable")]
        positions[t] = candidate
        cap_weights[t] = prior_cap[candidate]
    return positions, cap_weights


def market_return_from_prices(
    prices: pd.DataFrame, positions: np.ndarray, cap_weights: np.ndarray
) -> np.ndarray:
    """Compute the cap-weighted top-K market return after caps have been freed."""
    price_values = prices.to_numpy(copy=False)
    market = np.full(len(prices), np.nan, dtype=np.float64)
    for t in range(1, len(prices)):
        candidate = positions[t]
        if np.any(candidate < 0):
            continue
        simple_return = price_values[t, candidate] / price_values[t - 1, candidate] - 1.0
        valid = np.isfinite(simple_return)
        if valid.any():
            weights = cap_weights[t, valid].astype(np.float64, copy=False)
            market[t] = float(np.dot(weights, simple_return[valid]) / weights.sum())
    return market

def load_current_adjacency(snapshot: GraphSnapshot, expected_names: np.ndarray) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    matrix = pd.read_csv(snapshot.adjacency_file, compression="infer", dtype={"permno": str})
    if "permno" not in matrix or matrix.permno.duplicated().any():
        raise ValueError(f"invalid adjacency export: {snapshot.adjacency_file}")
    names = matrix.permno.to_numpy(dtype=str)
    if not np.array_equal(names, expected_names):
        raise ValueError(f"node ordering mismatch between graph exports at {snapshot.refit_date.date()}")
    adjacency = matrix.drop(columns="permno").to_numpy(dtype=np.float64, copy=False)
    if adjacency.shape != (len(names), len(names)) or not np.allclose(adjacency, adjacency.T, atol=1e-10):
        raise ValueError(f"invalid non-symmetric adjacency at {snapshot.refit_date.date()}")
    np.fill_diagonal(adjacency, 0.0)
    community = pd.read_csv(snapshot.community_file, dtype={"permno": str}).set_index("permno")
    communities = community.loc[names, "community_id"].to_numpy(dtype=np.int64)
    return adjacency, {name: i for i, name in enumerate(names)}, communities


def normalize_signal(raw_signal: np.ndarray, winsor: float) -> tuple[np.ndarray, np.ndarray]:
    """Winsorize, z-score and enforce a self-financing gross-one book."""
    valid = np.isfinite(raw_signal)
    if valid.sum() < 2:
        return valid, np.empty(0, dtype=np.float64)
    values = raw_signal[valid].astype(np.float64, copy=True)
    if winsor:
        lo, hi = np.quantile(values, [winsor, 1.0 - winsor])
        values = np.clip(values, lo, hi)
    values -= values.mean()
    scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return valid, np.full(len(values), np.nan)
    weights = values / scale
    weights -= weights.mean()  # keeps numerical net exposure at zero
    gross = np.abs(weights).sum()
    return valid, weights / gross if gross else np.full(len(values), np.nan)


def graph_local_signal(adjacency: np.ndarray, returns: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    signal = np.full(len(returns), np.nan)
    usable = degree > 0
    peer_return = adjacency[usable] @ returns / degree[usable]
    signal[usable] = -(returns[usable] - peer_return)
    return signal


def graph_community_signal(communities: np.ndarray, returns: np.ndarray) -> np.ndarray:
    signal = np.full(len(returns), np.nan)
    _, inverse, count = np.unique(communities, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=returns)
    group_mean = sums[inverse] / count[inverse]
    usable = count[inverse] >= 2
    signal[usable] = -(returns[usable] - group_mean[usable])
    return signal


def performance_metrics(daily: pd.DataFrame) -> dict[str, float | int | str]:
    pnl = daily.daily_excess_return.dropna().astype(float)
    cumulative = pnl.cumsum()
    volatility = float(pnl.std(ddof=1) * np.sqrt(252))
    ann_return = float(pnl.mean() * 252)
    return {
        "days": int(len(pnl)),
        "start": str(pnl.index.min().date()),
        "end": str(pnl.index.max().date()),
        "AnnRet": ann_return,
        "AnnVol": volatility,
        "Sharpe": ann_return / volatility if volatility else np.nan,
        "AvgDlyBets": float(daily.n_holdings.mean()),
        "2WayTurnover": float(daily.two_way_turnover.dropna().mean()),
        "MDD": float((cumulative - cumulative.cummax()).min()),
        "t_stat": float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))) if len(pnl) > 1 and pnl.std(ddof=1) else np.nan,
        "cumulative_simple_return": float(cumulative.iloc[-1]),
    }


def main() -> None:
    args = parse_args()
    if args.signal_window < 1 or args.top_k < 2 or not 0 <= args.winsor < 0.5:
        raise ValueError("signal-window >= 1, top-k >= 2, and 0 <= winsor < 0.5 are required")
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}; choose a new directory")
    for filename in ("prices.pkl", "caps.pkl"):
        if not (args.panel / filename).is_file():
            raise FileNotFoundError(args.panel / filename)
    snapshots, graph_names = load_snapshots(args.graphs)
    graph_name_union = pd.Index(np.concatenate(list(graph_names.values()))).unique().astype(str)
    print(f"loading panel once; graph-node union has {len(graph_name_union)} permnos", flush=True)

    # Keep the two full wide panels out of memory at the same time.  This is
    # essential when a current-PnL refresh runs beside active graph fitting.
    caps = pd.read_pickle(args.panel / "caps.pkl")
    positions, cap_weights = top_universe_from_caps(caps, args.top_k)
    cap_columns = caps.columns
    cap_index = caps.index
    del caps

    prices = pd.read_pickle(args.panel / "prices.pkl")
    if not (prices.index.equals(cap_index) and prices.columns.equals(cap_columns)):
        raise ValueError("prices and caps must have the same calendar and columns")
    del cap_index, cap_columns
    panel_names = prices.columns.astype(str)
    missing_names = graph_name_union.difference(panel_names)
    if len(missing_names):
        raise ValueError(f"graph contains names absent from panel, e.g. {list(missing_names[:5])}")
    market_return = market_return_from_prices(prices, positions, cap_weights)
    del cap_weights
    compact_prices = prices.loc[:, graph_name_union]
    dates = pd.DatetimeIndex(prices.index)
    del prices

    raw_return = compact_prices.pct_change(fill_method=None).to_numpy(dtype=np.float64, copy=False)
    del compact_prices
    excess_return = raw_return - market_return[:, None]
    compact_position = {name: i for i, name in enumerate(graph_name_union)}
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise ValueError("end must not precede start")
    args.out.mkdir(parents=True)

    snapshots_by_date = {snapshot.refit_date: snapshot for snapshot in snapshots}
    refit_dates = np.array(sorted(snapshots_by_date), dtype="datetime64[ns]")
    active_date: pd.Timestamp | None = None
    active_adjacency: np.ndarray | None = None
    active_index: dict[str, int] = {}
    active_communities: np.ndarray | None = None
    prior_weights: dict[str, dict[str, float]] = {method: {} for method in args.methods}
    daily_records: dict[str, list[dict[str, object]]] = {method: [] for method in args.methods}

    with gzip.open(args.out / "daily_weights.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "payoff_date", "refit_date", "method", "permno", "raw_signal", "weight"])
        for t in range(args.signal_window - 1, len(dates) - 1):
            signal_date = pd.Timestamp(dates[t])
            if signal_date < start or signal_date > end:
                continue
            graph_pos = np.searchsorted(refit_dates, signal_date.to_datetime64(), side="right") - 1
            if graph_pos < 0 or np.any(positions[t] < 0) or not np.isfinite(market_return[t + 1]):
                continue
            refit_date = pd.Timestamp(refit_dates[graph_pos])
            if refit_date != active_date:
                active_date = refit_date
                active_adjacency, active_index, active_communities = load_current_adjacency(
                    snapshots_by_date[refit_date], graph_names[refit_date]
                )
                print(f"loaded graph {refit_date.date()} for signal dates from {signal_date.date()}", flush=True)
            assert active_adjacency is not None and active_communities is not None

            top_names = set(panel_names[positions[t]].astype(str))
            state_names = graph_names[refit_date]
            graph_rows = np.fromiter(
                (i for i, name in enumerate(state_names) if name in top_names), dtype=np.int64
            )
            if len(graph_rows) < 20:
                continue
            compact_rows = np.asarray([compact_position[state_names[i]] for i in graph_rows], dtype=np.int64)
            history = excess_return[t - args.signal_window + 1 : t + 1, compact_rows]
            payoff = excess_return[t + 1, compact_rows]
            valid = np.isfinite(history).all(axis=0) & np.isfinite(payoff)
            graph_rows, compact_rows, payoff = graph_rows[valid], compact_rows[valid], payoff[valid]
            if len(graph_rows) < 20:
                continue
            x = history[:, valid].sum(axis=0)
            adjacency = active_adjacency[np.ix_(graph_rows, graph_rows)]
            communities = active_communities[graph_rows]

            raw_signals = {
                "graph_local": graph_local_signal(adjacency, x),
                "graph_community": graph_community_signal(communities, x),
            }
            for method in args.methods:
                kept, weights = normalize_signal(raw_signals[method], args.winsor)
                if not len(weights) or not np.isfinite(weights).all():
                    continue
                names = state_names[graph_rows[kept]]
                method_payoff = payoff[kept]
                daily_return = float(np.dot(weights, method_payoff))
                current_weights = dict(zip(names, weights, strict=True))
                keys = prior_weights[method].keys() | current_weights.keys()
                turnover = float(sum(abs(current_weights.get(k, 0.0) - prior_weights[method].get(k, 0.0)) for k in keys))
                prior_weights[method] = current_weights
                daily_records[method].append(
                    {
                        "date": signal_date,
                        "payoff_date": pd.Timestamp(dates[t + 1]),
                        "refit_date": refit_date,
                        "daily_excess_return": daily_return,
                        "n_holdings": len(weights),
                        "gross_exposure": float(np.abs(weights).sum()),
                        "net_exposure": float(weights.sum()),
                        "two_way_turnover": turnover,
                    }
                )
                for name, signal, weight in zip(names, raw_signals[method][kept], weights, strict=True):
                    writer.writerow([signal_date.date(), dates[t + 1].date(), refit_date.date(), method, name, signal, weight])

    summaries: list[dict[str, object]] = []
    for method, records in daily_records.items():
        if not records:
            raise RuntimeError(f"{method}: no daily observations were produced")
        daily = pd.DataFrame.from_records(records).set_index("date").sort_index()
        daily["cum_simple_return"] = daily.daily_excess_return.cumsum()
        daily.to_csv(args.out / f"{method}_daily.csv", index_label="date")
        metrics = performance_metrics(daily)
        metrics["method"] = method
        summaries.append(metrics)
    pd.DataFrame(summaries).set_index("method").sort_index().to_csv(args.out / "performance_metrics.csv")
    metadata = {
        "methods": list(args.methods),
        "graph_request": "FinGraph k=1 target; realized component counts remain in the graph manifest",
        "community_step": "upstream weighted Fast-Greedy community step",
        "graph_update": "uses latest graph refit on or before each signal date",
        "universe": f"top {args.top_k} stocks by prior-day market cap, intersected with active graph nodes",
        "market_return": "prior-day-cap-weighted return of the daily top-K universe",
        "signal": f"trailing {args.signal_window}-day cumulative daily market-excess return",
        "trading_lag": "signal uses information through t; daily PnL uses t+1 excess returns",
        "normalization": f"{args.winsor:.1%} two-sided winsorization, z-score, de-mean, gross=1",
        "turnover": "sum(abs(weight_t - weight_t_minus_1)) over the union of names",
        "source_panel": str(args.panel.resolve()),
        "source_graphs": str(args.graphs.resolve()),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out / 'performance_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
