"""Daily KMeans mean-reversion portfolio construction.

A book formed from information available at signal-date ``t`` is paid on the
next trading day.  The routine retains the original repository's ``naive``
portfolio rule: use the 500 largest securities by the supplied prior-row
capitalization matrix, subtract the within-cluster five-day residual path,
and equal-weight the positive and negative sides independently.  When both
sides are populated, its gross exposure is two and its net exposure is zero.

The inputs are intentionally arrays and saved daily cluster labels so this
module has no data-path assumptions or vendor-data dependencies.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

BookCallback = Callable[[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]


def naive_stream(
    cache: dict,
    label_key: str,
    dates: pd.DatetimeIndex,
    column_positions: dict[str, int],
    prior_caps: np.ndarray,
    cap_columns: np.ndarray,
    *,
    R_trade: np.ndarray,
    R_affil: np.ndarray,
    R_pay: np.ndarray | None = None,
    refit_dates: Sequence[pd.Timestamp] | None = None,
    clusters: int = 30,
    beta_window: int = 60,
    signal_window: int = 5,
    affiliation_window: int = 21,
    max_label_age_days: int = 10,
    date_start: str | pd.Timestamp = "1996-01-01",
    date_end: str | pd.Timestamp = "2024-12-31",
    book_callback: BookCallback | None = None,
) -> pd.Series:
    """Return daily original-rule KMeans PnL for a saved partition cache.

    ``prior_caps`` is deliberately indexed one row before the signal date. The
    surrounding runner shifts it so a book formed at t uses ``DlyCap[t]`` while
    its payoff is ``R_pay[t+1]`` (or ``R_trade[t+1]`` when omitted).

    A security missing from a saved cluster snapshot is assigned to the cluster
    with the highest mean 21-day residual correlation. ``book_callback`` emits
    realised stock-level books without changing portfolio construction.
    """
    if refit_dates is None:
        refit_dates = sorted(pd.Timestamp(key) for key in cache)
    else:
        refit_dates = sorted(pd.Timestamp(date) for date in refit_dates)

    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
    pnl: dict[pd.Timestamp, float] = {}

    for date_index in range(beta_window + signal_window, len(dates) - 1):
        signal_date = pd.Timestamp(dates[date_index])
        if signal_date < start or signal_date > end:
            continue

        cap_row = prior_caps[date_index - 1]
        finite_caps = np.flatnonzero(np.isfinite(cap_row))
        if len(finite_caps) < 50:
            continue
        top_count = min(500, len(finite_caps))
        top = finite_caps[np.argpartition(cap_row[finite_caps], -top_count)[-top_count:]]
        top = top[np.argsort(-cap_row[top])]
        universe = cap_columns[top]
        positions = np.asarray([column_positions.get(str(ticker), -1) for ticker in universe])
        mapped = positions >= 0
        universe, positions = universe[mapped], positions[mapped]

        lookback = R_trade[date_index - signal_window + 1 : date_index + 1][:, positions]
        affiliation = R_affil[date_index - affiliation_window : date_index][:, positions]
        next_trade = R_trade[date_index + 1][positions]
        next_payoff = next_trade if R_pay is None else R_pay[date_index + 1][positions]
        valid = (
            np.isfinite(lookback).all(axis=0)
            & np.isfinite(affiliation).all(axis=0)
            & np.isfinite(next_trade)
            & np.isfinite(next_payoff)
        )
        if valid.sum() < 50:
            continue

        valid_indices = np.flatnonzero(valid)
        tickers = universe[valid_indices]
        cache_position = bisect_right(refit_dates, signal_date) - 1
        if cache_position < 0 or (signal_date - refit_dates[cache_position]).days > max_label_age_days:
            continue
        snapshot = cache[refit_dates[cache_position]]

        # Correlation-based affiliation for securities absent from the snapshot.
        affiliation_data = affiliation[:, valid_indices]
        centered = affiliation_data - affiliation_data.mean(axis=0, keepdims=True)
        norms = np.sqrt((centered * centered).sum(axis=0))
        norms[norms == 0] = 1.0
        normalized = centered / norms

        stored_labels = {str(ticker): int(label) for ticker, label in zip(snapshot["universe"], snapshot[label_key])}
        labels = np.zeros(len(tickers), dtype=np.int32)
        missing = []
        for index, ticker in enumerate(tickers):
            if str(ticker) in stored_labels:
                labels[index] = stored_labels[str(ticker)]
            else:
                missing.append(index)
        if missing:
            correlation = normalized[:, missing].T @ normalized
            all_indices = np.arange(len(tickers))
            for row, missing_index in enumerate(missing):
                best_label, best_affinity = 0, -np.inf
                for cluster in range(clusters):
                    members = np.flatnonzero((labels == cluster) & (all_indices != missing_index))
                    if len(members):
                        affinity = float(correlation[row, members].mean())
                        if affinity > best_affinity:
                            best_affinity, best_label = affinity, cluster
                labels[missing_index] = best_label

        residual_path = lookback[:, valid_indices]
        cluster_mean = np.zeros_like(residual_path)
        for cluster in range(clusters):
            members = labels == cluster
            if members.any():
                cluster_mean[:, members] = residual_path[:, members].mean(axis=1, keepdims=True)
        raw_signal = -(residual_path - cluster_mean).sum(axis=0)

        positive, negative = raw_signal > 0, raw_signal < 0
        weights = np.zeros_like(raw_signal)
        weights[positive] = 1.0 / max(int(positive.sum()), 1)
        weights[negative] = -1.0 / max(int(negative.sum()), 1)
        realised = next_payoff[valid_indices]
        pnl[signal_date] = float((weights * realised).sum())

        if book_callback is not None:
            book_callback(
                signal_date,
                pd.Timestamp(dates[date_index + 1]),
                np.asarray(tickers, dtype=str),
                raw_signal.copy(),
                weights.copy(),
                realised.copy(),
            )

    return pd.Series(pnl, dtype=float).sort_index()
