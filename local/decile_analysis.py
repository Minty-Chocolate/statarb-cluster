"""Reusable cross-sectional decile analysis for daily strategy books.

The primary input is a *long* table with one row per signal date and security.
It must contain a realised return already aligned to that signal date: a row
labelled t holds a book formed with information through t and its t+1 payoff.
This design keeps the trading lag explicit and avoids accidental re-shifting in
notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


ScoreFunction = Callable[[pd.DataFrame], pd.Series | np.ndarray]


@dataclass(frozen=True)
class DecileResult:
    """Daily equal-weight decile returns and their cumulative log returns."""

    daily_returns: pd.DataFrame
    cumulative_log_returns: pd.DataFrame
    holdings: pd.DataFrame


def cumulative_log_returns(daily_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert simple daily returns to cumulative log returns, starting at zero."""
    below_minus_one = (daily_returns <= -1).to_numpy().any() if isinstance(daily_returns, pd.DataFrame) else (daily_returns <= -1).to_numpy().any()
    if below_minus_one:
        raise ValueError("A return <= -100% has no finite log return")
    return np.log1p(daily_returns).cumsum()


def standardized_long_short_weights(
    books: pd.DataFrame,
    *,
    date_col: str = "signal_date",
    asset_col: str = "permno",
    score_col: str = "raw_signal",
    return_col: str = "payoff_return",
    winsor: float = 0.01,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Make continuous daily net-zero, gross-one weights from a raw score.

    On each trade-date cross-section, finite scores are two-sided winsorized,
    z-scored, de-meaned once more for numerical exactness, and L1-normalized.
    The result has ``weight.sum() == 0`` and ``abs(weight).sum() == 1`` each
    day.  A finite realised return is required because the book is evaluated on
    that return; this is the same tradable set used by ``decile_portfolios``.
    """
    if not 0 <= winsor < 0.5:
        raise ValueError("winsor must be in [0, 0.5)")
    required = {date_col, asset_col, score_col, return_col}
    missing = required.difference(books.columns)
    if missing:
        raise KeyError(f"books is missing: {sorted(missing)}")
    data = books.loc[:, [date_col, asset_col, score_col, return_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data[asset_col] = data[asset_col].astype(str)
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    data = data.loc[np.isfinite(data[score_col]) & np.isfinite(data[return_col])]

    weighted: list[pd.DataFrame] = []
    for _, day in data.groupby(date_col, sort=False, observed=True):
        values = day[score_col].to_numpy(dtype=float, copy=True)
        if len(values) < 2:
            continue
        if winsor:
            lo, hi = np.quantile(values, [winsor, 1.0 - winsor])
            values = np.clip(values, lo, hi)
        values -= values.mean()
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            continue
        values /= std
        values -= values.mean()
        gross = np.abs(values).sum()
        if not np.isfinite(gross) or gross == 0:
            continue
        result = day.copy()
        result[weight_col] = values / gross
        weighted.append(result)
    if not weighted:
        raise ValueError("No daily cross-section supports standardized weighting")
    return pd.concat(weighted, ignore_index=True)


def weighted_daily_returns(
    weighted_books: pd.DataFrame,
    *,
    date_col: str = "signal_date",
    weight_col: str = "weight",
    return_col: str = "payoff_return",
) -> pd.Series:
    """Aggregate a long table of realised security returns into daily book PnL."""
    required = {date_col, weight_col, return_col}
    missing = required.difference(weighted_books.columns)
    if missing:
        raise KeyError(f"weighted_books is missing: {sorted(missing)}")
    pnl = (weighted_books[weight_col] * weighted_books[return_col]).groupby(
        pd.to_datetime(weighted_books[date_col]), sort=True
    ).sum()
    pnl.index.name = date_col
    return pnl.rename("daily_pnl")


def decile_portfolios(
    books: pd.DataFrame,
    *,
    date_col: str = "signal_date",
    asset_col: str = "permno",
    return_col: str = "payoff_return",
    score_col: str = "weight",
    score_fn: ScoreFunction | None = None,
    n_deciles: int = 10,
    higher_is_decile_1: bool = True,
) -> DecileResult:
    """Form equal-weight daily deciles, plus a gross-one/net-zero D1-D10 book.

    ``books`` has already-aligned realised returns: the t row contains the payoff
    earned by a t-close signal.  By default the executed portfolio weight is the
    sorting score.  Pass ``score_fn(day_frame)`` to rank another daily score; it
    must return one value per input row.

    Decile 1 is the highest score when ``higher_is_decile_1=True``.  Thus the
    default long D1 / short D10 series is exactly the requested direction.  Each
    constituent decile is equal dollar weighted; D1-D10 applies +0.5 / -0.5, so
    its gross exposure is one and its net exposure is zero.
    """
    if n_deciles < 2:
        raise ValueError("n_deciles must be at least two")
    required = {date_col, asset_col, return_col, score_col}
    missing = required.difference(books.columns)
    if missing:
        raise KeyError(f"books is missing: {sorted(missing)}")

    prepared = books.loc[:, [date_col, asset_col, return_col, score_col]].copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col])
    prepared[asset_col] = prepared[asset_col].astype(str)
    prepared[return_col] = pd.to_numeric(prepared[return_col], errors="coerce")
    prepared[score_col] = pd.to_numeric(prepared[score_col], errors="coerce")

    return_rows: list[dict[str, float | pd.Timestamp]] = []
    holding_rows: list[dict[str, int | pd.Timestamp]] = []
    ascending = not higher_is_decile_1
    for date, day in prepared.groupby(date_col, sort=True, observed=True):
        if score_fn is None:
            score = day[score_col]
        else:
            score = score_fn(day.copy())
            score = pd.Series(score, index=day.index, dtype=float)
        frame = day.assign(_score=score)
        frame = frame.loc[np.isfinite(frame._score) & np.isfinite(frame[return_col])]
        if len(frame) < n_deciles:
            continue
        # Stable asset-ID tie breaking makes tied scores deterministic, including
        # the zero-weight middle of a discrete strategy book.
        frame = frame.sort_values(["_score", asset_col], ascending=[ascending, True], kind="stable")
        groups = np.array_split(np.arange(len(frame)), n_deciles)
        record: dict[str, float | pd.Timestamp] = {date_col: pd.Timestamp(date)}
        counts: dict[str, int | pd.Timestamp] = {date_col: pd.Timestamp(date)}
        for number, group in enumerate(groups, start=1):
            label = f"decile_{number}"
            record[label] = float(frame.iloc[group][return_col].mean())
            counts[label] = int(len(group))
        record["long_decile_1_short_decile_10"] = 0.5 * (
            float(record["decile_1"]) - float(record[f"decile_{n_deciles}"])
        )
        return_rows.append(record)
        holding_rows.append(counts)

    if not return_rows:
        raise ValueError("No date has enough finite score/return pairs for deciles")
    daily = pd.DataFrame(return_rows).set_index(date_col).sort_index()
    holdings = pd.DataFrame(holding_rows).set_index(date_col).sort_index()
    return DecileResult(
        daily_returns=daily,
        cumulative_log_returns=cumulative_log_returns(daily),
        holdings=holdings,
    )


def plot_decile_curves(result: DecileResult, *, ax, title: str, long_short_label: str = "Long D1 / Short D10"):
    """Plot ten cumulative-log deciles and the gross-one long/short portfolio."""
    curves = result.cumulative_log_returns
    decile_columns = [c for c in curves if c.startswith("decile_")]
    for column in decile_columns:
        ax.plot(curves.index, curves[column], lw=0.8, alpha=0.8, label=column.replace("_", " ").title())
    ax.plot(
        curves.index,
        curves["long_decile_1_short_decile_10"],
        color="black",
        lw=1.8,
        label=long_short_label,
    )
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set(title=title, xlabel="", ylabel="Cumulative log return")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    return ax
