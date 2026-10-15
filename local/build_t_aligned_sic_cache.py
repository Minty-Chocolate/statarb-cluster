#!/usr/bin/env python3
"""Create the compact SIC2 cache used by the t-aligned Top-500 benchmark.

The cache is deliberately small and belongs on fast local scratch (``/cache`` in
Docker).  It stores only securities that ever enter the daily Top-K universe,
not a date x all-CRSP SIC matrix.  Dates use the t-close convention:

* Top-K membership and weights use CRSP DlyCap[t].  In the legacy-compatible
  panel, DlyCap[t] is caps.pkl[t+1] because caps.pkl[t] is DlyPrevCap[t].
* SICCD is read from the raw CIZ record dated t.

The raw CIZ CSV must still be read once to obtain time-varying SICCD, but every
write is compact and local; no full-market industry-return panel is materialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

SPY = "SPY"
SOURCE_COLUMNS = ["PERMNO", "DlyCalDt", "SICCD"]


def digest(values: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in values.astype(str):
        h.update(value.encode())
        h.update(b"\0")
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--ciz", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--history-rows", type=int, default=126)
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--block-size-mib", type=int, default=64)
    return p.parse_args()


def choose_rows(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, history: int) -> tuple[int, int]:
    first = int(index.searchsorted(start, side="left"))
    last = int(index.searchsorted(end, side="right")) - 1
    if first >= len(index) or last < first or last + 1 >= len(index):
        raise ValueError("requested dates require a following panel row to obtain DlyCap[t]")
    return max(0, first - history), last


def build_membership(caps: pd.DataFrame, first: int, last: int, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return (union original cols, date x union membership, date x union DlyCap, dates)."""
    columns = caps.columns.astype(str).to_numpy()
    allowed = np.flatnonzero(columns != SPY)
    dates = pd.DatetimeIndex(caps.index[first : last + 1])
    selected = np.full((len(dates), top_k), -1, dtype=np.int32)
    started = time.monotonic()
    for local_row, source_row in enumerate(range(first, last + 1)):
        # caps[t+1] is DlyCap[t]; the last requested t was checked above.
        row = caps.iloc[source_row + 1].to_numpy(dtype=np.float64, copy=False)
        finite = allowed[np.isfinite(row[allowed]) & (row[allowed] > 0)]
        if len(finite):
            take = min(top_k, len(finite))
            best = finite[np.argpartition(row[finite], -take)[-take:]]
            best = best[np.argsort(-row[best], kind="stable")]
            selected[local_row, :take] = best
        if (local_row + 1) % 1000 == 0:
            print(f"top-k membership {local_row + 1:,}/{len(dates):,}; {time.monotonic()-started:.0f}s", flush=True)
    union_original = np.unique(selected[selected >= 0])
    original_to_union = np.full(caps.shape[1], -1, dtype=np.int32)
    original_to_union[union_original] = np.arange(len(union_original), dtype=np.int32)
    member = np.zeros((len(dates), len(union_original)), dtype=bool)
    for row in range(len(dates)):
        take = selected[row, selected[row] >= 0]
        member[row, original_to_union[take]] = True
    cap_t = caps.iloc[first + 1 : last + 2, union_original].to_numpy(dtype=np.float64, copy=True)
    return union_original, member, cap_t, dates


def parse_permnos(names: np.ndarray) -> dict[int, int]:
    result: dict[int, int] = {}
    for position, name in enumerate(names.astype(str)):
        try:
            result[int(name.rsplit(".", 1)[1])] = position
        except (IndexError, ValueError):
            continue
    return result


def extract_sic(ciz: Path, dates: pd.DatetimeIndex, names: np.ndarray, out: Path, block_size_mib: int) -> np.ndarray:
    date_to_row = {np.datetime64(day, "ns"): i for i, day in enumerate(dates.to_numpy())}
    permno_to_col = parse_permnos(names)
    codes = np.lib.format.open_memmap(out / "sic2.npy", mode="w+", dtype=np.int16, shape=(len(dates), len(names)))
    codes[:] = 0
    reader = pacsv.open_csv(
        ciz,
        read_options=pacsv.ReadOptions(block_size=block_size_mib * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(include_columns=SOURCE_COLUMNS),
    )
    started = time.monotonic()
    mapped = 0
    for batch_num, batch in enumerate(reader, start=1):
        frame = batch.to_pandas()
        source_date = pd.to_datetime(frame["DlyCalDt"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        rows = np.fromiter((date_to_row.get(value, -1) for value in source_date), dtype=np.int32, count=len(frame))
        permno = pd.to_numeric(frame["PERMNO"], errors="coerce").to_numpy(dtype=float)
        cols = np.full(len(frame), -1, dtype=np.int32)
        valid_permno = np.isfinite(permno)
        cols[valid_permno] = np.fromiter(
            (permno_to_col.get(int(value), -1) for value in permno[valid_permno]),
            dtype=np.int32,
            count=int(valid_permno.sum()),
        )
        sic = pd.to_numeric(frame["SICCD"], errors="coerce").to_numpy(dtype=float)
        sic2 = np.zeros(len(frame), dtype=np.int16)
        valid_sic = np.isfinite(sic) & (sic >= 100) & (sic <= 9999)
        sic2[valid_sic] = (sic[valid_sic].astype(np.int64) // 100).astype(np.int16)
        keep = (rows >= 0) & (cols >= 0) & (sic2 > 0)
        if keep.any():
            rr, cc, ss = rows[keep], cols[keep], sic2[keep]
            linear = rr.astype(np.int64) * len(names) + cc
            _, first = np.unique(linear, return_index=True)
            rr, cc, ss = rr[first], cc[first], ss[first]
            fresh = codes[rr, cc] == 0
            codes[rr[fresh], cc[fresh]] = ss[fresh]
            mapped += int(fresh.sum())
        if batch_num % 50 == 0:
            print(f"SIC scan batches={batch_num:,}; mapped={mapped:,}; elapsed={time.monotonic()-started:.0f}s", flush=True)
    codes.flush()
    return codes


def main() -> None:
    a = parse_args()
    if a.out.exists():
        raise FileExistsError(f"cache already exists: {a.out}")
    if a.top_k < 1 or a.history_rows < 0:
        raise ValueError("top-k must be positive and history-rows must be nonnegative")
    a.out.mkdir(parents=True)
    caps = pd.read_pickle(a.panel / "caps.pkl")
    index = pd.DatetimeIndex(caps.index)
    first, last = choose_rows(index, pd.Timestamp(a.start), pd.Timestamp(a.end), a.history_rows)
    union_original, member, cap_t, dates = build_membership(caps, first, last, a.top_k)
    names = caps.columns.astype(str).to_numpy()[union_original]
    del caps
    np.save(a.out / "dates.npy", dates.to_numpy(dtype="datetime64[ns]"))
    np.save(a.out / "columns.npy", names)
    np.save(a.out / "member.npy", member)
    np.save(a.out / "cap_t.npy", cap_t)
    extract_sic(a.ciz, dates, names, a.out, a.block_size_mib)
    metadata = {
        "complete": True,
        "panel": str(a.panel.resolve()),
        "raw_ciz": str(a.ciz.resolve()),
        "start": a.start,
        "end": a.end,
        "history_rows": a.history_rows,
        "top_k": a.top_k,
        "n_dates": len(dates),
        "n_union": len(names),
        "columns_sha256": digest(names),
        "universe": "Top-K among panel common shares, selected by CRSP DlyCap[t]",
        "sic": "CRSP SICCD[t], two-digit floor(SICCD / 100)",
        "timing": "all cache rows use t-close information; cap_t is caps.pkl[t+1] = DlyCap[t]",
    }
    (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote compact cache {a.out}: {len(dates):,} dates x {len(names):,} union securities", flush=True)


if __name__ == "__main__":
    main()
