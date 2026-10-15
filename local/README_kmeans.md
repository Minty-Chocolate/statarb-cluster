# KMeans workflow

The final notebook reads the saved KMeans book in
`data/derived/t_aligned_kmeans/market/`.  It includes the continuous raw
signal, the original sign/equal-weight portfolio weights, and the aligned
next-day raw return.

To rebuild from the CRSP-compatible panel, first build the compact Top-500
cache inside the Docker volume, then run KMeans:

```sh
python local/build_t_aligned_sic_cache.py \
  --panel data/derived/ciz_compat_panel \
  --raw-ciz /data/raw/crsp/wejd75a6vl0ayj0c.csv \
  --out /cache/t_aligned_top500_1997_2024 \
  --start 1997-05-23 --end 2024-12-30 --top-k 500

python local/run_lightweight_cpu.py \
  --panel data/derived/ciz_compat_panel \
  --sic-cache /cache/t_aligned_top500_1997_2024 \
  --out data/derived/t_aligned_kmeans/market
```

Then export the daily books without refitting KMeans:

```sh
python local/export_kmeans_daily_books.py \
  --panel data/derived/ciz_compat_panel \
  --sic-cache /cache/t_aligned_top500_1997_2024 \
  --run data/derived/t_aligned_kmeans/market
```

Build the matching Top-500 market leg for the decile diagnostic:

```sh
python local/build_kmeans_next_day_market_return.py \
  --panel data/derived/ciz_compat_panel \
  --sic-cache /cache/t_aligned_top500_1997_2024 \
  --books data/derived/t_aligned_kmeans/market/kmeans_market_daily_books.csv.gz \
  --out data/derived/t_aligned_kmeans/market/kmeans_market_next_day_market_return.csv
```

At signal date `t`, this market leg uses Top-500 membership and `DlyCap[t]`
weights, then earns raw return at `t+1`. Notebook 01 subtracts it only from
the individual D1-D10 decile payoffs. The original and continuous z-score
KMeans PnLs retain raw stock returns. The original strategy uses gross 2; the
continuous z-score and D1-D10 books are each gross 1 and net 0.
