# Graph-local workflow

The saved FinGraph graph exports live in `data/derived/fingraph_current_pnl/graphs`.
They can be converted directly into the graph-local and graph-community daily
signal books without fitting a new graph:

```sh
python local/backtest_fingraph_signals.py \
  --panel data/derived/ciz_compat_panel \
  --graphs data/derived/fingraph_current_pnl/graphs \
  --out data/derived/graph_local_refresh \
  --start 2010-01-04 \
  --end 2024-10-10 \
  --signal-window 5 \
  --top-k 500 \
  --winsor 0.01 \
  --methods graph_local
```

Signals use information through the close of `t`; PnL uses the top-500
prior-cap-weighted market-excess return at `t+1`.  The backtest winsorizes the
daily graph-local signal at 1%, z-scores and de-means it, then normalizes to
gross exposure one and net exposure zero.

Open `notebooks/02_graph_local.ipynb` to display the saved weights, exposure
checks, performance metrics, and cumulative-log PnL.

## Re-fitting a graph

`local/build_fingraph_study_windows.py`, `local/fit_fingraph_study.R`, and
`local/export_fingraph_artifacts.R` are retained for a new FinGraph fit.  They
require the mounted upstream `fingraph-main` source and the CRSP-compatible
panel.  The final signal backtest remains the command above.

## Decile payoff book

`notebooks/02_graph_local.ipynb` additionally displays daily raw-signal deciles
and the gross-one/net-zero D1-D10 endpoint portfolio.  Its input book pairs
each saved graph-local weight with its exact realised security-level t+1
market-excess return. Rebuild that book when refreshing the saved backtest:

```sh
python local/export_graph_local_daily_books.py \
  --panel data/derived/ciz_compat_panel \
  --backtest data/derived/fingraph_current_pnl/backtest
```

The exporter checks that the reconstructed security-level returns reproduce the
saved graph-local daily PnL before writing its output.
