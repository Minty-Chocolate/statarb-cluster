# Data policy

This repository contains source code and executed notebook output only. It does
not distribute CRSP, Compustat, CCM, TAQ, or any panel or result derived from
those vendor datasets.

The notebooks render their saved figures and tables on GitHub. To execute them
locally, supply a licensed CRSP-compatible daily panel at
`data/derived/ciz_compat_panel/`, build the required compact cache in the
Docker volume, and then follow the KMeans or Graph-local instructions in
`local/README_kmeans.md` and `local/README_graph_local.md`.

All files below `data/raw/` and `data/derived/` are ignored by Git. This
includes saved books, PnL, caches, model partitions, and any other results
computed from restricted data.
