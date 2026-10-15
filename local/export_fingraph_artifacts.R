#!/usr/bin/env Rscript
# Export the compact, language-neutral inputs consumed by the Python FinGraph
# signal backtest.  The fitted RDS states remain the canonical graph cache;
# this script writes one compressed adjacency matrix and Fast-Greedy partition
# per refit date so the trading layer neither calls R nor re-fits a graph.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: export_fingraph_artifacts.R GRAPH_DIR OUTPUT_DIR")
graph_dir <- normalizePath(args[[1]], mustWork = TRUE)
out_dir <- args[[2]]
states_dir <- file.path(graph_dir, "per_window")
if (!dir.exists(states_dir)) stop("missing per_window directory: ", states_dir)
if (dir.exists(out_dir)) stop("output exists: ", out_dir)
dir.create(out_dir, recursive = TRUE)
dir.create(file.path(out_dir, "adjacency"))
dir.create(file.path(out_dir, "communities"))

suppressPackageStartupMessages(library(igraph))
state_paths <- sort(list.files(states_dir, pattern = "_state\\.rds$", full.names = TRUE))
if (!length(state_paths)) stop("no fitted graph states found in ", states_dir)

records <- vector("list", length(state_paths))
for (i in seq_along(state_paths)) {
  state_path <- state_paths[[i]]
  state <- readRDS(state_path)
  date <- sub("_state\\.rds$", "", basename(state_path))
  if (is.null(state$adjacency) || is.null(state$permno)) {
    stop("state must contain adjacency and permno: ", state_path)
  }
  adjacency <- as.matrix(state$adjacency)
  permno <- as.character(state$permno)
  if (nrow(adjacency) != length(permno) || ncol(adjacency) != length(permno)) {
    stop("invalid adjacency dimensions in ", state_path)
  }
  diag(adjacency) <- 0
  if (!all(is.finite(adjacency)) || any(adjacency < 0)) stop("invalid adjacency in ", state_path)

  adjacency_path <- file.path(out_dir, "adjacency", paste0(date, ".csv.gz"))
  connection <- gzfile(adjacency_path, open = "wt")
  write.csv(
    data.frame(permno = permno, adjacency, check.names = FALSE),
    connection,
    row.names = FALSE,
    quote = FALSE
  )
  close(connection)

  graph <- graph_from_adjacency_matrix(adjacency, mode = "undirected", weighted = TRUE, diag = FALSE)
  communities <- cluster_fast_greedy(as_undirected(graph))
  community_path <- file.path(out_dir, "communities", paste0(date, ".csv"))
  write.csv(
    data.frame(
      permno = permno,
      component_id = components(graph)$membership - 1L,
      community_id = membership(communities) - 1L
    ),
    community_path,
    row.names = FALSE
  )
  records[[i]] <- data.frame(
    refit_date = date,
    adjacency_file = file.path("adjacency", basename(adjacency_path)),
    community_file = file.path("communities", basename(community_path)),
    n_assets = length(permno),
    n_components = length(sizes(components(graph))),
    n_communities = length(sizes(communities)),
    fast_greedy_modularity = modularity(communities),
    stringsAsFactors = FALSE
  )
  message(sprintf("exported %d/%d: %s", i, length(state_paths), date))
}

write.csv(do.call(rbind, records), file.path(out_dir, "manifest.csv"), row.names = FALSE)
writeLines(c(
  "{",
  '  "source": "FinGraph RDS states",',
  '  "community_method": "igraph::cluster_fast_greedy on the full weighted undirected adjacency",',
  '  "purpose": "read-only export for Python graph-local and graph-community signal backtests"',
  "}"
), file.path(out_dir, "metadata.json"))
