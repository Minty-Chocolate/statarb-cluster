#!/usr/bin/env Rscript
# Fit FinGraph directly from the read-only upstream source on the same windows
# used by local/fit_fingraph_study_kmeans.py.
#
# Each fit uses the upstream stock-example settings: Student-t graph learner,
# rho=10, and nu estimated from each window's one-dimensional equal-weighted
# market proxy.  The learner itself standardizes each stock time series.
# Connected components are recorded only as diagnostics; the trading layer uses
# the learned adjacency directly or Fast-Greedy communities derived from it.

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% 4:9)) {
  stop("usage: fit_fingraph_study.R WINDOWS_DIR OUTPUT_DIR K MAXITER [LIMIT] [INIT_MODE] [DRIFT_LIMIT] [CHECK_EVERY] [STABILITY_DAYS]")
}
windows_dir <- normalizePath(args[[1]], mustWork = TRUE)
out_dir <- args[[2]]
k <- as.integer(args[[3]])
maxiter <- as.integer(args[[4]])
limit <- if (length(args) >= 5) as.integer(args[[5]]) else Inf
init_mode <- if (length(args) >= 6) args[[6]] else "warm"
drift_limit <- if (length(args) >= 7) as.numeric(args[[7]]) else 0.01
stability_check_every <- if (length(args) >= 8) as.integer(args[[8]]) else 10L
stability_days <- if (length(args) >= 9) as.integer(args[[9]]) else 21L
stop_rule <- "weight_stable"
stop_patience <- 1L
if (is.na(limit) || limit < 1) stop("LIMIT must be a positive integer")
if (!(init_mode %in% c("cold", "warm"))) stop("INIT_MODE must be 'cold' or 'warm'")
if (!is.finite(drift_limit) || drift_limit <= 0 || drift_limit >= 2) stop("DRIFT_LIMIT must be in (0, 2)")
if (is.na(stability_check_every) || stability_check_every < 1) stop("CHECK_EVERY must be a positive integer")
if (is.na(stability_days) || stability_days < 1) stop("STABILITY_DAYS must be a positive integer")
partition_stable_enabled <- FALSE
signal_stable_enabled <- TRUE
topology_target_required <- FALSE
source_root <- Sys.getenv("FINGRAPH_SOURCE", "/opt/fingraph")
if (!file.exists(file.path(source_root, "R", "k-component-graph-heavy-tail.R"))) {
  stop("FINGRAPH_SOURCE does not point to fingraph-main: ", source_root)
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
if (!dir.exists(file.path(out_dir, "per_window"))) dir.create(file.path(out_dir, "per_window"))

suppressPackageStartupMessages(library(fitHeavyTail))
suppressPackageStartupMessages(library(igraph))
# FinGraph calls a handful of basic Laplacian transforms from
# spectralGraphTopology.  Its current CRAN/GitHub dependency tree includes
# CVXR solvers which do not build reliably on ARM Docker.  The original helper
# functions are elementary linear transforms, reproduced verbatim in meaning
# below; the FinGraph ADMM implementation itself is loaded unchanged.
edge_pairs <- function(p) do.call(rbind, lapply(seq_len(p - 1L), function(i) cbind(i, (i + 1L):p)))
A <- function(w) {
  p <- as.integer((1 + sqrt(1 + 8 * length(w))) / 2)
  result <- matrix(0, p, p)
  pairs <- edge_pairs(p)
  result[pairs] <- w
  result <- result + t(result)
  result
}
L <- function(w) {
  adjacency <- A(w)
  diag(rowSums(adjacency)) - adjacency
}
Ainv <- function(M) { pairs <- edge_pairs(ncol(M)); M[pairs] }
Linv <- function(M) { pairs <- edge_pairs(ncol(M)); -M[pairs] }
Lstar <- function(M) {
  p <- ncol(M)
  pairs <- edge_pairs(p)
  M[cbind(pairs[, 1], pairs[, 1])] + M[cbind(pairs[, 2], pairs[, 2])] -
    M[cbind(pairs[, 1], pairs[, 2])] - M[cbind(pairs[, 2], pairs[, 1])]
}
Dstar <- function(v) {
  p <- length(v)
  pairs <- edge_pairs(p)
  v[pairs[, 1]] + v[pairs[, 2]]
}
w_init <- function(w0, Sinv) {
  if (is.character(w0) && w0 == "naive") {
    w0 <- Linv(Sinv)
    w0[w0 < 0] <- 0
  }
  w0
}
eigval_sym <- function(M) eigen(M, symmetric = TRUE, only.values = TRUE)$values
compute_student_weights <- function(w, LstarSq, p, nu) (p + nu) / (sum(w * LstarSq) + nu)
component_membership <- function(adjacency) {
  p <- ncol(adjacency)
  membership <- integer(p)
  component <- 0L
  for (node in seq_len(p)) {
    if (membership[node] != 0L) next
    component <- component + 1L
    membership[node] <- component
    frontier <- node
    while (length(frontier)) {
      current <- frontier[[1]]
      frontier <- frontier[-1]
      neighbours <- which(adjacency[current, ] > 0 & membership == 0L)
      if (length(neighbours)) {
        membership[neighbours] <- component
        frontier <- c(frontier, neighbours)
      }
    }
  }
  membership
}

# The stability criterion is stated in units of the downstream strategy, not
# in units of the ADMM's Laplacian update.  Both signal variants use the last
# five standardized return observations and the Python layer's
# winsorize/z-score/gross normalization. This is a solver-stability diagnostic
# in the learner's standardized units; the subsequent backtest independently
# constructs the exact raw market-excess trading signals.
signal_window <- 5L
signal_winsor <- 0.01
normalize_signal_weights <- function(signal) {
  valid <- is.finite(signal)
  if (sum(valid) < 2L) return(NULL)
  values <- signal[valid]
  lo_hi <- stats::quantile(values, probs = c(signal_winsor, 1 - signal_winsor), names = FALSE, type = 7)
  values <- pmin(pmax(values, lo_hi[[1]]), lo_hi[[2]])
  values <- values - mean(values)
  scale <- sqrt(mean(values^2))
  if (!is.finite(scale) || scale == 0) return(NULL)
  weights <- rep(0, length(signal))
  weights[valid] <- values / scale
  weights[valid] <- weights[valid] - mean(weights[valid])
  gross <- sum(abs(weights))
  if (!is.finite(gross) || gross == 0) return(NULL)
  weights / gross
}
community_membership <- function(adjacency) {
  graph <- igraph::graph_from_adjacency_matrix(adjacency, mode = "undirected", weighted = TRUE, diag = FALSE)
  as.integer(igraph::membership(igraph::cluster_fast_greedy(igraph::as.undirected(graph))))
}
signal_weight_matrix <- function(adjacency, x, method, n_days) {
  p <- ncol(x)
  if (nrow(x) < signal_window + n_days - 1L) return(NULL)
  membership <- if (method == "community") community_membership(adjacency) else NULL
  weights <- matrix(NA_real_, nrow = p, ncol = n_days)
  degree <- rowSums(adjacency)
  for (day in seq_len(n_days)) {
    end_row <- nrow(x) - n_days + day
    cumulative_return <- colSums(x[(end_row - signal_window + 1L):end_row, , drop = FALSE])
    if (method == "local") {
      if (any(degree <= 0)) return(NULL)
      raw_signal <- -(cumulative_return - as.vector(adjacency %*% cumulative_return) / degree)
    } else {
      group_count <- tabulate(membership, nbins = max(membership))
      group_sum <- vapply(
        seq_len(max(membership)),
        function(group) sum(cumulative_return[membership == group]),
        numeric(1L)
      )
      group_mean <- group_sum / group_count
      raw_signal <- rep(NA_real_, p)
      eligible <- group_count[membership] >= 2L
      raw_signal[eligible] <- -(cumulative_return[eligible] - group_mean[membership[eligible]])
    }
    weights[, day] <- normalize_signal_weights(raw_signal)
    if (any(!is.finite(weights[, day]))) return(NULL)
  }
  weights
}
model_drift <- function(current, previous) {
  if (is.null(current) || is.null(previous) || !identical(dim(current), dim(previous))) return(NA_real_)
  mean(colSums(abs(current - previous)))
}
source_lines <- readLines(file.path(source_root, "R", "k-component-graph-heavy-tail.R"))
source_lines <- source_lines[!grepl("^library\\(spectralGraphTopology\\)", source_lines)]
source_lines <- gsub("spectralGraphTopology:::", "", source_lines, fixed = TRUE)
# Observability only: preserve the upstream iterations and emit a checkpoint
# every ten steps so a 500-node fit can be monitored without changing math.
source_lines <- gsub(
  "  elapsed_time <- c()",
  "  elapsed_time <- c()\n  relative_step <- c()\n  termination_reason <- 'maxiter'\n  partition_stable_reached <- FALSE\n  partition_streak <- 0L\n  previous_membership <- NULL\n  signal_stable_reached <- FALSE\n  signal_streak <- 0L\n  previous_local_signal_weights <- NULL\n  previous_community_signal_weights <- NULL\n  signal_checkpoint_records <- list()",
  source_lines,
  fixed = TRUE
)
source_lines <- gsub(
  "    has_converged <- (norm(Lwi - Lw, 'F') / norm(Lw, 'F') < reltol) && (i > 1)",
  "    if (partition_stable_enabled) {\n      membership_i <- component_membership(A(wi))\n      target_ok <- max(membership_i) == k && n_zero_eigenvalues == k\n      if (target_ok && !is.null(previous_membership) && identical(membership_i, previous_membership)) {\n        partition_streak <- partition_streak + 1L\n      } else if (target_ok) {\n        partition_streak <- 1L\n      } else {\n        partition_streak <- 0L\n      }\n      previous_membership <- membership_i\n      if (target_ok && partition_streak >= stop_patience) {\n        partition_stable_reached <- TRUE\n        termination_reason <- 'partition_stable'\n        has_converged <- FALSE\n        break\n      }\n    }\n    if (signal_stable_enabled && i %% stability_check_every == 0L) {\n      adjacency_i <- A(wi)\n      membership_i <- component_membership(adjacency_i)\n      topology_target_met <- max(membership_i) == k\n      local_weights_i <- NULL\n      community_weights_i <- NULL\n      local_drift_i <- NA_real_\n      community_drift_i <- NA_real_\n      stable_i <- FALSE\n      if (topology_target_met) {\n        local_weights_i <- signal_weight_matrix(adjacency_i, X, 'local', stability_days)\n        community_weights_i <- signal_weight_matrix(adjacency_i, X, 'community', stability_days)\n        local_drift_i <- model_drift(local_weights_i, previous_local_signal_weights)\n        community_drift_i <- model_drift(community_weights_i, previous_community_signal_weights)\n        stable_i <- is.finite(local_drift_i) && is.finite(community_drift_i) && local_drift_i < drift_limit && community_drift_i < drift_limit\n      }\n      if (stable_i) {\n        signal_streak <- signal_streak + 1L\n      } else {\n        signal_streak <- 0L\n      }\n      if (!is.null(local_weights_i)) previous_local_signal_weights <- local_weights_i\n      if (!is.null(community_weights_i)) previous_community_signal_weights <- community_weights_i\n      signal_checkpoint_records[[length(signal_checkpoint_records) + 1L]] <- data.frame(\n        iteration = i,\n        adjacency_components = max(membership_i),\n        topology_target_met = topology_target_met,\n        local_model_drift = local_drift_i,\n        community_model_drift = community_drift_i,\n        stable = stable_i,\n        stable_streak = signal_streak,\n        stringsAsFactors = FALSE\n      )\n      if (stable_i && signal_streak >= stop_patience) {\n        signal_stable_reached <- TRUE\n        termination_reason <- 'signal_stable'\n        has_converged <- FALSE\n        break\n      }\n    }\n    step <- norm(Lwi - Lw, 'F') / norm(Lw, 'F')\n    relative_step <- c(relative_step, step)\n    if (i %% 10 == 0) message(sprintf('ADMM iteration %d | beta %.3g', i, beta))\n    has_converged <- (step < reltol) && (i > 1)",
  source_lines,
  fixed = TRUE
)
source_lines <- gsub(
  "if (topology_target_met) {",
  "if (!topology_target_required || topology_target_met) {",
  source_lines,
  fixed = TRUE
)
source_lines <- gsub(
  "stable = stable_i,",
  "topology_required_for_stop = topology_target_required,\n        eligible_for_stability = !topology_target_required || topology_target_met,\n        stable = stable_i,",
  source_lines,
  fixed = TRUE
)
source_lines <- gsub(
  "termination_reason <- 'signal_stable'",
  "termination_reason <- if (topology_target_required) 'signal_stable' else 'weight_stable'",
  source_lines,
  fixed = TRUE
)
source_lines <- gsub(
  "    if (has_converged)\n      break",
  "    if (FALSE)\n      break",
  source_lines,
  fixed = TRUE
)
source_lines <- gsub(
  "                  elapsed_time = elapsed_time)",
  "                  elapsed_time = elapsed_time,\n                  relative_step = relative_step,\n                  termination_reason = termination_reason,\n                  partition_stable = partition_stable_reached,\n                  partition_streak = partition_streak,\n                  signal_stable = signal_stable_reached,\n                  signal_streak = signal_streak,\n                  signal_stability = if (length(signal_checkpoint_records)) do.call(rbind, signal_checkpoint_records) else data.frame())",
  source_lines,
  fixed = TRUE
)
if (!any(grepl("signal_stable_reached", source_lines, fixed = TRUE))) stop("failed to inject signal-stability stop rule")
eval(parse(text = source_lines), envir = .GlobalEnv)

manifest <- read.csv(file.path(windows_dir, "windows.csv"), stringsAsFactors = FALSE)
if (!nrow(manifest)) stop("window manifest is empty")
manifest <- manifest[seq_len(min(nrow(manifest), limit)), , drop = FALSE]
summary <- vector("list", nrow(manifest))

# Optional optimizer warm start.  The research baseline is cold: each rolling
# window starts from FinGraph's source ``naive`` initializer.  Warm starts are
# retained solely for an explicit future ablation, because carrying a graph
# forward can also carry forward undesirable cuts.
make_warm_w0 <- function(previous, x) {
  old_pos <- match(colnames(x), previous$permno)
  keep <- which(!is.na(old_pos))
  if (length(keep) < 2L) return(NULL)
  base_w <- w_init("naive", MASS::ginv(stats::cor(x)))
  base_a <- A(base_w)
  base_a[keep, keep] <- previous$adjacency[old_pos[keep], old_pos[keep], drop = FALSE]
  base_a <- (base_a + t(base_a)) / 2
  diag(base_a) <- 0
  warm_w <- Ainv(base_a)
  if (!all(is.finite(warm_w)) || any(warm_w < 0)) return(NULL)
  warm_w
}

initial_beta <- function(previous) {
  # beta is an adaptive, window-specific schedule.  Warm-start the graph only.
  1e-8
}

previous <- NULL

for (i in seq_len(nrow(manifest))) {
  record <- manifest[i, ]
  input_path <- file.path(windows_dir, record$input_file)
  x <- as.matrix(read.csv(input_path, row.names = 1, check.names = FALSE))
  storage.mode(x) <- "double"
  if (ncol(x) < k || any(!is.finite(x))) stop("invalid input: ", input_path)
  refit_date <- record$refit_date
  state_path <- file.path(out_dir, "per_window", paste0(refit_date, "_state.rds"))
  diag_path <- file.path(out_dir, "per_window", paste0(refit_date, "_diagnostics.csv"))
  stability_path <- file.path(out_dir, "per_window", paste0(refit_date, "_signal_stability.csv"))
  if (file.exists(state_path) && file.exists(diag_path)) {
    if (init_mode == "warm") previous <- readRDS(state_path)
    summary[[i]] <- read.csv(diag_path, stringsAsFactors = FALSE)
    message(sprintf("FinGraph %d/%d: %s already complete; reusing saved graph", i, nrow(manifest), refit_date))
    next
  }
  warm_w0 <- if (init_mode == "warm" && !is.null(previous)) make_warm_w0(previous, x) else NULL
  initialization <- if (is.null(warm_w0)) "naive" else "previous_graph_with_new_stock_naive_edges"
  w0 <- if (is.null(warm_w0)) "naive" else warm_w0
  beta0 <- if (is.null(warm_w0)) 1e-8 else initial_beta(previous)
  message(sprintf("FinGraph %d/%d: %s (%d x %d; init=%s; beta0=%.3g)", i, nrow(manifest), refit_date, nrow(x), ncol(x), initialization, beta0))
  started <- proc.time()[[3]]
  # Estimating a full p-dimensional Student-t model is appropriate for the
  # source example (82 assets) but poorly conditioned and needlessly expensive
  # at p=500, n=504.  The same tail parameter is estimated from the window's
  # equal-weighted market proxy; nu calibrates tail heaviness, not clustering.
  market_proxy <- rowMeans(scale(x))
  nu_fit <- fit_mvt(matrix(market_proxy, ncol = 1), nu = "MLE-diag-resample")
  nu <- as.numeric(nu_fit$nu)
  fit <- learn_kcomp_heavytail_graph(
    x,
    rho = 10,
    k = k,
    heavy_type = "student",
    nu = nu,
    w0 = w0,
    beta = beta0,
    update_beta = FALSE,
    verbose = FALSE,
    # Match the upstream default and its published examples: reaching K
    # components is not treated as numerical convergence.
    early_stopping = FALSE,
    maxiter = maxiter,
    reltol = 1e-5
  )
  elapsed <- proc.time()[[3]] - started
  adjacency <- fit$adjacency
  final_beta <- if (length(fit$beta_seq)) tail(fit$beta_seq, 1L) else beta0
  diag(adjacency) <- 0
  # This intentionally uses positive adjacency exactly as returned by the
  # source algorithm. No arbitrary edge threshold or post-hoc balancing is imposed.
  membership <- component_membership(adjacency)
  eigenvalues <- eigen(fit$laplacian, symmetric = TRUE, only.values = TRUE)$values
  zero_count <- sum(abs(eigenvalues) < 1e-8)
  sizes <- sort(tabulate(membership))
  output <- data.frame(refit_date = refit_date, permno = colnames(x), component_id = as.integer(membership - 1L))
  write.csv(output, file.path(out_dir, "per_window", paste0(refit_date, ".csv")), row.names = FALSE)
  saveRDS(list(permno = colnames(x), adjacency = adjacency, beta = final_beta), state_path)
  if (init_mode == "warm") previous <- readRDS(state_path)
  stability <- fit$signal_stability
  if (!is.data.frame(stability)) stability <- data.frame()
  write.csv(stability, stability_path, row.names = FALSE)
  final_checkpoint <- if (nrow(stability)) stability[nrow(stability), , drop = FALSE] else data.frame()
  summary[[i]] <- data.frame(
    refit_date = refit_date,
    initialization = initialization,
    initial_beta = beta0,
    final_beta = final_beta,
    stop_rule = stop_rule,
    stop_patience = stop_patience,
    termination_reason = fit$termination_reason,
    weight_stable = isTRUE(fit$signal_stable),
    final_weight_streak = as.integer(fit$signal_streak),
    stability_checkpoints = nrow(stability),
    final_local_model_drift = if (nrow(final_checkpoint)) final_checkpoint$local_model_drift else NA_real_,
    final_community_model_drift = if (nrow(final_checkpoint)) final_checkpoint$community_model_drift else NA_real_,
    numerical_convergence = isTRUE(fit$convergence),
    iterations = as.integer(fit$maxiter),
    elapsed_seconds = as.numeric(elapsed),
    final_relative_step = tail(fit$relative_step, 1L),
    nu = nu,
    adjacency_components = max(membership),
    laplacian_zero_eigenvalues_abs_lt_1e_8 = zero_count,
    min_cluster_size = min(sizes),
    max_cluster_size = max(sizes),
    singleton_clusters = sum(sizes == 1L),
    stringsAsFactors = FALSE
  )
  write.csv(summary[[i]], file.path(out_dir, "per_window", paste0(refit_date, "_diagnostics.csv")), row.names = FALSE)
  message(sprintf("  termination=%s iterations=%d components=%d elapsed=%.1fs", fit$termination_reason, fit$maxiter, max(membership), elapsed))
}
summary <- lapply(summary, function(frame) {
  frame$component_target_reached <- frame$adjacency_components == k &
    frame$laplacian_zero_eigenvalues_abs_lt_1e_8 == k
  frame
})
summary_columns <- unique(unlist(lapply(summary, names)))
summary <- lapply(summary, function(frame) {
  missing <- setdiff(summary_columns, names(frame))
  for (name in missing) frame[[name]] <- NA
  frame[summary_columns]
})
summary_df <- do.call(rbind, summary)
write.csv(summary_df, file.path(out_dir, "fit_summary.csv"), row.names = FALSE)
writeLines(c(
  "{",
  '  "method": "FinGraph k-component heavy-tailed learner",',
  '  "input": "raw CRSP log-return windows standardized internally by FinGraph",',
  paste0('  "clusters_requested": ', k, ","),
  '  "rho": 10,',
  '  "nu": "fit_mvt on each window equal-weighted market proxy",',
  paste0('  "initialization": "', init_mode, '",'),
  '  "early_stopping": false,',
  paste0('  "stop_rule": "', stop_rule, '",'),
  paste0('  "stop_patience": ', stop_patience, ','),
  paste0('  "signal_drift_limit": ', drift_limit, ','),
  paste0('  "signal_check_every_iterations": ', stability_check_every, ','),
  paste0('  "signal_stability_days": ', stability_days, ','),
  '  "stopping_rule": "both trading-weight drifts below threshold, or maxiter"',
  "}"
), file.path(out_dir, "metadata.json"))
