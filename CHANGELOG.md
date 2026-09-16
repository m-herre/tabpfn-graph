# Changelog

## 0.2.0

Breaking changes are listed first. The package had not been published, so no
deprecation cycle is provided.

### Breaking

- `approximation_nodes` is replaced by `approximation_pivots`. Approximate
  descriptors no longer build a random induced subgraph; they estimate the
  whole-graph quantity (pivot-sampled betweenness, sampled-source distances,
  rescaled triangle counts, truncated-spectrum heat traces).
- WL initial labels default to the categorical node/edge attributes discovered
  during `fit` (`wl_node_attributes="auto"`). Pass `None` for the previous,
  purely structural behavior.
- The WL vocabulary is pruned by document frequency and capped
  (`wl_min_graph_count`, `wl_max_features`), so the table width no longer grows
  with the training set.
- Spectral descriptors default to the normalized Laplacian (`laplacian=`), so
  the NetLSD time grid is comparable across graph sizes and
  `spectral__algebraic_connectivity` is the normalized spectral gap.
- `motifs__average_clustering` and `motifs__core_number__*` are removed; those
  quantities were exact duplicates of `local_profile` columns.
- Histogram boundaries are deduplicated, so discrete attributes get fewer bins
  instead of structurally empty ones, and bin column counts change.
- `networkx>=3.5` is required, and `transform` refuses to run under a different
  NetworkX feature release than the one used at `fit`.

### Added

- Edge weight support: `edge_weight`, `edge_weight_agg`, and
  `edge_weight_semantics`. Weighted strength profiles, weighted clustering,
  weighted PageRank/betweenness/assortativity/shortest paths, and a weighted
  Laplacian spectrum.
- Direction-aware columns whenever any training graph is directed: reciprocity,
  in/out degree summaries and correlation, strongly connected components,
  largest-SCC fraction, acyclicity, and PageRank on the native arc set.
- `motifs__square_clustering` and `basic__is_bipartite`, so bipartite datasets
  are not described entirely by identically-zero triangle columns.
- Largest-component path descriptors (`paths__lcc_*`) and component size
  summaries, which stay defined for disconnected graphs.
- `undefined="nan"` to stop conflating an undefined descriptor with a genuine
  zero.
- `diagnostics_` plus `fit`-time warnings for ignored edge weights, mixed
  directedness, extreme size heterogeneity, all-bipartite datasets, and
  aggressive WL pruning.
- `tabpfn_graph.column_report` and `prune_uninformative=True`, which drops
  constant and exactly duplicated columns as a fit-learned schema decision.
- `docs/graph-types.md`, covering which graph families need a non-default
  configuration.

### Fixed

- Mixed directed/undirected batches produced silent NaN columns, or dropped the
  in/out degree columns entirely, depending on which graph came first. The
  schema now depends only on fitted state.
- `build-system.requires` allowed hatchling 1.25/1.26, which cannot build the
  PEP 639 license metadata this project declares.
- Removed the deprecated `License ::` classifier, which PyPI rejects alongside
  a license expression, and added `[project.urls]`.

## 0.1.0

Initial internal release.
