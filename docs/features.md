# Features, semantics, and complexity

## Provenance

The implementation deliberately composes maintained primitives rather than
introducing a new graph representation. Local profiles follow LDP/LTP's per-node
degree and neighbor-degree minimum, maximum, mean, and standard deviation
followed by graph-level distributions. WL uses NetworkX's published
`weisfeiler_lehman_subgraph_hashes`, with categorical node and edge attributes
supplied as initial labels by default. Centrality, paths, motifs, core,
clustering, assortativity, and Laplacian construction use NetworkX; numerical
summaries use NumPy/SciPy. NetLSD columns are heat traces of the normalized
Laplacian spectrum at configured times, normalized by graph size.

These features are descriptor baselines, not claimed implementations of LTP,
MOLTOP, or NetLSD in their entirety.

`docs/graph-types.md` covers which graph families need a non-default
configuration and why.

## Native semantics and descriptor projection

Columns beginning with `basic__..._native` preserve the input graph's
directedness, self-loops, reciprocal arcs, and parallel edges. The table also
records flags/counts for those properties.

Descriptors whose definitions need an ordinary undirected graph use one explicit
projection:

1. Copy all nodes and node attributes.
2. Forget direction, collapsing reciprocal arcs.
3. Collapse parallel edges, aggregating `edge_weight` with `edge_weight_agg`
   when weights are configured.
4. Drop self-loops.

No silent projection is used for native edge or degree statistics, and direction
additionally gets its own columns (see below). A PyG batch is converted once to
canonical NetworkX graphs. When every non-loop PyG arc has a reciprocal arc with
matching multiplicity, paired arcs are treated as one undirected edge; otherwise
the graph remains directed.

## Edge weights

With `edge_weight=None` every structural descriptor is computed on the
unweighted projection, and weights reach the table only as the marginal
distribution of an edge attribute. `fit` warns when training graphs carry
numeric edge attributes in this case.

`edge_weight="<name>"` switches strength profiles, weighted clustering,
PageRank, betweenness, assortativity, shortest paths, and the Laplacian spectrum
to their weighted definitions. `edge_weight_semantics` decides how a weight
becomes a traversal cost — `"similarity"` uses `1 / w`, `"distance"` uses `w`
— because path-like descriptors are otherwise inverted with respect to the
domain's meaning.

## Direction

Whenever any training graph is directed, the table gains reciprocity, in/out
degree summaries and their correlation, strongly connected component count,
largest-SCC fraction, acyclicity, and PageRank on the native arc set. An
undirected graph in the same batch is described as its own symmetrization, so
mixed batches keep one schema without missing values.

## Groups and approximate complexity

Let `n=|V|`, `m=|E|`, `b` be the histogram count, and `d` an attribute vector
width.

| Group | Main operations | Typical cost |
|---|---|---|
| `basic` | size, density, components, isolates, bipartiteness, degree, direction | `O(n + m)` |
| `attributes` | scalar/vector/category aggregation | `O((n + m) d)` |
| `text` | deterministic sort and bounded join | `O(k log k)` per text attribute |
| `local_profile` | neighbor-degree, clustering, core | generally `O(n + m)`; clustering depends on wedges |
| `wl` | NetworkX WL subtree hashes | roughly `O(iterations * (n + m))` plus sorting/hashing |
| `centrality` | PageRank, node/edge betweenness, assortativity | betweenness dominates, about `O(nm)` unweighted |
| `paths` | all-pairs unweighted shortest paths | `O(n(n + m))` |
| `motifs` | triangles, transitivity, square clustering | graph-structure dependent |
| `spectral` | dense Laplacian eigendecomposition and heat traces | `O(n^3)` time, `O(n^2)` memory |

`comprehensive` therefore checks `max_exact_nodes` before extraction. With
`allow_approximate=True`, expensive descriptors switch to estimators of the
whole-graph quantity rather than exact values on a sampled subgraph; see the
table in `docs/graph-types.md`. The output schema does not change, and
`approx__active` records which path each graph took. This is an explicit
approximation, not an algorithm selected silently by graph size.

Installing `[fast]` lets `backend="auto"` use Networkit for compatible exact
component/isolate primitives. `backend="networkx"` forces the canonical
implementation; `backend="networkit"` fails clearly when the extra is absent.

## Schema learning

Only `fit` graphs determine:

- attribute names and numeric/vector/categorical/text roles;
- consistent vector width;
- quantile histogram boundaries;
- low-cardinality categorical vocabulary and the `other` bucket;
- WL initial labels and the pruned WL hash vocabulary;
- whether the batch contains directed graphs; and
- exact output column order.

Numeric node/edge columns contain missing count, count, validity, sum, mean,
population standard deviation, min/max, quartiles, median, and normalized
distribution bins. Consistent vectors get the same columns per dimension.
Categorical values get count/proportion pairs plus `other`. High-cardinality
strings and explicit `text_attributes` become sorted, permutation-invariant
documents. Documents are bounded by `text_max_items` and `text_max_chars`, with
original and truncated counts.

Histogram boundaries are deduplicated interior quantiles. A discrete attribute
therefore gets fewer, non-empty bins rather than the structurally empty bins
that repeated boundaries produce, and an attribute with no usable variation gets
no histogram columns at all.

Graph metadata scalars, categories, and text are retained directly where
compatible. Unseen categories map to `__other__`.

Node IDs and multigraph keys are excluded unless `include_node_ids=True` or
`include_edge_keys=True`. Enabling them intentionally breaks relabeling
invariance and may expose private or target-correlated identifiers.

## Weisfeiler-Lehman columns

WL initial labels come from `wl_node_attributes` and `wl_edge_attributes`.
`"auto"` (the default) uses every categorical attribute discovered during `fit`;
`None` reproduces purely structural WL. Numeric and high-cardinality attributes
are never used as labels, because they would make nearly every hash unique. Edge
labels are built from the sorted token multiset of every arc collapsed into a
logical edge, so they stay deterministic and relabeling-invariant for
multigraphs and reciprocal arc pairs.

Keeping every observed hash would make the table width grow with the training
set, which conflicts with the fixed feature budget of a tabular foundation
model. Hashes appearing in fewer than `wl_min_graph_count` training graphs are
dropped (the threshold is skipped for fewer than ten training graphs, where
document frequency is not informative), and the remainder is capped at
`wl_max_features`, most frequent first. Discarded mass is reported in
`wl__other_count`, and `fit` warns when pruning is substantial.

WL subtree hashes for graphs without node or edge attributes changed in NetworkX
3.5. The package therefore requires `networkx>=3.5`, records the version used at
`fit`, and refuses to `transform` under a different feature release rather than
silently reading every `wl__` column as zero.

## Undefined values

A descriptor that is mathematically undefined for a graph (assortativity of a
graph with no variation, diameter of a disconnected graph, algebraic
connectivity of a disconnected graph) is written as `undefined` with a
neighboring `__valid` column. `undefined="zero"` (the default) suits estimators
that reject NaN. `undefined="nan"` avoids conflating "undefined" with a genuine
zero and suits TabPFN and gradient-boosting estimators, which handle missing
values natively.

Empty, singleton, disconnected, directed, and multigraph inputs retain a stable
schema under either setting.

## Diagnostics

`fit` stores `diagnostics_`: graph counts, size distribution, how many graphs
are directed, multigraph, bipartite, or connected, the discovered attribute
kinds, the WL vocabulary before and after pruning, and the NetworkX version.
With `diagnostics=True` (the default) it also warns about ignored edge weights,
mixed directedness, extreme size heterogeneity, all-bipartite datasets, and
aggressive WL pruning.

`fit_transform` additionally records a `columns` report of constant and exactly
duplicated columns, which is available standalone as
`tabpfn_graph.column_report(frame)`.

`prune_uninformative=True` turns that report into a schema decision: columns
that are constant or exact duplicates **on the training graphs** are dropped
from the fitted schema, like any other fit-learned choice, so `transform` stays
stable. It costs one extra extraction pass during `fit` and is off by default.

## Caching and determinism

`memory=None` disables persistent caching. A joblib `Memory` or path enables it.
Cache keys cover the graph object content, fitted schema, selected groups,
extractor configuration, and a cache format version. Mutating a graph changes
its key. Parallel extraction preserves input order. Exact descriptors are
invariant under relabeling; approximate descriptors are seeded and reproducible
for a given input but are not relabeling-invariant.

## Leakage risks

Always fit the extractor inside each training fold. Reusing a globally fitted
extractor leaks histogram boundaries, vocabularies, attribute schema, WL
vocabulary, and pruning decisions. Identifiers, timestamps, source names, split
labels, and graph metadata may encode the target — including through
`wl_node_attributes="auto"`, which will happily use a target-correlated
categorical attribute as a WL label. Duplicate or isomorphic graphs across folds
can also inflate results; audit them before fitting and group duplicate WL hash
candidates in cross-validation.
