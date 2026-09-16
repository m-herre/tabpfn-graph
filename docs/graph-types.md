# Graph families and how to configure for them

The descriptors in this package are domain-general, but several graph families
carry their signal in a place a default configuration does not look. This page
states what each family needs and what remains out of scope, so the required
handling is a configuration decision rather than a silent one.

`fit` inspects the training graphs and warns about the mismatches it can see
(ignored edge weights, mixed directedness, extreme size heterogeneity, an
all-bipartite dataset, aggressive WL pruning). `extractor.diagnostics_` holds
the same information as data.

## Unweighted, undirected, size-homogeneous graphs

The default configuration. Descriptors run on the simple-undirected projection
and the class signal is expected to be topological shape plus the marginal
distribution of attributes. Small molecules and most TU datasets sit here.

## Weighted graphs

Connectomes, correlation networks, transport and financial graphs, and any
similarity graph. **Weights are ignored unless you ask for them**, because a
weight has no universal meaning.

```python
GraphFeatureExtractor(edge_weight="weight", edge_weight_semantics="similarity")
```

`edge_weight_semantics` is the part that matters scientifically. Betweenness and
shortest-path descriptors need a traversal *cost*:

- `"similarity"` (default): a larger weight is a stronger tie, so the cost is
  `1 / w`. Correlation, co-occurrence, capacity, and affinity weights are of
  this kind. A zero similarity is the absence of a tie, so it becomes an
  infinite cost.
- `"distance"`: the weight already is a length or cost and is used directly. A
  zero distance is a legitimate free traversal, not an unreachable edge.

Passing similarity weights to a shortest-path routine that treats them as
distances inverts the meaning of every path descriptor; that is why this choice
has no default guess. Parallel and reciprocal arcs collapse with
`edge_weight_agg` (`sum` by default).

Negative weights are rejected under both semantics, during `fit` and
`transform`. They have no meaningful inverse and make shortest paths
ill-defined, and mapping them to an infinite cost would silently delete the edge
from the path and betweenness descriptors while leaving it in the degree and
clustering descriptors. An edge whose weight is missing or non-numeric on every
arc keeps weight `0.0` and is unreachable for path purposes.

Weighted configuration adds strength (weighted degree) profiles and weighted
clustering, and switches PageRank, betweenness, assortativity, shortest paths,
and the Laplacian spectrum to their weighted definitions.

## Directed graphs

Citation, web, food-web, regulatory, and call graphs. Most descriptors are
defined on undirected graphs and keep running on the projection, but direction
gets its own columns whenever any training graph is directed: reciprocity,
in/out degree distributions and their correlation, strongly connected component
count, largest-SCC fraction, acyclicity, and PageRank on a direction-preserving
projection of the native arcs.

PageRank on the projection is close to a rescaled degree, so
`centrality__directed_pagerank__*` is the informative one for directed input.

Mixed batches are supported: an undirected graph is described as its own
symmetrization (every edge reciprocal, in-degree equal to out-degree), so one
schema covers the batch without missing values. `fit` warns when a batch is
mixed.

## Bipartite graphs

User–item, author–paper, and transaction graphs. Triangles, transitivity, and
triangle clustering are identically zero by construction, so the `motifs` group
also reports square clustering, which is the standard four-cycle analogue.
`basic__is_bipartite` is always available, and `fit` warns when every training
graph is bipartite.

## Size-heterogeneous collections

Ego networks and social graphs, where node counts span orders of magnitude.
Extensive descriptors (counts and sums) then dominate every split, and the
pooled quantile histogram boundaries are set mostly by the largest graphs. Each
summary block also reports intensive statistics (mean, quantiles, normalized
histogram proportions), and NetLSD columns are normalized by graph size, so
prefer those and consider dropping the extensive columns. `fit` warns when the
largest training graph exceeds 50x the median.

## Disconnected graphs

Path descriptors average over reachable ordered pairs only, and diameter and
average path length are undefined for a disconnected graph. Largest-component
variants (`paths__lcc_*`) stay defined, and `basic__component_size__*` plus
`paths__lcc_fraction` describe the component structure, so a singleton and a
two-clique graph are no longer both summarized as zero.

## Large graphs

`comprehensive` is `O(n^3)` time and `O(n^2)` memory in its spectral group, so
it refuses graphs above `max_exact_nodes` rather than silently degrading.
`allow_approximate=True` switches to estimators of the whole-graph quantity:

| Descriptor | Approximation |
|---|---|
| node/edge betweenness | Brandes–Pich pivot sampling (`approximation_pivots` sources) |
| shortest-path distribution | sampled-source BFS/Dijkstra; unbiased for the distance distribution |
| diameter | maximum eccentricity over sampled sources, i.e. a lower bound |
| triangles, transitivity, square clustering | sampled nodes rescaled to the full graph; wedge count stays exact |
| clustering profile | sampled nodes |
| Laplacian spectrum, NetLSD | Lanczos extreme eigenvalues with a linearly interpolated interior |

`approx__active` records per graph whether the approximate path was taken. The
column schema is identical either way. Approximations are seeded and
reproducible for a given input, but unlike the exact descriptors they are not
invariant under relabeling.

## Out of scope

- **Heterogeneous graphs / knowledge graphs.** `torch_geometric.data.HeteroData`
  has no top-level `edge_index` and is rejected rather than flattened.
- **Temporal or dynamic graphs.** Timestamps are treated as ordinary edge
  attributes; there is no temporal descriptor.
- **Multi-target and multilabel graph properties.** Single-target only.

## What the representation still does not model

Attributes reach the table in two ways: as marginal distributions (the
`attributes` group) and as WL initial labels (`wl_node_attributes` /
`wl_edge_attributes`, categorical attributes by default). WL labels are what
couple attributes to topology; with `wl_node_attributes=None` the table is a bag
of shapes alongside a bag of labels, with no interaction between them. Numeric
and high-cardinality attributes are never used as WL labels, because they would
make almost every subtree hash unique, so continuous node signals still reach
the model only as marginals.
