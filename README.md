# tabpfn-graph

`tabpfn-graph` turns a collection of graphs into a schema-stable pandas table and applies
TabPFN or any scikit-learn-compatible estimator. It targets single-target, graph-level
classification and regression.

```python
from tabpfn_graph import GraphClassifier, GraphRegressor

clf = GraphClassifier().fit(graphs_train, y_train)
predictions = clf.predict(graphs_test)

reg = GraphRegressor(estimator=my_pipeline).fit(graphs_train, y_train)
```

The representation uses established graph descriptors: Local Degree/Topological Profile
statistics, Weisfeiler–Lehman subtree hashes over categorical node/edge labels, attribute
distribution aggregation, and optional centrality, paths, motifs, Laplacian, and NetLSD
summaries. This follows the graph-level descriptor evidence from
[LTP](https://arxiv.org/abs/2305.00724) and [MOLTOP](https://arxiv.org/abs/2407.12136). The
broader graph-to-table foundation-model pattern has also been explored for node tasks by
[G2T-FM](https://arxiv.org/abs/2508.20906) and [TabPFN-GN](https://arxiv.org/abs/2512.08798);
those papers do not imply that this package implements their node-level methods.

Weighted, directed, bipartite, disconnected, and size-heterogeneous graphs each need a
different configuration, and `fit` warns when the training data looks like a mismatch.
`docs/graph-types.md` in the source repository states what each family needs.

## Install

```bash
pip install tabpfn-graph
```

Python 3.10–3.14 is supported. The standard installation includes TabPFN, NetworkX (3.5 or
newer, because WL subtree hashes changed in that release), NumPy, pandas, SciPy,
scikit-learn, and joblib. Optional extras are:

```bash
pip install 'tabpfn-graph[fast]'    # Networkit for compatible primitives
pip install 'tabpfn-graph[client]'  # use hosted clients as user-supplied estimators
pip install 'tabpfn-graph[dev]'     # tests, lint, typing, packaging
```

The first local TabPFN fit may require accepting the checkpoint terms and downloading model
weights. Package source code is Apache-2.0; TabPFN checkpoints have separate terms. More detail
is available in `docs/model-access.md` in the source repository.

## NetworkX quickstarts

Classification with the default local TabPFN:

```python
import networkx as nx
from tabpfn_graph import GraphClassifier

graphs = [nx.path_graph(5), nx.cycle_graph(5), nx.star_graph(4), nx.complete_graph(5)]
y = [0, 1, 0, 1]

clf = GraphClassifier(random_state=0).fit(graphs, y)
labels = clf.predict([nx.path_graph(7), nx.cycle_graph(7)])
probabilities = clf.predict_proba([nx.path_graph(7), nx.cycle_graph(7)])
```

Regression uses the same extraction contract:

```python
from tabpfn_graph import GraphRegressor

reg = GraphRegressor(random_state=0).fit(graphs, [1.2, 2.5, 0.8, 4.1])
values = reg.predict(graphs)
```

The default local estimators are created lazily during `fit`, with TabPFN's local text
transformation enabled. Importing or constructing `GraphClassifier()` does not access model
weights.

## PyTorch Geometric datasets

PyG is intentionally optional. Pass a homogeneous iterable of `torch_geometric.data.Data`
objects when it is installed:

```python
from torch_geometric.datasets import TUDataset
from tabpfn_graph import GraphClassifier

dataset = TUDataset(root="data/TU", name="MUTAG")
graphs = [data for data in dataset]
y = [int(data.y.item()) for data in dataset]
model = GraphClassifier().fit(graphs[:150], y[:150])
prediction = model.predict(graphs[150:])
```

PyG's common doubled-edge representation is recognized: if every non-loop arc has a reciprocal
arc with matching multiplicity, each pair is collapsed into one undirected logical edge.

## Standalone feature extraction

```python
from tabpfn_graph import GraphFeatureExtractor

extractor = GraphFeatureExtractor(features="balanced", n_jobs=-1, random_state=0)
X_train = extractor.fit_transform(graphs_train)
X_test = extractor.transform(graphs_test)

assert list(X_train.columns) == list(X_test.columns)
```

`X_train` and `X_test` are pandas DataFrames. Numeric values remain numeric, graph-level
categoricals use pandas categorical dtype, and semantic documents use pandas string dtype.

## Custom estimators

No text encoding is inserted for custom estimators. For numeric-only feature tables:

```python
from sklearn.ensemble import RandomForestClassifier
from tabpfn_graph import GraphClassifier

model = GraphClassifier(
    features="balanced",
    estimator=RandomForestClassifier(n_estimators=500, random_state=0),
).fit(graphs_train, y_train)
```

XGBoost and CatBoost-style estimators work the same way:

```python
from xgboost import XGBClassifier

model = GraphClassifier(
    features="balanced",
    estimator=XGBClassifier(n_estimators=500, random_state=0),
).fit(graphs_train, y_train)
```

CatBoost can consume categorical/text columns when configured with their column names or
indices. Hosted TabPFN clients can be passed through `estimator=`; the package does not require a
particular client API beyond sklearn-style `fit` and `predict`.

For a text-aware sklearn pipeline, explicitly select and transform the document column:

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn_graph import GraphClassifier, GraphFeatureExtractor

extractor = GraphFeatureExtractor(
    features=("basic", "text"),
    text_attributes=("node.description",),
)
preprocess = ColumnTransformer([
    ("text", TfidfVectorizer(), "node_attr__description__document"),
    ("numeric", StandardScaler(), ["basic__n_nodes", "basic__n_edges_native"]),
])
estimator = make_pipeline(preprocess, LogisticRegression(random_state=0))
model = GraphClassifier(feature_extractor=extractor, estimator=estimator)
```

## Weighted and directed graphs

Structural descriptors ignore edge weights unless you name the attribute, because a weight has
no universal meaning. Path and betweenness descriptors need a traversal cost, so the semantics
are explicit rather than guessed:

```python
extractor = GraphFeatureExtractor(
    edge_weight="weight",
    edge_weight_semantics="similarity",  # cost = 1 / w; use "distance" when w is a length
)
```

That switches strength profiles, weighted clustering, PageRank, betweenness, assortativity,
shortest paths, and the Laplacian spectrum to their weighted definitions. When `edge_weight` is
left unset but the training graphs carry numeric edge attributes, `fit` warns.

Directed input additionally gets reciprocity, in/out degree summaries and their correlation,
strongly connected components, acyclicity, and PageRank on a direction-preserving projection of
the native arcs; PageRank on the
undirected projection is close to a rescaled degree. Mixed batches are supported and yield one
schema, with undirected graphs described as their own symmetrization.

## Diagnostics

```python
extractor = GraphFeatureExtractor().fit(graphs_train)
extractor.diagnostics_        # sizes, directedness, bipartiteness, WL vocabulary, versions
```

`fit` warns about ignored edge weights, mixed directedness, extreme size heterogeneity,
all-bipartite datasets, and aggressive WL pruning. `fit_transform` also records how many
columns are constant or exactly duplicated on the training data, available standalone as
`tabpfn_graph.column_report(frame)`. Setting `prune_uninformative=True` turns that into a
fit-learned schema decision and drops those columns.

## Feature selection

Presets are `fast`, `balanced` (default), and `comprehensive`:

- `fast`: native/basic topology and node, edge, graph metadata, and text aggregation.
- `balanced`: fast plus clustering/core/LDP profiles and two-iteration hashed WL counts.
- `comprehensive`: balanced plus centrality, paths, motifs, Laplacian, and NetLSD summaries.

Or select groups directly:

```python
extractor = GraphFeatureExtractor(features=("basic", "wl", "spectral"))
```

Valid groups are `basic`, `local_profile`, `attributes`, `text`, `wl`, `centrality`, `paths`,
`motifs`, and `spectral`. Expensive groups preflight against `max_exact_nodes=2000`. To make the
change in semantics explicit, larger graphs require either a raised exact limit or
`allow_approximate=True`, which switches to pivot-sampled betweenness, sampled-source
distances, rescaled triangle counts, and a truncated-spectrum heat trace — estimators of the
whole-graph quantity, recorded per graph in `approx__active`.

## Scope

Single-target graph-level classification and regression on homogeneous NetworkX or PyG batches.
Heterogeneous graphs (`HeteroData`), temporal graphs, and multilabel targets are out of scope.

This is an alpha release. The descriptors are established ones and the schema contract is
tested, but the package is not backed by a broad benchmark study; treat it as a descriptor
baseline rather than a method shown to beat tuned GBDT-on-descriptors or GNNs.

Detailed feature semantics, graph-family guidance, evaluation guidance, and reproducible
benchmarks are kept in the source repository under `docs/` and `benchmarks/`. Benchmark code is
not part of the published wheel or source distribution.
