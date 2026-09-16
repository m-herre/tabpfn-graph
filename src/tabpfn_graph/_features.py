"""Schema-learning graph feature extraction."""

from __future__ import annotations

import copy
import hashlib
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._adapters import adapt_batch, simple_directed_projection, simple_undirected_projection

FeatureGroup = Literal[
    "basic",
    "local_profile",
    "attributes",
    "text",
    "wl",
    "centrality",
    "paths",
    "motifs",
    "spectral",
]

_GROUPS: set[str] = {
    "basic",
    "local_profile",
    "attributes",
    "text",
    "wl",
    "centrality",
    "paths",
    "motifs",
    "spectral",
}
_PRESETS = {
    "fast": ("basic", "attributes", "text"),
    "balanced": ("basic", "attributes", "text", "local_profile", "wl"),
    "comprehensive": (
        "basic",
        "attributes",
        "text",
        "local_profile",
        "wl",
        "centrality",
        "paths",
        "motifs",
        "spectral",
    ),
}
_EXPENSIVE = {"centrality", "paths", "motifs", "spectral"}
_CACHE_FORMAT_VERSION = 2

#: Weisfeiler-Lehman subtree hashes for graphs without node or edge attributes
#: changed in NetworkX 3.5 (bugfix). A schema fitted under an older release
#: would silently stop matching, so the package requires >= 3.5 and records the
#: version it was fitted with.
_WL_STABLE_NETWORKX = (3, 5)

#: Below this many training graphs, document-frequency pruning is skipped.
_WL_MIN_CORPUS = 10

_PROFILE_NAMES = ("degree", "neighbor_min", "neighbor_max", "neighbor_mean", "neighbor_std")
_WEIGHTED_PROFILE_NAMES = ("strength", "neighbor_strength_mean")
_DISTANCE_KEY = "__tabpfn_graph_distance__"
_WL_NODE_KEY = "__tabpfn_graph_wl_node__"
_WL_EDGE_KEY = "__tabpfn_graph_wl_edge__"


def _networkx_version() -> tuple[int, int]:
    parts = re.findall(r"\d+", nx.__version__)
    if len(parts) < 2:
        return (0, 0)
    return (int(parts[0]), int(parts[1]))


@dataclass(frozen=True)
class AttributeSpec:
    scope: Literal["node", "edge"]
    name: str
    kind: Literal["numeric", "vector", "categorical", "text"]
    dimension: int = 1
    categories: tuple[str, ...] = ()
    bin_edges: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class GraphAttributeSpec:
    name: str
    kind: Literal["numeric", "categorical", "text"]
    categories: tuple[str, ...] = ()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(
        value, bool | np.bool_
    )


def _numeric_vector(value: Any) -> np.ndarray | None:
    if isinstance(value, str | bytes | dict) or _is_missing(value):
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    return array.astype(float)


def _token(value: Any) -> str:
    return str(value)


def _learn_edges(values: Sequence[float], n_bins: int) -> tuple[float, ...]:
    """Return strictly increasing interior quantile boundaries.

    Duplicate quantiles are removed. A discrete attribute therefore produces
    fewer, non-empty bins instead of the structurally empty bins that repeated
    boundaries would create, and an attribute with no usable variation produces
    no histogram columns at all.
    """

    if n_bins <= 1:
        return ()
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return ()
    quantiles = np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])
    unique = np.unique(quantiles)
    interior = unique[(unique > finite.min()) & (unique < finite.max())]
    return tuple(float(value) for value in interior)


def _summary(
    prefix: str,
    values: Sequence[float] | np.ndarray,
    *,
    undefined: float,
    include_sum: bool = True,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    valid = array.size > 0
    result = {
        f"{prefix}__valid": float(valid),
        f"{prefix}__count": float(array.size),
    }
    stats = ("mean", "std", "min", "q25", "median", "q75", "max")
    names = ("sum", *stats) if include_sum else stats
    if not valid:
        result.update({f"{prefix}__{name}": undefined for name in names})
        return result
    if include_sum:
        result[f"{prefix}__sum"] = float(array.sum())
    result.update(
        {
            f"{prefix}__mean": float(array.mean()),
            f"{prefix}__std": float(array.std(ddof=0)),
            f"{prefix}__min": float(array.min()),
            f"{prefix}__q25": float(np.quantile(array, 0.25)),
            f"{prefix}__median": float(np.quantile(array, 0.5)),
            f"{prefix}__q75": float(np.quantile(array, 0.75)),
            f"{prefix}__max": float(array.max()),
        }
    )
    return result


def _histogram(prefix: str, values: Sequence[float], edges: Sequence[float]) -> dict[str, float]:
    n_bins = len(edges) + 1
    if n_bins < 2:
        return {}
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    counts = np.bincount(np.searchsorted(np.asarray(edges), array, side="right"), minlength=n_bins)
    denominator = max(1, int(array.size))
    return {
        f"{prefix}__bin_{index:02d}": float(count / denominator)
        for index, count in enumerate(counts)
    }


def _bounded_document(
    values: Sequence[Any], max_items: int, max_chars: int
) -> tuple[str, int, int]:
    items = sorted(_token(value) for value in values if not _is_missing(value))
    original_count = len(items)
    selected = items[:max_items]
    document = "\n".join(selected)
    truncated = original_count - len(selected)
    if len(document) > max_chars:
        document = document[:max_chars]
        truncated = max(1, truncated)
    return document, original_count, truncated


def _safe_float(value: Any, undefined: float) -> tuple[float, float]:
    if _is_missing(value):
        return undefined, 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return undefined, 0.0
    return (result, 1.0) if np.isfinite(result) else (undefined, 0.0)


def _canonical_order(graph: nx.Graph) -> nx.Graph:
    """Rebuild a graph with a label-determined insertion order.

    Sampling primitives in NetworkX draw from the graph's iteration order, so a
    canonical order makes approximate descriptors reproducible for a given
    input. Approximations remain sampling-based and are therefore not invariant
    under relabeling, unlike the exact descriptors.
    """

    ordered = nx.Graph()
    ordered.add_nodes_from(sorted(graph.nodes(data=True), key=lambda item: repr(item[0])))
    ordered.add_edges_from(
        sorted(graph.edges(data=True), key=lambda item: (repr(item[0]), repr(item[1])))
    )
    return ordered


def _cached_extract(graph: nx.Graph, extractor: GraphFeatureExtractor) -> dict[str, Any]:
    return extractor._extract_one_uncached(graph)


def column_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize constant and exactly duplicated columns of a feature table.

    Both indicate wasted model capacity: a constant column carries no signal on
    this dataset, and a duplicated column repeats one that is already present.
    """

    n_rows = len(frame)
    constant: list[str] = []
    duplicates: list[list[str]] = []
    seen: dict[Any, str] = {}
    for name in frame.columns:
        column = frame[name]
        if n_rows and column.nunique(dropna=False) <= 1:
            constant.append(str(name))
        try:
            key = pd.util.hash_pandas_object(column, index=False).values.tobytes()
        except TypeError:
            continue
        if key in seen:
            for group in duplicates:
                if group[0] == seen[key]:
                    group.append(str(name))
                    break
            else:
                duplicates.append([seen[key], str(name)])
        else:
            seen[key] = str(name)
    return {
        "n_rows": int(n_rows),
        "n_columns": int(frame.shape[1]),
        "n_constant_columns": len(constant),
        "constant_columns": constant,
        "n_duplicate_columns": sum(len(group) - 1 for group in duplicates),
        "duplicate_column_groups": duplicates,
    }


class GraphFeatureExtractor(TransformerMixin, BaseEstimator):
    """Convert graph-level datasets into stable pandas feature tables.

    The fitted schema contains only information learned from training graphs:
    attribute names/types, vector widths, categorical vocabularies, histogram
    boundaries, and WL hash vocabulary. Descriptor algorithms always operate on
    a simple-undirected, loop-free projection unless their name starts with
    ``native__``, with the exception of the explicitly directed columns.

    Parameters that change what the descriptors mean:

    ``edge_weight``
        Name of the numeric edge attribute carrying weights. When ``None`` (the
        default) every structural descriptor is computed on the unweighted
        graph, and ``fit`` warns if the training graphs carry numeric edge
        attributes that are being ignored.
    ``edge_weight_semantics``
        Whether a larger weight means a stronger tie (``"similarity"``, the
        default, traversal cost ``1 / w``) or a longer edge (``"distance"``).
        Path and betweenness descriptors need a cost, so this choice is
        required rather than assumed.
    ``wl_node_attributes`` / ``wl_edge_attributes``
        Attributes used as WL initial labels. ``"auto"`` (the default) uses the
        categorical attributes discovered during ``fit``; ``None`` reproduces
        purely structural WL.
    ``laplacian``
        ``"normalized"`` (the default) keeps eigenvalues in ``[0, 2]`` so that a
        fixed NetLSD time grid is comparable across graph sizes;
        ``"combinatorial"`` restores the unnormalized spectrum.
    ``undefined``
        Value written where a descriptor is mathematically undefined. ``"zero"``
        (the default) suits estimators that reject NaN; ``"nan"`` avoids
        conflating "undefined" with a genuine zero and suits TabPFN and
        gradient-boosting estimators.
    ``prune_uninformative``
        Learn, from the training graphs alone, which columns are constant or
        exact duplicates and drop them from the schema. Off by default because
        it costs one extra extraction pass during ``fit``.
    """

    def __init__(
        self,
        features: str | tuple[FeatureGroup, ...] = "balanced",
        *,
        n_bins: int = 8,
        max_categories: int = 20,
        edge_weight: str | None = None,
        edge_weight_agg: Literal["sum", "mean", "max", "min"] = "sum",
        edge_weight_semantics: Literal["similarity", "distance"] = "similarity",
        text_attributes: tuple[str, ...] | None = None,
        text_max_items: int = 256,
        text_max_chars: int = 16_384,
        wl_iterations: int = 2,
        wl_digest_size: int = 8,
        wl_node_attributes: str | Sequence[str] | None = "auto",
        wl_edge_attributes: str | Sequence[str] | None = "auto",
        wl_min_graph_count: int = 2,
        wl_max_features: int | None = 2_048,
        laplacian: Literal["normalized", "combinatorial"] = "normalized",
        netlsd_times: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        max_exact_nodes: int = 2_000,
        allow_approximate: bool = False,
        approximation_pivots: int = 512,
        undefined: Literal["zero", "nan"] = "zero",
        backend: Literal["auto", "networkx", "networkit"] = "auto",
        include_node_ids: bool = False,
        include_edge_keys: bool = False,
        diagnostics: bool = True,
        prune_uninformative: bool = False,
        n_jobs: int | None = 1,
        random_state: int | None = 0,
        memory: str | Path | Memory | None = None,
    ) -> None:
        self.features = features
        self.n_bins = n_bins
        self.max_categories = max_categories
        self.edge_weight = edge_weight
        self.edge_weight_agg = edge_weight_agg
        self.edge_weight_semantics = edge_weight_semantics
        self.text_attributes = text_attributes
        self.text_max_items = text_max_items
        self.text_max_chars = text_max_chars
        self.wl_iterations = wl_iterations
        self.wl_digest_size = wl_digest_size
        self.wl_node_attributes = wl_node_attributes
        self.wl_edge_attributes = wl_edge_attributes
        self.wl_min_graph_count = wl_min_graph_count
        self.wl_max_features = wl_max_features
        self.laplacian = laplacian
        self.netlsd_times = netlsd_times
        self.max_exact_nodes = max_exact_nodes
        self.allow_approximate = allow_approximate
        self.approximation_pivots = approximation_pivots
        self.undefined = undefined
        self.backend = backend
        self.include_node_ids = include_node_ids
        self.include_edge_keys = include_edge_keys
        self.diagnostics = diagnostics
        self.prune_uninformative = prune_uninformative
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.memory = memory

    # ------------------------------------------------------------------ fit

    def fit(self, graphs: Iterable[Any], y: Any = None) -> GraphFeatureExtractor:
        del y
        self.feature_groups_ = self._resolve_groups()
        self._validate_parameters()
        self.backend_ = self._resolve_backend()
        self.networkx_version_ = _networkx_version()
        batch = adapt_batch(graphs)
        self.input_kind_ = batch.kind
        self.has_directed_ = any(graph.is_directed() for graph in batch.graphs)
        self.has_multigraph_ = any(graph.is_multigraph() for graph in batch.graphs)
        self._preflight(batch.graphs)
        self.attribute_schema_ = self._learn_attribute_schema(batch.graphs)
        self.emitted_attribute_schema_ = self._emitted_specs()
        self.graph_attribute_schema_ = self._learn_graph_attribute_schema(batch.graphs)
        self.wl_node_labels_, self.wl_edge_labels_ = self._resolve_wl_labels()
        self.profile_bin_edges_ = self._learn_profile_bins(batch.graphs)
        self.wl_vocabulary_ = self._learn_wl_vocabulary(batch.graphs)
        schema_row = self._extract_one_uncached(batch.graphs[0])
        self.feature_names_out_ = np.asarray(list(schema_row), dtype=object)
        self.text_feature_names_ = self._text_columns()
        self.categorical_feature_names_ = self._categorical_columns()
        self.category_levels_ = {
            f"graph_attr__{spec.name}": (*spec.categories, "__other__", "__missing__")
            for spec in self.graph_attribute_schema_
            if spec.kind == "categorical"
        }
        special = set(self.text_feature_names_) | set(self.categorical_feature_names_)
        self.numeric_feature_names_ = tuple(
            name for name in self.feature_names_out_ if name not in special
        )
        self.diagnostics_ = self._schema_diagnostics(batch.graphs)
        if self.prune_uninformative:
            self._prune(batch.graphs)
        if self.diagnostics:
            self._warn_about_dataset(self.diagnostics_)
        return self

    def _prune(self, graphs: Sequence[nx.Graph]) -> None:
        """Drop columns that are constant or exactly duplicated on the training graphs.

        The decision is learned from ``fit`` graphs only, like every other part
        of the schema, so ``transform`` stays stable. It costs one extra
        extraction pass and is therefore opt-in.
        """

        frame = self._frame(self._extract_many(graphs))
        report = column_report(frame)
        removable = set(report["constant_columns"])
        for group in report["duplicate_column_groups"]:
            removable.update(group[1:])
        self.diagnostics_["pruned"] = {
            "n_before": int(frame.shape[1]),
            "n_removed": len(removable),
            "removed_columns": sorted(removable),
        }
        if not removable:
            return
        self.feature_names_out_ = np.asarray(
            [name for name in self.feature_names_out_ if name not in removable], dtype=object
        )
        keep = set(self.feature_names_out_)
        self.text_feature_names_ = tuple(n for n in self.text_feature_names_ if n in keep)
        self.categorical_feature_names_ = tuple(
            n for n in self.categorical_feature_names_ if n in keep
        )
        self.numeric_feature_names_ = tuple(n for n in self.numeric_feature_names_ if n in keep)

    def fit_transform(
        self, graphs: Iterable[Any], y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        del fit_params
        graphs = list(graphs)
        self.fit(graphs, y)
        frame = self.transform(graphs)
        if self.diagnostics:
            self.diagnostics_["columns"] = column_report(frame)
        return frame

    def transform(self, graphs: Iterable[Any]) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        self._check_networkx_version()
        batch = adapt_batch(graphs, expected_kind=self.input_kind_, allow_empty=True)
        self._preflight(batch.graphs)
        return self._frame(self._extract_many(batch.graphs))

    def _frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows).reindex(columns=self.feature_names_out_)
        for name in self.numeric_feature_names_:
            frame[name] = frame[name].astype(float)
        for name in self.text_feature_names_:
            frame[name] = frame[name].astype("string")
        for name in self.categorical_feature_names_:
            frame[name] = pd.Categorical(frame[name], categories=self.category_levels_[name])
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    # ------------------------------------------------------- configuration

    def _resolve_groups(self) -> tuple[str, ...]:
        if isinstance(self.features, str):
            if self.features not in _PRESETS:
                raise ValueError(
                    f"unknown feature preset {self.features!r}; choose from {sorted(_PRESETS)}"
                )
            return _PRESETS[self.features]
        groups = tuple(self.features)
        unknown = {str(group) for group in groups if group not in _GROUPS}
        if unknown:
            raise ValueError(f"unknown feature groups: {sorted(unknown)}")
        if len(groups) != len(set(groups)):
            raise ValueError("feature groups must not contain duplicates")
        if not groups:
            raise ValueError("at least one feature group is required")
        return groups

    def _validate_parameters(self) -> None:
        if self.n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        if self.max_categories < 1:
            raise ValueError("max_categories must be at least 1")
        if self.wl_iterations < 1 or self.wl_digest_size < 1:
            raise ValueError("WL iterations and digest size must be positive")
        if self.wl_min_graph_count < 1:
            raise ValueError("wl_min_graph_count must be at least 1")
        if self.wl_max_features is not None and self.wl_max_features < 1:
            raise ValueError("wl_max_features must be positive or None")
        if self.max_exact_nodes < 1 or self.approximation_pivots < 2:
            raise ValueError("node limits must be positive")
        if self.text_max_items < 1 or self.text_max_chars < 1:
            raise ValueError("text bounds must be positive")
        if self.edge_weight_agg not in {"sum", "mean", "max", "min"}:
            raise ValueError("edge_weight_agg must be 'sum', 'mean', 'max', or 'min'")
        if self.edge_weight_semantics not in {"similarity", "distance"}:
            raise ValueError("edge_weight_semantics must be 'similarity' or 'distance'")
        if self.laplacian not in {"normalized", "combinatorial"}:
            raise ValueError("laplacian must be 'normalized' or 'combinatorial'")
        if self.undefined not in {"zero", "nan"}:
            raise ValueError("undefined must be 'zero' or 'nan'")

    def _resolve_backend(self) -> str:
        if self.backend not in {"auto", "networkx", "networkit"}:
            raise ValueError("backend must be 'auto', 'networkx', or 'networkit'")
        if self.backend == "networkx":
            return "networkx"
        try:
            import networkit  # noqa: F401
        except ImportError:
            if self.backend == "networkit":
                raise ImportError(
                    "backend='networkit' requires the fast extra: pip install 'tabpfn-graph[fast]'"
                ) from None
            return "networkx"
        return "networkit"

    def _check_networkx_version(self) -> None:
        if "wl" not in self.feature_groups_:
            return
        current = _networkx_version()
        if current != self.networkx_version_:
            raise ValueError(
                f"this extractor learned its WL vocabulary under NetworkX "
                f"{self.networkx_version_[0]}.{self.networkx_version_[1]} but NetworkX "
                f"{current[0]}.{current[1]} is installed. WL subtree hashes are not stable "
                "across NetworkX feature releases, so every wl__ column would silently read "
                "zero. Reinstall the fitted NetworkX version or refit the extractor."
            )

    @property
    def _undefined(self) -> float:
        return float("nan") if self.undefined == "nan" else 0.0

    def _seed(self) -> int:
        digest = hashlib.blake2b(str(self.random_state).encode(), digest_size=4).digest()
        return int.from_bytes(digest, "little")

    def _preflight(self, graphs: Sequence[nx.Graph]) -> None:
        if not (_EXPENSIVE & set(self.feature_groups_)) or self.allow_approximate:
            return
        oversized = [
            (index, graph.number_of_nodes())
            for index, graph in enumerate(graphs)
            if len(graph) > self.max_exact_nodes
        ]
        if oversized:
            index, size = oversized[0]
            raise ValueError(
                f"graph {index} has {size} nodes, exceeding max_exact_nodes={self.max_exact_nodes} "
                "for exact comprehensive descriptors; increase max_exact_nodes, remove expensive "
                "groups, or set allow_approximate=True to switch to sampled-source distances, "
                "pivot-sampled betweenness, and a truncated-spectrum heat trace"
            )

    # ------------------------------------------------------------ projection

    def _projection(self, graph: nx.Graph) -> nx.Graph:
        if self.edge_weight is None:
            return simple_undirected_projection(graph)
        return simple_undirected_projection(
            graph,
            weight=self.edge_weight,
            weight_agg=self.edge_weight_agg,
            distance_key=_DISTANCE_KEY,
            distance_from_similarity=self.edge_weight_semantics == "similarity",
        )

    def _directed_projection(self, graph: nx.Graph) -> nx.Graph:
        return simple_directed_projection(
            graph,
            weight=self.edge_weight,
            weight_agg=self.edge_weight_agg,
            distance_key=_DISTANCE_KEY if self.edge_weight is not None else None,
            distance_from_similarity=self.edge_weight_semantics == "similarity",
        )

    @property
    def _weight_key(self) -> str | None:
        return "weight" if self.edge_weight is not None else None

    @property
    def _distance_key(self) -> str | None:
        return _DISTANCE_KEY if self.edge_weight is not None else None

    def _is_approximate(self, projected: nx.Graph) -> bool:
        return self.allow_approximate and projected.number_of_nodes() > self.max_exact_nodes

    def _pivots(self, graph: nx.Graph) -> list[Any]:
        ordered = sorted(graph.nodes, key=repr)
        size = min(self.approximation_pivots, len(ordered))
        rng = np.random.default_rng(self._seed())
        chosen = rng.choice(len(ordered), size=size, replace=False)
        return [ordered[index] for index in sorted(chosen)]

    # -------------------------------------------------------- schema learning

    def _needs_attribute_schema(self) -> bool:
        if {"attributes", "text"} & set(self.feature_groups_):
            return True
        auto = "auto" in (self.wl_node_attributes, self.wl_edge_attributes)
        return "wl" in self.feature_groups_ and auto

    def _scope_values(self, graphs: Sequence[nx.Graph], scope: str, name: str) -> list[Any]:
        values: list[Any] = []
        for graph in graphs:
            records = graph.nodes(data=True) if scope == "node" else graph.edges(data=True)
            values.extend(attrs.get(name) for *_, attrs in records if name in attrs)
        return values

    def _learn_attribute_schema(self, graphs: Sequence[nx.Graph]) -> tuple[AttributeSpec, ...]:
        if not self._needs_attribute_schema():
            return ()
        names: dict[str, set[str]] = {"node": set(), "edge": set()}
        for graph in graphs:
            for _, attrs in graph.nodes(data=True):
                names["node"].update(map(str, attrs))
            for *_, attrs in graph.edges(data=True):
                names["edge"].update(map(str, attrs))
        specs: list[AttributeSpec] = []
        explicit = set(self.text_attributes or ())
        for scope in ("node", "edge"):
            for name in sorted(names[scope]):
                values = [
                    value
                    for value in self._scope_values(graphs, scope, name)
                    if not _is_missing(value)
                ]
                kind: Literal["numeric", "vector", "categorical", "text"]
                if not values:
                    kind = "numeric"
                elif name in explicit or f"{scope}.{name}" in explicit:
                    kind = "text"
                elif all(_is_number(value) for value in values):
                    kind = "numeric"
                else:
                    vectors = [_numeric_vector(value) for value in values]
                    dimensions = {len(vector) for vector in vectors if vector is not None}
                    if all(vector is not None for vector in vectors) and len(dimensions) == 1:
                        kind = "vector"
                    elif all(isinstance(value, str | bool | np.bool_) for value in values):
                        unique = {_token(value) for value in values}
                        kind = "categorical" if len(unique) <= self.max_categories else "text"
                    else:
                        kind = "text"
                if kind == "numeric":
                    edges = (
                        _learn_edges(
                            [float(value) for value in values if _is_number(value)], self.n_bins
                        ),
                    )
                    specs.append(AttributeSpec(scope, name, kind, bin_edges=edges))
                elif kind == "vector":
                    dimension = next(iter(dimensions))
                    per_dimension = [
                        _learn_edges(
                            [float(vector[index]) for vector in vectors if vector is not None],
                            self.n_bins,
                        )
                        for index in range(dimension)
                    ]
                    specs.append(
                        AttributeSpec(scope, name, kind, dimension, bin_edges=tuple(per_dimension))
                    )
                elif kind == "categorical":
                    categories = tuple(sorted({_token(value) for value in values}))
                    specs.append(AttributeSpec(scope, name, kind, categories=categories))
                else:
                    specs.append(AttributeSpec(scope, name, kind))
        return tuple(specs)

    def _emitted_specs(self) -> tuple[AttributeSpec, ...]:
        groups = set(self.feature_groups_)
        return tuple(
            spec
            for spec in self.attribute_schema_
            if ("text" in groups if spec.kind == "text" else "attributes" in groups)
        )

    def _learn_graph_attribute_schema(
        self, graphs: Sequence[nx.Graph]
    ) -> tuple[GraphAttributeSpec, ...]:
        if not ({"attributes", "text"} & set(self.feature_groups_)):
            return ()
        names = sorted({str(name) for graph in graphs for name in graph.graph})
        explicit = set(self.text_attributes or ())
        specs: list[GraphAttributeSpec] = []
        for name in names:
            values = [
                graph.graph[name]
                for graph in graphs
                if name in graph.graph and not _is_missing(graph.graph[name])
            ]
            kind: Literal["numeric", "categorical", "text"]
            if not values or all(_is_number(value) for value in values):
                kind = "numeric"
            elif name in explicit or f"graph.{name}" in explicit:
                kind = "text"
            elif all(isinstance(value, str | bool | np.bool_) for value in values):
                unique = {_token(value) for value in values}
                kind = "categorical" if len(unique) <= self.max_categories else "text"
            else:
                continue
            if kind == "text" and "text" not in self.feature_groups_:
                continue
            if kind != "text" and "attributes" not in self.feature_groups_:
                continue
            categories = (
                tuple(sorted({_token(value) for value in values})) if kind == "categorical" else ()
            )
            specs.append(GraphAttributeSpec(name, kind, categories))
        return tuple(specs)

    def _resolve_wl_labels(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if "wl" not in self.feature_groups_:
            return (), ()

        def resolve(spec: str | Sequence[str] | None, scope: str) -> tuple[str, ...]:
            if spec is None:
                return ()
            if isinstance(spec, str):
                if spec != "auto":
                    raise ValueError(
                        f"wl_{scope}_attributes must be 'auto', None, or a sequence of names"
                    )
                return tuple(
                    item.name
                    for item in self.attribute_schema_
                    if item.scope == scope and item.kind == "categorical"
                )
            return tuple(str(name) for name in spec)

        return resolve(self.wl_node_attributes, "node"), resolve(self.wl_edge_attributes, "edge")

    def _profile_values(self, projected: nx.Graph) -> dict[str, list[float]]:
        degree = dict(projected.degree())
        names = list(_PROFILE_NAMES)
        weighted = self.edge_weight is not None
        if weighted:
            strength = dict(projected.degree(weight="weight"))
            names.extend(_WEIGHTED_PROFILE_NAMES)
        profiles: dict[str, list[float]] = {name: [] for name in names}
        for node in projected:
            profiles["degree"].append(float(degree[node]))
            neighbors = list(projected.neighbors(node))
            neighbor_degrees = np.asarray([degree[other] for other in neighbors], dtype=float)
            if neighbor_degrees.size:
                profiles["neighbor_min"].append(float(neighbor_degrees.min()))
                profiles["neighbor_max"].append(float(neighbor_degrees.max()))
                profiles["neighbor_mean"].append(float(neighbor_degrees.mean()))
                profiles["neighbor_std"].append(float(neighbor_degrees.std()))
            else:
                for name in ("neighbor_min", "neighbor_max", "neighbor_mean", "neighbor_std"):
                    profiles[name].append(0.0)
            if weighted:
                profiles["strength"].append(float(strength[node]))
                neighbor_strength = np.asarray(
                    [strength[other] for other in neighbors], dtype=float
                )
                profiles["neighbor_strength_mean"].append(
                    float(neighbor_strength.mean()) if neighbor_strength.size else 0.0
                )
        return profiles

    def _learn_profile_bins(self, graphs: Sequence[nx.Graph]) -> dict[str, tuple[float, ...]]:
        if "local_profile" not in self.feature_groups_:
            return {}
        names = list(_PROFILE_NAMES)
        if self.edge_weight is not None:
            names.extend(_WEIGHTED_PROFILE_NAMES)
        collected: dict[str, list[float]] = {name: [] for name in names}
        for graph in graphs:
            for name, values in self._profile_values(self._projection(graph)).items():
                collected[name].extend(values)
        return {name: _learn_edges(values, self.n_bins) for name, values in collected.items()}

    def _wl_graph(self, graph: nx.Graph, projected: nx.Graph) -> nx.Graph:
        """Build the labeled, simple-undirected graph handed to WL hashing.

        Node labels join the configured node attributes. Edge labels join the
        sorted tokens of every arc collapsed into that logical edge, so the
        label is deterministic and invariant under relabeling even for
        multigraphs and reciprocal arc pairs.
        """

        if not (self.wl_node_labels_ or self.wl_edge_labels_):
            return projected
        labeled = nx.Graph()
        for node, attrs in projected.nodes(data=True):
            label = "|".join(_token(attrs.get(name)) for name in self.wl_node_labels_)
            labeled.add_node(node, **{_WL_NODE_KEY: label})
        if not self.wl_edge_labels_:
            labeled.add_edges_from(projected.edges())
            return labeled
        grouped: dict[Any, list[str]] = {}
        for u, v, attrs in graph.edges(data=True):
            if u == v:
                continue
            key = (u, v) if repr(u) <= repr(v) else (v, u)
            token = "|".join(_token(attrs.get(name)) for name in self.wl_edge_labels_)
            grouped.setdefault(key, []).append(token)
        for (u, v), tokens in grouped.items():
            labeled.add_edge(u, v, **{_WL_EDGE_KEY: "/".join(sorted(tokens))})
        return labeled

    def _wl_counts(self, graph: nx.Graph, projected: nx.Graph) -> Counter[str]:
        if not projected:
            return Counter()
        target = self._wl_graph(graph, projected)
        with warnings.catch_warnings():
            # The v3.5 hash change is announced unconditionally; the package
            # pins NetworkX >= 3.5 and refuses to transform under a different
            # feature release, so the message is not actionable here.
            warnings.filterwarnings("ignore", message="The hashes produced for graphs")
            hashes = nx.weisfeiler_lehman_subgraph_hashes(
                target,
                node_attr=_WL_NODE_KEY if self.wl_node_labels_ else None,
                edge_attr=_WL_EDGE_KEY if self.wl_edge_labels_ else None,
                iterations=self.wl_iterations,
                digest_size=self.wl_digest_size,
            )
        return Counter(value for node_hashes in hashes.values() for value in node_hashes)

    def _learn_wl_vocabulary(self, graphs: Sequence[nx.Graph]) -> tuple[str, ...]:
        """Select WL hashes by document frequency.

        Keeping every hash makes the table width grow with the training set, so
        hashes seen in fewer than ``wl_min_graph_count`` training graphs are
        dropped and the remainder is capped at ``wl_max_features``, most
        frequent first. Discarded mass stays available in ``wl__other_count``.
        """

        if "wl" not in self.feature_groups_:
            self._wl_vocabulary_stats_ = {"seen": 0, "kept": 0, "threshold": 0}
            return ()
        document_frequency: Counter[str] = Counter()
        for graph in graphs:
            document_frequency.update(set(self._wl_counts(graph, self._projection(graph))))
        seen = len(document_frequency)
        # Document frequency is not informative on a handful of graphs, where
        # the threshold would discard the whole vocabulary.
        threshold = self.wl_min_graph_count if len(graphs) >= _WL_MIN_CORPUS else 1
        kept = [value for value, count in document_frequency.items() if count >= threshold]
        kept.sort(key=lambda value: (-document_frequency[value], value))
        if self.wl_max_features is not None:
            kept = kept[: self.wl_max_features]
        self._wl_vocabulary_stats_ = {
            "seen": seen,
            "kept": len(kept),
            "threshold": threshold,
        }
        return tuple(sorted(kept))

    # ---------------------------------------------------------- extraction

    def _memory_object(self) -> Memory | None:
        if self.memory is None:
            return None
        if isinstance(self.memory, Memory):
            return self.memory
        return Memory(location=str(self.memory), verbose=0)

    def _extract_many(self, graphs: Sequence[nx.Graph]) -> list[dict[str, Any]]:
        memory = self._memory_object()
        if memory is None:
            function = self._extract_one_uncached
        else:
            payload = copy.copy(self)
            payload.memory = None
            payload._cache_format_version_ = _CACHE_FORMAT_VERSION
            cached = memory.cache(_cached_extract)

            def function(graph: nx.Graph) -> dict[str, Any]:
                return cached(graph, payload)

        if self.n_jobs in (None, 1) or len(graphs) < 2:
            return [function(graph) for graph in graphs]
        return Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(function)(graph) for graph in graphs
        )

    def _extract_one_uncached(self, graph: nx.Graph) -> dict[str, Any]:
        row: dict[str, Any] = {}
        groups = set(self.feature_groups_)
        projected = self._projection(graph)
        if "basic" in groups:
            row.update(self._basic_features(graph, projected))
        if "local_profile" in groups:
            row.update(self._local_profile_features(projected))
        if {"attributes", "text"} & groups:
            row.update(self._attribute_features(graph))
            row.update(self._graph_attribute_features(graph))
        if self.include_node_ids:
            document, count, truncated = _bounded_document(
                list(graph), self.text_max_items, self.text_max_chars
            )
            row.update(
                {
                    "identifiers__node_ids": document,
                    "identifiers__node_ids__original_count": float(count),
                    "identifiers__node_ids__truncated_count": float(truncated),
                }
            )
        if self.include_edge_keys:
            keys = [key for *_, key in graph.edges(keys=True)] if graph.is_multigraph() else []
            document, count, truncated = _bounded_document(
                keys, self.text_max_items, self.text_max_chars
            )
            row.update(
                {
                    "identifiers__edge_keys": document,
                    "identifiers__edge_keys__original_count": float(count),
                    "identifiers__edge_keys__truncated_count": float(truncated),
                }
            )
        if "wl" in groups:
            counts = self._wl_counts(graph, projected)
            row.update({f"wl__{value}": float(counts[value]) for value in self.wl_vocabulary_})
            vocabulary = set(self.wl_vocabulary_)
            row["wl__other_count"] = float(
                sum(count for value, count in counts.items() if value not in vocabulary)
            )
        if _EXPENSIVE & groups:
            approximate = self._is_approximate(projected)
            target = _canonical_order(projected) if approximate else projected
            row["approx__active"] = float(approximate)
            if "centrality" in groups:
                row.update(self._centrality_features(graph, target, approximate))
            if "paths" in groups:
                row.update(self._path_features(target, approximate))
            if "motifs" in groups:
                row.update(self._motif_features(target, approximate))
            if "spectral" in groups:
                row.update(self._spectral_features(target, approximate))
        return row

    # ------------------------------------------------------------ descriptors

    def _basic_features(self, graph: nx.Graph, projected: nx.Graph) -> dict[str, float]:
        undefined = self._undefined
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        loops = nx.number_of_selfloops(graph)
        reciprocal = 0
        if graph.is_directed():
            seen: set[frozenset[Any]] = set()
            for u, v in graph.edges():
                pair = frozenset((u, v))
                if u != v and pair not in seen and graph.has_edge(v, u):
                    reciprocal += 1
                    seen.add(pair)
        unique_native = {
            (u, v) if graph.is_directed() else frozenset((u, v)) for u, v in graph.edges()
        }
        density_valid = n_nodes > 1
        components, isolates, sizes = self._component_stats(projected)
        result = {
            "basic__n_nodes": float(n_nodes),
            "basic__n_edges_native": float(n_edges),
            "basic__n_edges_projected": float(projected.number_of_edges()),
            "basic__is_directed": float(graph.is_directed()),
            "basic__is_multigraph": float(graph.is_multigraph()),
            "basic__self_loops": float(loops),
            "basic__reciprocal_pairs": float(reciprocal),
            "basic__parallel_edge_excess": float(max(0, n_edges - len(unique_native))),
            "basic__density_native": float(nx.density(graph)) if density_valid else undefined,
            "basic__density_native__valid": float(density_valid),
            "basic__components": float(components),
            "basic__isolates": float(isolates),
            "basic__connected": float(n_nodes > 0 and nx.is_connected(projected)),
            "basic__connected__valid": float(n_nodes > 0),
            "basic__is_bipartite": float(nx.is_bipartite(projected)) if n_nodes else undefined,
            "basic__is_bipartite__valid": float(n_nodes > 0),
            "basic__largest_component_fraction": (
                float(max(sizes) / n_nodes) if sizes and n_nodes else undefined
            ),
            "basic__largest_component_fraction__valid": float(bool(sizes)),
        }
        result.update(_summary("basic__component_size", sizes, undefined=undefined))
        result.update(
            _summary(
                "basic__degree_native",
                [degree for _, degree in graph.degree()],
                undefined=undefined,
            )
        )
        if self.edge_weight is not None:
            result.update(
                _summary(
                    "basic__edge_weight",
                    [
                        float(attrs["weight"])
                        for *_, attrs in projected.edges(data=True)
                        if _is_number(attrs.get("weight"))
                    ],
                    undefined=undefined,
                )
            )
        if self.has_directed_:
            result.update(self._directed_features(graph, projected, sizes))
        return result

    def _directed_features(
        self, graph: nx.Graph, projected: nx.Graph, sizes: list[int]
    ) -> dict[str, float]:
        """Direction-aware columns, emitted whenever any training graph is directed.

        An undirected graph is treated as its own symmetrization: every edge is
        reciprocal, in-degree equals out-degree, and strong connectivity equals
        connectivity. Mixed batches therefore keep one schema without padding
        undirected rows with missing values.
        """

        undefined = self._undefined
        n_nodes = graph.number_of_nodes()
        if graph.is_directed():
            in_degrees = [degree for _, degree in graph.in_degree()]
            out_degrees = [degree for _, degree in graph.out_degree()]
            arcs = sum(1 for u, v in graph.edges() if u != v)
            reciprocated = sum(1 for u, v in graph.edges() if u != v and graph.has_edge(v, u))
            reciprocity = float(reciprocated / arcs) if arcs else undefined
            reciprocity_valid = float(bool(arcs))
            strong = nx.number_strongly_connected_components(graph) if n_nodes else 0
            largest_strong = (
                max(len(component) for component in nx.strongly_connected_components(graph))
                if n_nodes
                else 0
            )
            is_dag = float(nx.is_directed_acyclic_graph(graph))
        else:
            in_degrees = out_degrees = [degree for _, degree in graph.degree()]
            reciprocity = 1.0 if graph.number_of_edges() else undefined
            reciprocity_valid = float(bool(graph.number_of_edges()))
            strong = len(sizes)
            largest_strong = max(sizes) if sizes else 0
            is_dag = float(graph.number_of_edges() == 0)
        result = {
            "basic__reciprocity": reciprocity,
            "basic__reciprocity__valid": reciprocity_valid,
            "basic__strongly_connected_components": float(strong),
            "basic__largest_scc_fraction": (
                float(largest_strong / n_nodes) if n_nodes else undefined
            ),
            "basic__largest_scc_fraction__valid": float(n_nodes > 0),
            "basic__is_dag": is_dag,
            "basic__is_dag__valid": float(n_nodes > 0),
        }
        result.update(_summary("basic__in_degree_native", in_degrees, undefined=undefined))
        result.update(_summary("basic__out_degree_native", out_degrees, undefined=undefined))
        usable = len(in_degrees) > 1 and np.std(in_degrees) > 0 and np.std(out_degrees) > 0
        result["basic__in_out_degree_corr"] = (
            float(np.corrcoef(in_degrees, out_degrees)[0, 1]) if usable else undefined
        )
        result["basic__in_out_degree_corr__valid"] = float(usable)
        return result

    def _component_stats(self, graph: nx.Graph) -> tuple[int, int, list[int]]:
        if not graph:
            return 0, 0, []
        if self.backend_ == "networkit":
            import networkit as nk

            positions = {node: index for index, node in enumerate(graph)}
            converted = nk.Graph(len(positions), weighted=False, directed=False)
            for u, v in graph.edges():
                converted.addEdge(positions[u], positions[v])
            components = nk.components.ConnectedComponents(converted).run()
            sizes = sorted(components.getComponentSizes().values(), reverse=True)
            isolates = sum(converted.degree(index) == 0 for index in range(len(positions)))
            return int(components.numberOfComponents()), int(isolates), [int(s) for s in sizes]
        sizes = sorted(
            (len(component) for component in nx.connected_components(graph)), reverse=True
        )
        return len(sizes), nx.number_of_isolates(graph), sizes

    def _local_profile_features(self, projected: nx.Graph) -> dict[str, float]:
        undefined = self._undefined
        row: dict[str, float] = {}
        profiles = self._profile_values(projected)
        for name, values in profiles.items():
            prefix = f"local_profile__{name}"
            row.update(_summary(prefix, values, undefined=undefined))
            row.update(_histogram(prefix, values, self.profile_bin_edges_[name]))
        approximate = self._is_approximate(projected)
        nodes = self._pivots(projected) if approximate else None
        clustering = list(nx.clustering(projected, nodes=nodes).values())
        row.update(_summary("local_profile__clustering", clustering, undefined=undefined))
        if self.edge_weight is not None:
            weighted = list(nx.clustering(projected, nodes=nodes, weight="weight").values())
            row.update(
                _summary("local_profile__weighted_clustering", weighted, undefined=undefined)
            )
        core = (
            list(nx.core_number(projected).values())
            if projected.number_of_edges()
            else [0.0] * len(projected)
        )
        row.update(_summary("local_profile__core_number", core, undefined=undefined))
        return row

    def _attribute_values(self, graph: nx.Graph, spec: AttributeSpec) -> tuple[list[Any], int]:
        records = graph.nodes(data=True) if spec.scope == "node" else graph.edges(data=True)
        attrs = [record[-1] for record in records]
        return [record.get(spec.name) for record in attrs], len(attrs)

    def _attribute_features(self, graph: nx.Graph) -> dict[str, Any]:
        undefined = self._undefined
        row: dict[str, Any] = {}
        for spec in self.emitted_attribute_schema_:
            values, total = self._attribute_values(graph, spec)
            present = [value for value in values if not _is_missing(value)]
            prefix = f"{spec.scope}_attr__{spec.name}"
            row[f"{prefix}__missing_count"] = float(total - len(present))
            if spec.kind == "numeric":
                numeric = [float(value) for value in present if _is_number(value)]
                row.update(_summary(prefix, numeric, undefined=undefined))
                row.update(_histogram(prefix, numeric, spec.bin_edges[0]))
            elif spec.kind == "vector":
                vectors = [
                    vector
                    for value in present
                    if (vector := _numeric_vector(value)) is not None
                    and len(vector) == spec.dimension
                ]
                row[f"{prefix}__invalid_vector_count"] = float(len(present) - len(vectors))
                for index in range(spec.dimension):
                    numeric = [float(vector[index]) for vector in vectors]
                    dimension_prefix = f"{prefix}__dim_{index:03d}"
                    row.update(_summary(dimension_prefix, numeric, undefined=undefined))
                    row.update(_histogram(dimension_prefix, numeric, spec.bin_edges[index]))
            elif spec.kind == "categorical":
                tokens = [_token(value) for value in present]
                counts = Counter(tokens)
                denominator = max(1, total)
                for category in spec.categories:
                    safe = hashlib.blake2b(category.encode(), digest_size=6).hexdigest()
                    row[f"{prefix}__category_{safe}__count"] = float(counts[category])
                    row[f"{prefix}__category_{safe}__proportion"] = float(
                        counts[category] / denominator
                    )
                other = sum(
                    count for category, count in counts.items() if category not in spec.categories
                )
                row[f"{prefix}__other__count"] = float(other)
                row[f"{prefix}__other__proportion"] = float(other / denominator)
            else:
                document, count, truncated = _bounded_document(
                    present, self.text_max_items, self.text_max_chars
                )
                row[f"{prefix}__document"] = document
                row[f"{prefix}__original_count"] = float(count)
                row[f"{prefix}__truncated_count"] = float(truncated)
        return row

    def _graph_attribute_features(self, graph: nx.Graph) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for spec in self.graph_attribute_schema_:
            value = graph.graph.get(spec.name)
            prefix = f"graph_attr__{spec.name}"
            if spec.kind == "numeric":
                numeric, valid = _safe_float(value, self._undefined)
                row[prefix] = numeric
                row[f"{prefix}__valid"] = valid
            elif spec.kind == "categorical":
                if _is_missing(value):
                    row[prefix] = "__missing__"
                else:
                    token = _token(value)
                    row[prefix] = token if token in spec.categories else "__other__"
            else:
                row[prefix] = "" if _is_missing(value) else _token(value)[: self.text_max_chars]
                row[f"{prefix}__original_chars"] = float(
                    0 if _is_missing(value) else len(_token(value))
                )
                row[f"{prefix}__truncated"] = float(
                    not _is_missing(value) and len(_token(value)) > self.text_max_chars
                )
        return row

    def _text_columns(self) -> tuple[str, ...]:
        columns = [
            f"{spec.scope}_attr__{spec.name}__document"
            for spec in self.emitted_attribute_schema_
            if spec.kind == "text"
        ]
        columns.extend(
            f"graph_attr__{spec.name}"
            for spec in self.graph_attribute_schema_
            if spec.kind == "text"
        )
        if self.include_node_ids:
            columns.append("identifiers__node_ids")
        if self.include_edge_keys:
            columns.append("identifiers__edge_keys")
        return tuple(columns)

    def _categorical_columns(self) -> tuple[str, ...]:
        return tuple(
            f"graph_attr__{spec.name}"
            for spec in self.graph_attribute_schema_
            if spec.kind == "categorical"
        )

    def _centrality_features(
        self, graph: nx.Graph, projected: nx.Graph, approximate: bool
    ) -> dict[str, float]:
        """PageRank, betweenness, and assortativity.

        Betweenness uses Brandes-Pich pivot sampling in approximate mode, which
        estimates the same quantity on the whole graph rather than computing an
        exact value on a sampled subgraph. Directed inputs additionally get
        PageRank on a direction-preserving projection of the native arcs, where
        PageRank is actually defined; on an undirected graph it is close to a
        rescaled degree.
        """

        undefined = self._undefined
        distance = self._distance_key
        result: dict[str, float] = {}
        if not projected:
            empty: dict[str, Sequence[float]] = {
                "pagerank": [],
                "node_betweenness": [],
                "edge_betweenness": [],
            }
            result["centrality__assortativity"] = undefined
            result["centrality__assortativity__valid"] = 0.0
            values = empty
        else:
            pivots = min(self.approximation_pivots, len(projected)) if approximate else None
            values = {
                "pagerank": list(nx.pagerank(projected, weight=self._weight_key).values()),
                "node_betweenness": list(
                    nx.betweenness_centrality(
                        projected,
                        k=pivots,
                        normalized=True,
                        weight=distance,
                        seed=self._seed(),
                    ).values()
                ),
                "edge_betweenness": list(
                    nx.edge_betweenness_centrality(
                        projected,
                        k=pivots,
                        normalized=True,
                        weight=distance,
                        seed=self._seed(),
                    ).values()
                ),
            }
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    raw = float(
                        nx.degree_assortativity_coefficient(projected, weight=self._weight_key)
                    )
                result["centrality__assortativity__valid"] = float(np.isfinite(raw))
                result["centrality__assortativity"] = raw if np.isfinite(raw) else undefined
            except (ValueError, ZeroDivisionError, nx.NetworkXError):
                result["centrality__assortativity"] = undefined
                result["centrality__assortativity__valid"] = 0.0
        for name, sequence in values.items():
            result.update(_summary(f"centrality__{name}", sequence, undefined=undefined))
        if self.has_directed_:
            # The native graph stores weights under edge_weight; only the
            # projection normalizes them to the canonical "weight" key.
            directed = self._directed_projection(graph) if graph.is_directed() else projected
            ranks = (
                list(nx.pagerank(directed, weight=self._weight_key).values()) if directed else []
            )
            result.update(_summary("centrality__directed_pagerank", ranks, undefined=undefined))
        return result

    def _distances_from(self, graph: nx.Graph, sources: Sequence[Any]) -> tuple[list[float], float]:
        distance = self._distance_key
        collected: list[float] = []
        eccentricity = 0.0
        for source in sources:
            if distance is None:
                lengths = nx.single_source_shortest_path_length(graph, source)
            else:
                lengths = nx.single_source_dijkstra_path_length(graph, source, weight=distance)
            reached = [float(value) for target, value in lengths.items() if target != source]
            collected.extend(reached)
            if reached:
                eccentricity = max(eccentricity, max(reached))
        return collected, eccentricity

    def _path_features(self, graph: nx.Graph, approximate: bool) -> dict[str, float]:
        """Shortest-path descriptors, including largest-component variants.

        Distances are collected as ordered reachable pairs, so ``__count`` is
        twice the number of unordered pairs in exact mode. In approximate mode
        the sources are a pivot sample: the distance distribution stays an
        unbiased estimate and ``paths__diameter`` becomes a lower bound.
        """

        undefined = self._undefined
        sources = self._pivots(graph) if approximate else list(graph)
        distances, eccentricity = self._distances_from(graph, sources)
        result = _summary("paths__shortest", distances, undefined=undefined)
        connected = bool(graph) and nx.is_connected(graph)
        result["paths__diameter"] = eccentricity if connected else undefined
        result["paths__diameter__valid"] = float(connected)
        result["paths__average_shortest_path"] = (
            float(np.mean(distances)) if connected and distances else undefined
        )
        result["paths__average_shortest_path__valid"] = float(connected and bool(distances))

        # Largest-component variants stay defined for disconnected graphs, so a
        # singleton and a two-clique graph are no longer both reported as zero.
        if graph:
            components = sorted(nx.connected_components(graph), key=len, reverse=True)
            largest = graph.subgraph(components[0])
            lcc_sources = (
                [node for node in sources if node in components[0]]
                if approximate
                else list(largest)
            ) or list(largest)
            lcc_distances, lcc_eccentricity = self._distances_from(largest, lcc_sources)
        else:
            lcc_distances, lcc_eccentricity = [], 0.0
        result["paths__lcc_diameter"] = lcc_eccentricity if graph else undefined
        result["paths__lcc_diameter__valid"] = float(bool(graph))
        result["paths__lcc_average_shortest_path"] = (
            float(np.mean(lcc_distances)) if lcc_distances else undefined
        )
        result["paths__lcc_average_shortest_path__valid"] = float(bool(lcc_distances))
        result["paths__lcc_fraction"] = (
            float(len(components[0]) / len(graph)) if graph else undefined
        )
        result["paths__lcc_fraction__valid"] = float(bool(graph))
        return result

    def _motif_features(self, graph: nx.Graph, approximate: bool) -> dict[str, float]:
        """Triangle and square statistics.

        ``local_profile`` already reports the clustering and core-number
        distributions, so this group keeps only what it does not duplicate.
        Square clustering is included because triangle-based quantities are
        identically zero on bipartite graphs.
        """

        undefined = self._undefined
        n_nodes = len(graph)
        if not n_nodes:
            return {
                "motifs__triangles": undefined,
                "motifs__triangles__valid": 0.0,
                "motifs__nodes_in_triangles_fraction": undefined,
                "motifs__transitivity": undefined,
                "motifs__transitivity__valid": 0.0,
                **_summary("motifs__square_clustering", [], undefined=undefined),
            }
        sample = self._pivots(graph) if approximate else list(graph)
        triangles = nx.triangles(graph, nodes=sample)
        scale = n_nodes / len(sample)
        total = float(sum(triangles.values())) * scale / 3.0
        in_triangles = float(sum(value > 0 for value in triangles.values())) / len(sample)
        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
        wedges = float((degrees * (degrees - 1) / 2).sum())
        result = {
            "motifs__triangles": total,
            "motifs__triangles__valid": 1.0,
            "motifs__nodes_in_triangles_fraction": in_triangles,
            "motifs__transitivity": float(3 * total / wedges) if wedges else undefined,
            "motifs__transitivity__valid": float(wedges > 0),
        }
        squares = list(nx.square_clustering(graph, nodes=sample).values())
        result.update(_summary("motifs__square_clustering", squares, undefined=undefined))
        return result

    def _laplacian_eigenvalues(self, graph: nx.Graph, approximate: bool) -> np.ndarray:
        """Eigenvalues of the selected Laplacian, truncated when approximating.

        The approximate branch keeps the extreme eigenvalues from a Lanczos
        solve and linearly interpolates the interior, which is the truncation
        NetLSD proposes instead of eigendecomposing a sampled subgraph.
        """

        n_nodes = len(graph)
        if not n_nodes:
            return np.asarray([], dtype=float)
        weight = self._weight_key
        builder = (
            nx.normalized_laplacian_matrix
            if self.laplacian == "normalized"
            else nx.laplacian_matrix
        )
        matrix = builder(graph, weight=weight)
        if not approximate:
            return np.linalg.eigvalsh(matrix.astype(float).toarray())
        from scipy.sparse.linalg import ArpackError, eigsh

        k = int(min(self.approximation_pivots // 2, (n_nodes - 1) // 2))
        if k < 1:
            return np.linalg.eigvalsh(matrix.astype(float).toarray())
        matrix = matrix.astype(float).tocsr()
        try:
            low = np.sort(eigsh(matrix, k=k, which="SA", return_eigenvectors=False))
            high = np.sort(eigsh(matrix, k=k, which="LA", return_eigenvectors=False))
        except (ArpackError, ValueError, RuntimeError):
            return np.asarray([], dtype=float)
        middle = int(n_nodes - 2 * k)
        if middle <= 0:
            return np.sort(np.concatenate([low, high]))
        interpolated = np.linspace(low[-1], high[0], middle + 2)[1:-1]
        return np.sort(np.concatenate([low, interpolated, high]))

    def _spectral_features(self, graph: nx.Graph, approximate: bool) -> dict[str, float]:
        undefined = self._undefined
        eigenvalues = self._laplacian_eigenvalues(graph, approximate)
        result = _summary("spectral__laplacian_eigenvalue", eigenvalues, undefined=undefined)
        if eigenvalues.size:
            tolerance = (
                max(1, len(graph)) * np.finfo(float).eps * max(1.0, float(eigenvalues.max()))
            )
            nonzero = eigenvalues[eigenvalues > tolerance]
            connected = nx.is_connected(graph) if len(graph) else False
            result["spectral__zero_eigenvalues"] = float(eigenvalues.size - nonzero.size)
            result["spectral__algebraic_connectivity"] = (
                float(nonzero.min()) if nonzero.size and connected else undefined
            )
            result["spectral__algebraic_connectivity__valid"] = float(
                len(graph) > 1 and connected and bool(nonzero.size)
            )
        else:
            result.update(
                {
                    "spectral__zero_eigenvalues": undefined,
                    "spectral__algebraic_connectivity": undefined,
                    "spectral__algebraic_connectivity__valid": 0.0,
                }
            )
        for time in self.netlsd_times:
            value = (
                float(np.exp(-float(time) * eigenvalues).mean()) if eigenvalues.size else undefined
            )
            result[f"spectral__netlsd_t_{float(time):g}"] = value
            result[f"spectral__netlsd_t_{float(time):g}__valid"] = float(bool(eigenvalues.size))
        return result

    # ------------------------------------------------------------ diagnostics

    def _schema_diagnostics(self, graphs: Sequence[nx.Graph]) -> dict[str, Any]:
        sizes = [graph.number_of_nodes() for graph in graphs]
        projections = [self._projection(graph) for graph in graphs]
        numeric_edge_attributes = sorted(
            {
                spec.name
                for spec in self.attribute_schema_
                if spec.scope == "edge" and spec.kind in {"numeric", "vector"}
            }
        )
        if not self.attribute_schema_:
            candidates: set[str] = set()
            for graph in graphs:
                for *_, attrs in graph.edges(data=True):
                    candidates.update(name for name, value in attrs.items() if _is_number(value))
            numeric_edge_attributes = sorted(candidates)
        return {
            "n_graphs": len(graphs),
            "graph_size": {
                "min": int(min(sizes)) if sizes else 0,
                "median": float(np.median(sizes)) if sizes else 0.0,
                "max": int(max(sizes)) if sizes else 0,
            },
            "n_directed": sum(graph.is_directed() for graph in graphs),
            "n_multigraph": sum(graph.is_multigraph() for graph in graphs),
            "n_bipartite": sum(nx.is_bipartite(graph) for graph in projections),
            "n_connected": sum(bool(graph) and nx.is_connected(graph) for graph in projections),
            "edge_weight": self.edge_weight,
            "numeric_edge_attributes": numeric_edge_attributes,
            "attribute_kinds": {
                f"{spec.scope}.{spec.name}": spec.kind for spec in self.attribute_schema_
            },
            "wl_node_labels": list(getattr(self, "wl_node_labels_", ())),
            "wl_edge_labels": list(getattr(self, "wl_edge_labels_", ())),
            "wl_vocabulary": dict(getattr(self, "_wl_vocabulary_stats_", {"seen": 0, "kept": 0})),
            "networkx_version": ".".join(str(part) for part in self.networkx_version_),
        }

    def _warn_about_dataset(self, report: dict[str, Any]) -> None:
        n_graphs = report["n_graphs"]
        if self.edge_weight is None and report["numeric_edge_attributes"]:
            warnings.warn(
                "training graphs carry numeric edge attributes "
                f"{report['numeric_edge_attributes']} but edge_weight=None, so every structural "
                "descriptor is computed on the unweighted graph; pass edge_weight=<name> (and "
                "edge_weight_semantics) to use them",
                UserWarning,
                stacklevel=3,
            )
        if 0 < report["n_directed"] < n_graphs:
            warnings.warn(
                f"{report['n_directed']} of {n_graphs} training graphs are directed; direction-"
                "aware columns are emitted for the whole batch and undirected graphs are "
                "described as their own symmetrization",
                UserWarning,
                stacklevel=3,
            )
        median = report["graph_size"]["median"]
        if median and report["graph_size"]["max"] / max(median, 1.0) > 50:
            warnings.warn(
                "training graph sizes span more than 50x the median; extensive descriptors "
                "(counts and sums) will be dominated by graph size rather than structure, and "
                "pooled histogram boundaries are set mostly by the largest graphs",
                UserWarning,
                stacklevel=3,
            )
        if "motifs" in self.feature_groups_ and report["n_bipartite"] == n_graphs and n_graphs:
            warnings.warn(
                "every training graph is bipartite, so triangle-based columns are identically "
                "zero; motifs__square_clustering and local_profile are the informative "
                "alternatives here",
                UserWarning,
                stacklevel=3,
            )
        vocabulary = report["wl_vocabulary"]
        seen, kept = vocabulary.get("seen", 0), vocabulary.get("kept", 0)
        if seen and (kept == 0 or kept < 0.75 * seen):
            warnings.warn(
                f"WL vocabulary pruned from {seen} to {kept} hashes by "
                f"wl_min_graph_count={vocabulary.get('threshold')} and "
                f"wl_max_features={self.wl_max_features}; discarded mass is reported in "
                "wl__other_count",
                UserWarning,
                stacklevel=3,
            )
