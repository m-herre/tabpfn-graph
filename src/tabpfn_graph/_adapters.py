"""Input validation and conversion to the canonical NetworkX representation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class AdaptedBatch:
    graphs: list[nx.Graph]
    kind: Literal["networkx", "pyg"]


def _is_pyg_data(value: Any) -> bool:
    cls = type(value)
    return cls.__module__.startswith("torch_geometric.") and hasattr(value, "edge_index")


def materialize_graphs(graphs: Iterable[Any], *, allow_empty: bool = False) -> list[Any]:
    if isinstance(graphs, nx.Graph | str | bytes) or _is_pyg_data(graphs):
        raise TypeError("graphs must be an iterable of graph objects, not a single graph")
    try:
        values = list(graphs)
    except TypeError as exc:
        raise TypeError(
            "graphs must be an iterable of NetworkX graphs or PyG Data objects"
        ) from exc
    if not values and not allow_empty:
        raise ValueError("at least one graph is required")
    return values


def adapt_batch(
    graphs: Iterable[Any],
    *,
    expected_kind: str | None = None,
    allow_empty: bool = False,
) -> AdaptedBatch:
    values = materialize_graphs(graphs, allow_empty=allow_empty)
    if not values:
        if expected_kind not in {"networkx", "pyg"}:
            raise ValueError("cannot infer graph input type from an empty batch")
        return AdaptedBatch([], expected_kind)  # type: ignore[arg-type]

    nx_flags = [isinstance(value, nx.Graph) for value in values]
    pyg_flags = [_is_pyg_data(value) for value in values]
    if all(nx_flags):
        kind: Literal["networkx", "pyg"] = "networkx"
        result = values
    elif all(pyg_flags):
        kind = "pyg"
        result = [_pyg_to_networkx(value) for value in values]
    else:
        raise TypeError(
            "graph batches must be homogeneous and contain only NetworkX graphs "
            "or only torch_geometric.data.Data objects"
        )
    if expected_kind is not None and kind != expected_kind:
        raise TypeError(
            f"extractor was fitted on {expected_kind} graphs but received {kind} graphs"
        )
    return AdaptedBatch(result, kind)


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attribute_names(data: Any, method: str) -> list[str]:
    fn = getattr(data, method, None)
    if not callable(fn):
        return []
    return [str(name) for name in fn()]


def _pyg_to_networkx(data: Any) -> nx.Graph:
    """Convert PyG Data while collapsing paired arcs of an undirected graph.

    PyG commonly stores each undirected edge twice. If every non-loop arc has a
    reverse arc with the same multiplicity, those pairs become one undirected
    edge. Otherwise direction is preserved. Repeated logical edges remain
    parallel edges.
    """

    edge_index = np.asarray(_to_python(data.edge_index), dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("PyG edge_index must have shape (2, n_edges)")
    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if pairs:
        num_nodes = max(num_nodes, max(max(u, v) for u, v in pairs) + 1)

    counts: dict[tuple[int, int], int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    undirected = all(u == v or counts.get((v, u), 0) == count for (u, v), count in counts.items())
    if undirected:
        logical_positions: list[int] = []
        seen: dict[tuple[int, int], int] = {}
        for position, (u, v) in enumerate(pairs):
            key = (min(u, v), max(u, v))
            target = counts[(u, v)]
            if seen.get(key, 0) < target and (u <= v or counts.get((v, u), 0) == 0):
                logical_positions.append(position)
                seen[key] = seen.get(key, 0) + 1
        logical_pairs = [pairs[position] for position in logical_positions]
    else:
        logical_positions = list(range(len(pairs)))
        logical_pairs = pairs

    multiplicities: dict[tuple[int, int], int] = {}
    for u, v in logical_pairs:
        key = (min(u, v), max(u, v)) if undirected else (u, v)
        multiplicities[key] = multiplicities.get(key, 0) + 1
    multi = any(count > 1 for count in multiplicities.values())
    graph_type = (
        nx.MultiGraph
        if undirected and multi
        else nx.Graph
        if undirected
        else nx.MultiDiGraph
        if multi
        else nx.DiGraph
    )
    graph = graph_type()
    graph.add_nodes_from(range(num_nodes))

    node_names = _attribute_names(data, "node_attrs")
    edge_names = _attribute_names(data, "edge_attrs")
    for name in node_names:
        values = _to_python(getattr(data, name))
        if len(values) != num_nodes:
            continue
        for node, value in enumerate(values):
            graph.nodes[node][name] = value

    edge_values = {name: _to_python(getattr(data, name)) for name in edge_names}
    for position, (u, v) in zip(logical_positions, logical_pairs, strict=True):
        attrs = {
            name: values[position]
            for name, values in edge_values.items()
            if hasattr(values, "__len__") and len(values) == len(pairs)
        }
        graph.add_edge(u, v, **attrs)

    excluded = {"edge_index", "num_nodes", "y", "batch", "ptr", *node_names, *edge_names}
    keys = data.keys() if callable(getattr(data, "keys", None)) else []
    for name in keys:
        if name in excluded:
            continue
        value = _to_python(getattr(data, name))
        if np.isscalar(value) or isinstance(value, str):
            graph.graph[str(name)] = value
    return graph


def _project(
    graph: nx.Graph,
    *,
    directed: bool,
    weight: str | None,
    weight_agg: str,
    distance_key: str | None,
    distance_from_similarity: bool,
) -> nx.Graph:
    """Collapse a graph onto a simple projection, optionally keeping direction.

    Nodes and node attributes are copied and self-loops are dropped. Parallel
    edges always collapse; reciprocal arcs collapse only when ``directed`` is
    false.

    When ``weight`` names an edge attribute, the collapsed edge carries the
    aggregate of every contributing arc under the key ``"weight"``, so
    downstream callers always read the canonical key regardless of what the
    attribute is called on the input graph. Arcs whose weight is missing or
    non-numeric contribute nothing; an edge with no usable weight is given
    weight ``0.0`` and, for path purposes, an infinite traversal cost.

    A negative weight is rejected rather than coerced: under similarity
    semantics it has no meaningful inverse, and under distance semantics it
    makes shortest paths ill-defined. Silently mapping it to an infinite cost
    would delete the edge from every path and betweenness descriptor while
    leaving it in the degree and clustering descriptors.
    """

    projected: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    projected.add_nodes_from((node, dict(attrs)) for node, attrs in graph.nodes(data=True))
    if weight is None:
        for u, v in graph.edges():
            if u != v:
                projected.add_edge(u, v)
        return projected

    collected: dict[tuple[Any, Any], list[float]] = {}
    for u, v, attrs in graph.edges(data=True):
        if u == v:
            continue
        key = (u, v) if directed or repr(u) <= repr(v) else (v, u)
        value = attrs.get(weight)
        collected.setdefault(key, [])
        if isinstance(value, bool | np.bool_) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        if number < 0:
            semantics = "similarity" if distance_from_similarity else "distance"
            raise ValueError(
                f"edge attribute {weight!r} has a negative value ({number}) on edge "
                f"({u!r}, {v!r}); negative weights cannot be turned into a traversal cost "
                f"under edge_weight_semantics={semantics!r}. Rescale the weights to be "
                "non-negative, or drop those edges before extraction."
            )
        collected[key].append(number)

    def aggregate(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        if weight_agg == "mean":
            return float(array.mean())
        if weight_agg == "max":
            return float(array.max())
        if weight_agg == "min":
            return float(array.min())
        return float(array.sum())

    for (u, v), values in collected.items():
        total = aggregate(values) if values else 0.0
        attributes: dict[str, float] = {"weight": total}
        if distance_key is not None:
            if not values:
                # No usable weight anywhere on this edge: unreachable for paths.
                attributes[distance_key] = float(np.inf)
            elif distance_from_similarity:
                # A zero similarity is the absence of a tie.
                attributes[distance_key] = 1.0 / total if total > 0 else float(np.inf)
            else:
                # A zero distance is a legitimate free traversal.
                attributes[distance_key] = total
        projected.add_edge(u, v, **attributes)
    return projected


def simple_undirected_projection(
    graph: nx.Graph,
    *,
    weight: str | None = None,
    weight_agg: str = "sum",
    distance_key: str | None = None,
    distance_from_similarity: bool = True,
) -> nx.Graph:
    """Return the documented descriptor projection.

    Nodes and node attributes are copied; direction, reciprocal arcs, parallel
    edges, and self-loops are collapsed/dropped. Native statistics are always
    computed before this projection.
    """

    return _project(
        graph,
        directed=False,
        weight=weight,
        weight_agg=weight_agg,
        distance_key=distance_key,
        distance_from_similarity=distance_from_similarity,
    )


def simple_directed_projection(
    graph: nx.Graph,
    *,
    weight: str | None = None,
    weight_agg: str = "sum",
    distance_key: str | None = None,
    distance_from_similarity: bool = True,
) -> nx.Graph:
    """Return the direction-preserving counterpart of the descriptor projection.

    Used by descriptors that are only defined on directed graphs, so that they
    see the configured edge weights under the canonical ``"weight"`` key with
    parallel arcs aggregated the same way as everywhere else.
    """

    return _project(
        graph,
        directed=True,
        weight=weight,
        weight_agg=weight_agg,
        distance_key=distance_key,
        distance_from_similarity=distance_from_similarity,
    )
