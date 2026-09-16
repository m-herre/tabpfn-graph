from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tabpfn_graph import GraphFeatureExtractor
from tabpfn_graph._adapters import simple_directed_projection
from tabpfn_graph._features import column_report


@pytest.mark.parametrize(
    ("graph", "nodes", "edges", "components", "isolates"),
    [
        (nx.path_graph(5), 5, 4, 1, 0),
        (nx.cycle_graph(5), 5, 5, 1, 0),
        (nx.star_graph(4), 5, 4, 1, 0),
        (nx.complete_graph(5), 5, 10, 1, 0),
        (nx.disjoint_union(nx.path_graph(3), nx.path_graph(2)), 5, 3, 2, 0),
        (nx.Graph(), 0, 0, 0, 0),
        (nx.empty_graph(1), 1, 0, 1, 1),
    ],
)
def test_exact_basic_features(graph, nodes, edges, components, isolates):
    row = GraphFeatureExtractor(features="fast").fit_transform([graph]).iloc[0]
    assert row["basic__n_nodes"] == nodes
    assert row["basic__n_edges_native"] == edges
    assert row["basic__components"] == components
    assert row["basic__isolates"] == isolates


def test_directed_and_multigraph_native_semantics():
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(0, 1), (0, 1), (1, 0), (1, 1)])
    row = GraphFeatureExtractor(features="fast").fit_transform([graph]).iloc[0]
    assert row["basic__is_directed"] == 1
    assert row["basic__is_multigraph"] == 1
    assert row["basic__n_edges_native"] == 4
    assert row["basic__n_edges_projected"] == 1
    assert row["basic__self_loops"] == 1
    assert row["basic__reciprocal_pairs"] == 1
    assert row["basic__parallel_edge_excess"] == 1


def test_exact_balanced_and_comprehensive_values():
    path = GraphFeatureExtractor(features="balanced").fit_transform([nx.path_graph(4)]).iloc[0]
    assert path["basic__degree_native__mean"] == 1.5
    assert path["basic__degree_native__max"] == 2
    assert path["local_profile__clustering__mean"] == 0
    assert path["local_profile__core_number__mean"] == 1

    clique = (
        GraphFeatureExtractor(features="comprehensive")
        .fit_transform([nx.complete_graph(4)])
        .iloc[0]
    )
    assert clique["motifs__triangles"] == 4
    assert clique["motifs__transitivity"] == 1
    assert clique["paths__diameter"] == 1
    assert clique["paths__lcc_diameter"] == 1
    assert clique["spectral__zero_eigenvalues"] == 1
    # The default normalized Laplacian of K_n has spectrum {0, n/(n-1) x (n-1)}.
    assert clique["spectral__algebraic_connectivity"] == pytest.approx(4 / 3)

    combinatorial = (
        GraphFeatureExtractor(features="comprehensive", laplacian="combinatorial")
        .fit_transform([nx.complete_graph(4)])
        .iloc[0]
    )
    assert combinatorial["spectral__algebraic_connectivity"] == pytest.approx(4)


def test_relabeling_and_order_invariance():
    graph = nx.Graph()
    graph.add_edges_from([(10, 20), (20, 30), (30, 10), (30, 40)])
    relabeled = nx.relabel_nodes(graph, {10: "z", 20: "a", 30: "q", 40: "x"})
    relabeled = nx.Graph(list(reversed(list(relabeled.edges()))))
    extractor = GraphFeatureExtractor().fit([graph])
    result = extractor.transform([graph, relabeled])
    assert_frame_equal(
        result.iloc[[0]].reset_index(drop=True), result.iloc[[1]].reset_index(drop=True)
    )


def test_identifiers_are_opt_in_and_break_relabeling_invariance():
    graph = nx.path_graph(["a", "b", "c"])
    relabeled = nx.relabel_nodes(graph, {"a": 10, "b": 20, "c": 30})
    extractor = GraphFeatureExtractor(features="fast", include_node_ids=True).fit([graph])
    result = extractor.transform([graph, relabeled])
    assert result.loc[0, "identifiers__node_ids"] != result.loc[1, "identifiers__node_ids"]


def _attributed_graph(category: str, unseen: bool = False) -> nx.Graph:
    graph = nx.path_graph(3)
    graph.nodes[0].update(value=1.0, vector=[1.0, 2.0], kind=category, note="zeta")
    graph.nodes[1].update(value=3.0, vector=[3.0, 4.0], kind=category, note="alpha")
    graph.nodes[2].update(value=None, vector=None, kind="new" if unseen else category, note="beta")
    graph.edges[0, 1]["weight"] = 2.0
    graph.graph.update(source=category, description=f"description {category}")
    return graph


def test_attribute_schema_missing_vectors_categories_text_and_unseen():
    train = [_attributed_graph("a"), _attributed_graph("b")]
    test = _attributed_graph("c", unseen=True)
    extractor = GraphFeatureExtractor(
        features="fast",
        text_attributes=("node.note", "graph.description"),
        text_max_items=2,
        text_max_chars=20,
    ).fit(train)
    before = tuple(extractor.get_feature_names_out())
    table = extractor.transform([test])
    assert tuple(table.columns) == before
    assert table.loc[0, "node_attr__value__missing_count"] == 1
    assert table.loc[0, "node_attr__value__sum"] == 4
    assert table.loc[0, "node_attr__vector__dim_001__mean"] == 3
    assert table.loc[0, "node_attr__kind__other__count"] == 3
    assert table.loc[0, "node_attr__note__document"] == "alpha\nbeta"
    assert table.loc[0, "node_attr__note__truncated_count"] == 1
    assert isinstance(table["node_attr__note__document"].dtype, pd.StringDtype)
    assert isinstance(table["graph_attr__source"].dtype, pd.CategoricalDtype)
    assert table.loc[0, "graph_attr__source"] == "__other__"


def test_training_only_histogram_and_wl_schema():
    train = [nx.path_graph(4), nx.cycle_graph(4)]
    extractor = GraphFeatureExtractor(n_bins=4).fit(train)
    names = tuple(extractor.get_feature_names_out())
    transformed = extractor.transform([nx.complete_graph(7)])
    assert tuple(transformed.columns) == names
    assert transformed.filter(like="__bin_").shape[1] > 0
    assert transformed.loc[0, "wl__other_count"] > 0


def test_empty_transform_has_fitted_schema():
    extractor = GraphFeatureExtractor().fit([nx.path_graph(3)])
    result = extractor.transform([])
    assert result.empty
    assert tuple(result.columns) == tuple(extractor.get_feature_names_out())
    assert all(dtype == np.dtype(float) for dtype in result.dtypes)


def test_parallel_output_is_deterministic():
    graphs = [nx.gnp_random_graph(20, 0.2, seed=index) for index in range(8)]
    serial = GraphFeatureExtractor(n_jobs=1, random_state=7).fit_transform(graphs)
    parallel = GraphFeatureExtractor(n_jobs=2, random_state=7).fit_transform(graphs)
    assert_frame_equal(serial, parallel)


def test_cache_invalidates_after_graph_mutation(tmp_path):
    graph = nx.path_graph(3)
    extractor = GraphFeatureExtractor(features="fast", memory=tmp_path).fit([graph])
    first = extractor.transform([graph])
    graph.add_edge(0, 2)
    second = extractor.transform([graph])
    assert first.loc[0, "basic__n_edges_native"] == 2
    assert second.loc[0, "basic__n_edges_native"] == 3


def test_comprehensive_preflight_and_seeded_approximation():
    graph = nx.path_graph(8)
    with pytest.raises(ValueError, match="max_exact_nodes"):
        GraphFeatureExtractor(features="comprehensive", max_exact_nodes=5).fit([graph])
    table = GraphFeatureExtractor(
        features="comprehensive",
        max_exact_nodes=5,
        allow_approximate=True,
        approximation_pivots=5,
        random_state=3,
    ).fit_transform([graph])
    assert "spectral__netlsd_t_1" in table
    assert table.loc[0, "approx__active"] == 1
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()


def test_approximate_descriptors_estimate_the_whole_graph():
    """Sampling estimates the full-graph quantity instead of describing a subgraph."""
    graph = nx.barbell_graph(40, 20)
    exact = GraphFeatureExtractor(features=("motifs", "paths"), max_exact_nodes=500).fit_transform(
        [graph]
    )
    approximate = GraphFeatureExtractor(
        features=("motifs", "paths"),
        max_exact_nodes=10,
        allow_approximate=True,
        approximation_pivots=60,
        random_state=0,
    ).fit_transform([graph])
    # Triangle totals are rescaled to the full graph, not counted on a subgraph.
    assert approximate.loc[0, "motifs__triangles"] == pytest.approx(
        exact.loc[0, "motifs__triangles"], rel=0.35
    )
    # Sampled sources give a diameter lower bound on the true graph.
    assert 0 < approximate.loc[0, "paths__diameter"] <= exact.loc[0, "paths__diameter"]


class FakePyGData:
    __module__ = "torch_geometric.data.data"

    def __init__(self):
        self.edge_index = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.num_nodes = 3

    def node_attrs(self):
        return []

    def edge_attrs(self):
        return []

    def keys(self):
        return []


def test_networkx_pyg_adapter_equivalence_and_mixed_rejection():
    nx_table = GraphFeatureExtractor(features="fast").fit_transform([nx.path_graph(3)])
    pyg_table = GraphFeatureExtractor(features="fast").fit_transform([FakePyGData()])
    assert_frame_equal(nx_table, pyg_table)
    with pytest.raises(TypeError, match="homogeneous"):
        GraphFeatureExtractor().fit([nx.path_graph(3), FakePyGData()])


def test_edge_keys_are_opt_in():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key="private-key")
    plain = GraphFeatureExtractor(features="fast").fit_transform([graph])
    assert not any("private-key" in str(value) for value in plain.iloc[0])
    included = GraphFeatureExtractor(features="fast", include_edge_keys=True).fit_transform([graph])
    assert included.loc[0, "identifiers__edge_keys"] == "private-key"


def test_graph_mutation_does_not_happen_during_extraction():
    graph = _attributed_graph("a")
    original = copy.deepcopy(graph)
    GraphFeatureExtractor(features="comprehensive").fit_transform([graph])
    assert nx.utils.graphs_equal(graph, original)


def test_mixed_directedness_keeps_one_schema_without_missing_values():
    directed = nx.DiGraph([(0, 1), (1, 2)])
    undirected = nx.Graph([(0, 1), (1, 2)])
    for batch in ([directed, undirected], [undirected, directed]):
        table = GraphFeatureExtractor(features="fast").fit_transform(batch)
        assert not table.select_dtypes("number").isna().to_numpy().any()
        assert "basic__in_degree_native__sum" in table.columns
    # An undirected graph is described as its own symmetrization.
    row = GraphFeatureExtractor(features="fast").fit_transform([directed, undirected]).iloc[1]
    assert row["basic__in_degree_native__sum"] == row["basic__out_degree_native__sum"] == 4
    assert row["basic__reciprocity"] == 1


def test_directed_structure_is_described_beyond_the_symmetrization():
    cycle = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    chain = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    table = GraphFeatureExtractor(features=("basic",)).fit_transform([cycle, chain])
    assert table.loc[0, "basic__strongly_connected_components"] == 1
    assert table.loc[1, "basic__strongly_connected_components"] == 3
    assert table.loc[0, "basic__is_dag"] == 0
    assert table.loc[1, "basic__is_dag"] == 1
    assert table.loc[0, "basic__reciprocity"] == 0


def test_wl_uses_node_and_edge_labels():
    plain = nx.path_graph(5)
    labeled = nx.path_graph(5)
    for node in plain:
        plain.nodes[node]["atom"] = "C"
    for node in labeled:
        labeled.nodes[node]["atom"] = "C" if node % 2 else "N"
    table = GraphFeatureExtractor(features=("attributes", "wl")).fit_transform([plain, labeled])
    wl = table.filter(like="wl__")
    assert not wl.iloc[0].equals(wl.iloc[1])

    unlabeled = GraphFeatureExtractor(
        features=("attributes", "wl"), wl_node_attributes=None
    ).fit_transform([plain, labeled])
    assert unlabeled.filter(like="wl__").iloc[0].equals(unlabeled.filter(like="wl__").iloc[1])


def test_wl_edge_labels_are_permutation_invariant_for_multigraphs():
    def build(order):
        graph = nx.MultiGraph()
        for u, v, bond in order:
            graph.add_edge(u, v, bond=bond)
        return graph

    first = build([(0, 1, "single"), (0, 1, "double"), (1, 2, "single")])
    second = build([(1, 0, "double"), (2, 1, "single"), (1, 0, "single")])
    table = (
        GraphFeatureExtractor(features=("attributes", "wl")).fit([first]).transform([first, second])
    )
    assert_frame_equal(
        table.iloc[[0]].reset_index(drop=True), table.iloc[[1]].reset_index(drop=True)
    )


def test_wl_vocabulary_width_is_bounded():
    graphs = [nx.gnm_random_graph(30, 60, seed=index) for index in range(60)]
    unbounded = GraphFeatureExtractor(
        features=("wl",), wl_min_graph_count=1, wl_max_features=None
    ).fit(graphs)
    bounded = GraphFeatureExtractor(features=("wl",), wl_max_features=64).fit(graphs)
    assert len(bounded.get_feature_names_out()) < len(unbounded.get_feature_names_out())
    assert sum(name.startswith("wl__") for name in bounded.get_feature_names_out()) <= 65
    assert bounded.diagnostics_["wl_vocabulary"]["kept"] == 64


def test_wl_schema_refuses_a_different_networkx_feature_release(monkeypatch):
    extractor = GraphFeatureExtractor(features=("wl",)).fit([nx.path_graph(4), nx.cycle_graph(4)])
    monkeypatch.setattr("tabpfn_graph._features._networkx_version", lambda: (99, 0))
    with pytest.raises(ValueError, match="WL subtree hashes are not stable"):
        extractor.transform([nx.path_graph(4)])


def test_edge_weights_reach_structural_descriptors():
    balanced = nx.Graph()
    balanced.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    skewed = nx.Graph()
    skewed.add_weighted_edges_from([(0, 1, 99.0), (1, 2, 0.01), (2, 0, 0.01)])
    groups = ("basic", "local_profile", "centrality", "paths", "spectral")

    ignored = GraphFeatureExtractor(features=groups).fit_transform([balanced, skewed])
    assert ignored.iloc[0].equals(ignored.iloc[1])

    weighted = GraphFeatureExtractor(features=groups, edge_weight="weight").fit_transform(
        [balanced, skewed]
    )
    assert not weighted.iloc[0].equals(weighted.iloc[1])
    assert weighted.loc[1, "local_profile__strength__max"] == pytest.approx(99.01)
    # Similarity semantics: a strong tie is a short edge.
    assert weighted.loc[1, "paths__shortest__min"] == pytest.approx(1 / 99)


@pytest.mark.parametrize("semantics", ["similarity", "distance"])
def test_negative_weights_are_rejected_under_both_semantics(semantics):
    """A negative weight has no traversal cost; silently mapping it to an
    infinite one would delete the edge from path and betweenness descriptors
    while leaving it in the degree and clustering descriptors."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1, {"weight": -2.0}), (1, 2, {"weight": 1.0})])
    with pytest.raises(ValueError, match="negative value"):
        GraphFeatureExtractor(
            features=("paths",), edge_weight="weight", edge_weight_semantics=semantics
        ).fit([graph])


def test_negative_weights_are_rejected_at_transform_time():
    good = nx.Graph()
    good.add_edges_from([(0, 1, {"weight": 1.0})])
    bad = nx.Graph()
    bad.add_edges_from([(0, 1, {"weight": -1.0})])
    extractor = GraphFeatureExtractor(
        features=("paths",), edge_weight="weight", edge_weight_semantics="distance"
    ).fit([good])
    with pytest.raises(ValueError, match="negative value"):
        extractor.transform([bad])


def test_zero_distance_weight_stays_traversable():
    """Zero distance is free traversal, not an unreachable edge."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1, {"weight": 0.0}), (1, 2, {"weight": 2.0})])
    table = GraphFeatureExtractor(
        features=("paths",), edge_weight="weight", edge_weight_semantics="distance"
    ).fit_transform([graph])
    assert table.loc[0, "paths__shortest__count"] == 6
    assert table.loc[0, "paths__diameter"] == 2
    assert np.isfinite(table.select_dtypes("number").to_numpy()).all()


def test_directed_pagerank_honours_a_non_default_weight_attribute_name():
    """Directed PageRank runs on a projection, so it must read the configured
    attribute name rather than the canonical 'weight' key of the projection."""
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0), (1, 2)]
    values = [50.0, 1.0, 1.0, 50.0, 1.0, 1.0]
    named = nx.DiGraph()
    named.add_edges_from([(u, v, {"weight": w}) for (u, v), w in zip(edges, values, strict=True)])
    renamed = nx.DiGraph()
    renamed.add_edges_from([(u, v, {"w": w}) for (u, v), w in zip(edges, values, strict=True)])
    unweighted = nx.DiGraph()
    unweighted.add_edges_from(edges)

    def pagerank_std(graph, name):
        table = GraphFeatureExtractor(features=("centrality",), edge_weight=name).fit_transform(
            [graph]
        )
        return float(table.loc[0, "centrality__directed_pagerank__std"])

    assert pagerank_std(named, "weight") == pytest.approx(pagerank_std(renamed, "w"))
    assert pagerank_std(renamed, "w") != pytest.approx(pagerank_std(unweighted, None))


def test_directed_projection_aggregates_parallel_arcs():
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(0, 1, {"w": 2.0}), (0, 1, {"w": 3.0}), (1, 0, {"w": 1.0})])
    projected = simple_directed_projection(graph, weight="w", weight_agg="sum")
    assert projected.is_directed()
    assert projected[0][1]["weight"] == 5.0
    assert projected[1][0]["weight"] == 1.0


def test_quantile_bins_are_deduplicated():
    trees = [nx.random_labeled_tree(20, seed=index) for index in range(50)]
    extractor = GraphFeatureExtractor(features=("local_profile",), n_bins=8).fit(trees)
    edges = extractor.profile_bin_edges_["degree"]
    assert len(edges) == len(set(edges))
    table = extractor.transform(trees)
    bins = [name for name in table.columns if "__degree__bin_" in name]
    assert bins and all(table[name].abs().max() > 0 for name in bins)


def test_clustering_and_core_are_not_duplicated_across_groups():
    """motifs used to restate two quantities local_profile already reports."""
    graphs = [nx.gnp_random_graph(30, 0.2, seed=index) for index in range(10)]
    table = GraphFeatureExtractor(features="comprehensive").fit_transform(graphs)
    assert not table.columns.str.startswith("motifs__average_clustering").any()
    assert not table.columns.str.startswith("motifs__core_number").any()
    assert "local_profile__clustering__mean" in table.columns
    assert "local_profile__core_number__max" in table.columns
    motifs_only = GraphFeatureExtractor(features=("motifs",)).fit_transform(graphs)
    assert set(motifs_only.columns).isdisjoint(
        {"local_profile__clustering__mean", "local_profile__core_number__max"}
    )


def test_pruning_removes_constant_and_duplicated_columns():
    graphs = [nx.gnp_random_graph(30, 0.2, seed=index) for index in range(10)]
    full = GraphFeatureExtractor(features="comprehensive").fit_transform(graphs)
    extractor = GraphFeatureExtractor(features="comprehensive", prune_uninformative=True)
    pruned = extractor.fit_transform(graphs)
    assert pruned.shape[1] < full.shape[1]
    assert extractor.diagnostics_["pruned"]["n_removed"] == full.shape[1] - pruned.shape[1]
    assert column_report(pruned)["n_constant_columns"] == 0
    assert column_report(pruned)["n_duplicate_columns"] == 0
    # The pruned schema is still stable across transform calls.
    assert list(extractor.transform(graphs[:3]).columns) == list(pruned.columns)


def test_bipartite_graphs_get_square_clustering():
    graphs = [nx.complete_bipartite_graph(4, 3), nx.complete_bipartite_graph(2, 6)]
    with pytest.warns(UserWarning, match="bipartite"):
        table = GraphFeatureExtractor(features=("motifs",)).fit_transform(graphs)
    assert (table["motifs__triangles"] == 0).all()
    assert table["motifs__square_clustering__mean"].gt(0).all()


def test_largest_component_paths_separate_disconnected_from_trivial():
    singleton = nx.empty_graph(1)
    two_cliques = nx.disjoint_union(nx.complete_graph(50), nx.complete_graph(50))
    table = GraphFeatureExtractor(features=("paths",)).fit_transform([singleton, two_cliques])
    # Plain diameter: defined for the connected singleton, undefined for the split graph.
    assert (table["paths__diameter__valid"] == [1, 0]).all()
    # Largest-component diameter is defined for both and tells them apart.
    assert (table["paths__lcc_diameter__valid"] == [1, 1]).all()
    assert table.loc[0, "paths__lcc_diameter"] == 0
    assert table.loc[1, "paths__lcc_diameter"] == 1
    assert table.loc[0, "paths__lcc_fraction"] == 1
    assert table.loc[1, "paths__lcc_fraction"] == pytest.approx(0.5)


def test_undefined_can_be_encoded_as_nan():
    graphs = [nx.empty_graph(2), nx.complete_graph(4)]
    zeros = GraphFeatureExtractor(features=("centrality",), undefined="zero").fit_transform(graphs)
    nans = GraphFeatureExtractor(features=("centrality",), undefined="nan").fit_transform(graphs)
    assert zeros.loc[0, "centrality__assortativity"] == 0
    assert np.isnan(nans.loc[0, "centrality__assortativity"])
    assert nans.loc[0, "centrality__assortativity__valid"] == 0


def test_fit_warns_about_ignored_edge_weights_and_records_diagnostics():
    graphs = []
    for index in range(4):
        graph = nx.cycle_graph(5)
        nx.set_edge_attributes(graph, {edge: float(index + 1) for edge in graph.edges}, "weight")
        graphs.append(graph)
    with pytest.warns(UserWarning, match="edge_weight=None"):
        extractor = GraphFeatureExtractor(features=("basic", "attributes")).fit(graphs)
    assert extractor.diagnostics_["numeric_edge_attributes"] == ["weight"]
    assert extractor.diagnostics_["n_graphs"] == 4
    assert extractor.diagnostics_["networkx_version"]


def test_fit_transform_reports_constant_and_duplicate_columns():
    graphs = [nx.path_graph(4 + index) for index in range(6)]
    extractor = GraphFeatureExtractor(features=("basic",))
    extractor.fit_transform(graphs)
    report = extractor.diagnostics_["columns"]
    assert report["n_rows"] == 6
    assert report["n_columns"] == len(extractor.get_feature_names_out())
    assert report["n_constant_columns"] >= 1


def test_size_heterogeneity_is_flagged():
    graphs = [nx.path_graph(4) for _ in range(5)] + [nx.path_graph(900)]
    with pytest.warns(UserWarning, match="50x the median"):
        GraphFeatureExtractor(features=("basic",)).fit(graphs)
