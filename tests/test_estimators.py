from __future__ import annotations

import sys
import types

import networkx as nx
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn_graph import GraphClassifier, GraphFeatureExtractor, GraphRegressor
from tabpfn_graph._estimators import _default_tabpfn


def classification_graphs(n=12):
    return [
        nx.path_graph(3 + index % 4) if index % 2 else nx.cycle_graph(3 + index % 4)
        for index in range(n)
    ]


def test_classifier_clone_nested_params_probabilities_cv_and_grid_search():
    graphs = classification_graphs()
    y = np.asarray([index % 2 for index in range(len(graphs))])
    model = GraphClassifier(
        estimator=RandomForestClassifier(n_estimators=10, random_state=0),
        features="fast",
    )
    cloned = clone(model)
    cloned.set_params(estimator__max_depth=2)
    fitted = cloned.fit(graphs, y, sample_weight=np.ones(len(y)))
    assert fitted.predict(graphs).shape == (len(graphs),)
    assert fitted.predict_proba(graphs).shape == (len(graphs), 2)
    assert len(fitted.get_feature_names_out()) == fitted.n_features_in_
    assert cross_val_score(model, graphs, y, cv=3).shape == (3,)
    search = GridSearchCV(model, {"estimator__max_depth": [1, 2]}, cv=2).fit(graphs, y)
    assert search.best_estimator_.estimator_.max_depth in {1, 2}


def test_regressor_custom_estimator_and_score():
    graphs = [nx.path_graph(size) for size in range(2, 10)]
    y = np.asarray([graph.number_of_edges() for graph in graphs], dtype=float)
    model = GraphRegressor(
        estimator=RandomForestRegressor(n_estimators=30, random_state=0),
        features="fast",
    ).fit(graphs, y)
    assert model.predict(graphs).shape == y.shape
    assert model.score(graphs, y) > 0.9


def test_text_aware_custom_pipeline():
    graphs = classification_graphs(8)
    for index, graph in enumerate(graphs):
        for node in graph:
            graph.nodes[node]["description"] = "even graph" if index % 2 == 0 else "odd graph"
    extractor = GraphFeatureExtractor(
        features=("basic", "text"), text_attributes=("node.description",)
    )
    text_column = "node_attr__description__document"
    preprocessing = ColumnTransformer(
        [
            ("text", TfidfVectorizer(), text_column),
            ("numeric", StandardScaler(), ["basic__n_nodes", "basic__n_edges_native"]),
        ]
    )
    model = GraphClassifier(
        feature_extractor=extractor,
        estimator=make_pipeline(
            preprocessing, RandomForestClassifier(n_estimators=10, random_state=0)
        ),
    ).fit(graphs, np.arange(8) % 2)
    assert model.predict(graphs).shape == (8,)


def test_multioutput_targets_are_rejected():
    with pytest.raises(ValueError):
        GraphClassifier(estimator=RandomForestClassifier()).fit(
            classification_graphs(4), np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]])
        )


def test_unsupported_sample_weight_is_actionable():
    class NoWeightClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.repeat(self.classes_[0], len(X))

    with pytest.raises(TypeError, match="sample_weight"):
        GraphClassifier(estimator=NoWeightClassifier(), features="fast").fit(
            classification_graphs(4), [0, 1, 0, 1], sample_weight=np.ones(4)
        )


def test_default_tabpfn_construction_enables_text_without_fitting(monkeypatch):
    class FakeTabPFN:
        def __init__(self, *, random_state=None, inference_config=None):
            self.random_state = random_state
            self.inference_config = inference_config

    module = types.ModuleType("tabpfn")
    module.TabPFNClassifier = FakeTabPFN
    module.TabPFNRegressor = FakeTabPFN
    monkeypatch.setitem(sys.modules, "tabpfn", module)
    estimator = _default_tabpfn("classification", 19)
    assert estimator.random_state == 19
    assert estimator.inference_config == {"TRANSFORM_TEXT": True}
