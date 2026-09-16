"""Scikit-learn-compatible graph-level predictor wrappers."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d

from ._features import FeatureGroup, GraphFeatureExtractor


def _default_tabpfn(task: str, random_state: int | None) -> Any:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as exc:  # pragma: no cover - standard installs include it
        raise ImportError(
            "The default estimator requires tabpfn. Install the standard package with "
            "`pip install tabpfn-graph`, or pass estimator= to use another estimator."
        ) from exc
    estimator_class = TabPFNClassifier if task == "classification" else TabPFNRegressor
    signature = inspect.signature(estimator_class)
    kwargs: dict[str, Any] = {}
    if "random_state" in signature.parameters:
        kwargs["random_state"] = random_state
    if "inference_config" in signature.parameters:
        # TabPFN performs its local string expansion only for pandas string dtype.
        kwargs["inference_config"] = {"TRANSFORM_TEXT": True}
    return estimator_class(**kwargs)


def _fit_with_optional_weight(
    estimator: Any, features: Any, y: np.ndarray, sample_weight: Any
) -> Any:
    if sample_weight is None:
        return estimator.fit(features, y)
    try:
        signature = inspect.signature(estimator.fit)
        supports_weight = "sample_weight" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        supports_weight = True
    if not supports_weight:
        raise TypeError(f"{type(estimator).__name__}.fit does not accept sample_weight")
    return estimator.fit(features, y, sample_weight=sample_weight)


class _BaseGraphPredictor(BaseEstimator):
    _task: str

    def __init__(
        self,
        estimator: Any = None,
        *,
        feature_extractor: GraphFeatureExtractor | None = None,
        features: str | tuple[FeatureGroup, ...] = "balanced",
        edge_weight: str | None = None,
        edge_weight_semantics: str = "similarity",
        undefined: str = "zero",
        prune_uninformative: bool = False,
        n_jobs: int | None = 1,
        random_state: int | None = 0,
        memory: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.feature_extractor = feature_extractor
        self.features = features
        self.edge_weight = edge_weight
        self.edge_weight_semantics = edge_weight_semantics
        self.undefined = undefined
        self.prune_uninformative = prune_uninformative
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.memory = memory

    def _make_extractor(self) -> GraphFeatureExtractor:
        if self.feature_extractor is not None:
            return clone(self.feature_extractor)
        return GraphFeatureExtractor(
            features=self.features,
            edge_weight=self.edge_weight,
            edge_weight_semantics=self.edge_weight_semantics,  # type: ignore[arg-type]
            undefined=self.undefined,  # type: ignore[arg-type]
            prune_uninformative=self.prune_uninformative,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            memory=self.memory,
        )

    def fit(self, graphs: Any, y: Any, sample_weight: Any = None) -> _BaseGraphPredictor:
        target = column_or_1d(y, warn=True)
        if self._task == "classification":
            check_classification_targets(target)
        elif not np.issubdtype(np.asarray(target).dtype, np.number):
            raise ValueError("regression targets must be numeric")
        if sample_weight is not None and len(sample_weight) != len(target):
            raise ValueError("sample_weight and y must have the same length")
        self.feature_extractor_ = self._make_extractor()
        table = self.feature_extractor_.fit_transform(graphs, target)
        if len(table) != len(target):
            raise ValueError(f"received {len(table)} graphs but {len(target)} target values")
        self.estimator_ = (
            clone(self.estimator)
            if self.estimator is not None
            else _default_tabpfn(self._task, self.random_state)
        )
        _fit_with_optional_weight(self.estimator_, table, target, sample_weight)
        self.n_features_in_ = table.shape[1]
        self.feature_names_in_ = np.asarray(table.columns, dtype=object)
        if self._task == "classification":
            self.classes_ = np.asarray(getattr(self.estimator_, "classes_", np.unique(target)))
        return self

    def transform(self, graphs: Any) -> Any:
        check_is_fitted(self, "estimator_")
        return self.feature_extractor_.transform(graphs)

    def predict(self, graphs: Any) -> np.ndarray:
        return (
            np.asarray(self.estimator_.predict(self.transform(graphs)))
            if hasattr(self, "estimator_")
            else self._not_fitted()
        )

    def _not_fitted(self) -> Any:
        check_is_fitted(self, "estimator_")
        raise AssertionError("unreachable")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_extractor_")
        return self.feature_extractor_.get_feature_names_out(input_features)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimator_")


class GraphClassifier(ClassifierMixin, _BaseGraphPredictor):
    """Graph-level binary/multiclass classifier.

    ``estimator=None`` lazily constructs a local ``TabPFNClassifier`` with raw
    text transformation enabled. Passing a custom estimator leaves the feature
    DataFrame untouched, including categorical and text columns.
    """

    _task = "classification"

    def predict_proba(self, graphs: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                f"{type(self.estimator_).__name__} does not implement predict_proba"
            )
        return np.asarray(self.estimator_.predict_proba(self.transform(graphs)))

    def decision_function(self, graphs: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError(
                f"{type(self.estimator_).__name__} does not implement decision_function"
            )
        return np.asarray(self.estimator_.decision_function(self.transform(graphs)))

    def score(self, graphs: Any, y: Any, sample_weight: Any = None) -> float:
        return float(accuracy_score(y, self.predict(graphs), sample_weight=sample_weight))


class GraphRegressor(RegressorMixin, _BaseGraphPredictor):
    """Single-target graph-level regressor."""

    _task = "regression"

    def score(self, graphs: Any, y: Any, sample_weight: Any = None) -> float:
        return float(r2_score(y, self.predict(graphs), sample_weight=sample_weight))
