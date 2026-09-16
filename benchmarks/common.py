"""Shared benchmark utilities; benchmark scripts are not part of the package API."""

from __future__ import annotations

import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tabpfn_graph._adapters import adapt_batch, simple_undirected_projection


def structural_hashes(graphs: list[Any]) -> np.ndarray:
    converted = adapt_batch(graphs).graphs
    return np.asarray(
        [
            nx.weisfeiler_lehman_graph_hash(simple_undirected_projection(graph), iterations=3)
            if graph
            else "empty"
            for graph in converted
        ]
    )


def duplicate_audit(train_hashes: np.ndarray, test_hashes: np.ndarray) -> dict[str, int]:
    train_counts = Counter(train_hashes.tolist())
    test_counts = Counter(test_hashes.tolist())
    overlap = set(train_counts) & set(test_counts)
    return {
        "train_duplicate_hash_groups": sum(count > 1 for count in train_counts.values()),
        "test_duplicate_hash_groups": sum(count > 1 for count in test_counts.values()),
        "cross_split_hash_groups": len(overlap),
        "cross_split_train_graphs": sum(train_counts[value] for value in overlap),
        "cross_split_test_graphs": sum(test_counts[value] for value in overlap),
    }


def estimators(task: str, random_state: int, selected: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "random_forest" in selected:
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        result["random_forest"] = cls(
            n_estimators=500, min_samples_leaf=1, n_jobs=-1, random_state=random_state
        )
    if "xgboost" in selected:
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError("Install the bench extra to benchmark XGBoost") from exc
        cls = XGBClassifier if task == "classification" else XGBRegressor
        result["xgboost"] = cls(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=random_state,
        )
    if "tabpfn" in selected:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
        result["tabpfn"] = cls(
            random_state=random_state,
            inference_config={"TRANSFORM_TEXT": True},
            show_progress_bar=False,
        )
    return result


def timed_fit_predict(
    estimator: Any, X_train: Any, y_train: np.ndarray, X_test: Any, *, proba: bool
) -> tuple[np.ndarray, float, float]:
    model = clone(estimator)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        prediction = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
    else:
        prediction = model.predict(X_test)
    predict_seconds = time.perf_counter() - started
    return np.asarray(prediction), fit_seconds, predict_seconds


def write_records(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "records": records,
    }
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
