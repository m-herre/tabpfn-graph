#!/usr/bin/env python
"""Repeated nested, stratified, WL-hash-grouped evaluation on TU datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from common import duplicate_audit, estimators, structural_hashes, timed_fit_predict, write_records
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from tabpfn_graph import GraphClassifier, GraphFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--root", type=Path, default=Path("data/TU"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/tu.json"))
    parser.add_argument("--presets", nargs="+", default=["fast", "balanced", "comprehensive"])
    parser.add_argument("--models", nargs="+", default=["tabpfn", "random_forest", "xgboost"])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def parameter_grid(name: str) -> dict[str, list[object]] | list[dict[str, list[object]]]:
    if name == "random_forest":
        return {
            "estimator__max_features": ["sqrt", 0.5],
            "estimator__min_samples_leaf": [1, 2],
        }
    if name == "xgboost":
        return {"estimator__max_depth": [3, 6], "estimator__learning_rate": [0.03, 0.1]}
    return [{}]


def main() -> None:
    args = parse_args()
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root=str(args.root), name=args.dataset)
    graphs = [dataset[index] for index in range(len(dataset))]
    y = np.asarray([int(data.y.reshape(-1)[0]) for data in graphs])
    groups = structural_hashes(graphs)
    records = []

    for repeat in range(args.repeats):
        seed = args.random_state + repeat
        outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(outer.split(graphs, y, groups)):
            train_graphs = [graphs[index] for index in train_index]
            test_graphs = [graphs[index] for index in test_index]
            audit = duplicate_audit(groups[train_index], groups[test_index])
            inner = StratifiedGroupKFold(
                n_splits=args.inner_folds, shuffle=True, random_state=seed + 10_000 + fold
            )
            inner_splits = list(inner.split(train_graphs, y[train_index], groups[train_index]))

            for preset in args.presets:
                extractor = GraphFeatureExtractor(
                    features=preset, n_jobs=args.n_jobs, random_state=seed
                )
                started = time.perf_counter()
                extractor.fit(train_graphs)
                X_train = extractor.transform(train_graphs)
                X_test = extractor.transform(test_graphs)
                extraction_seconds = time.perf_counter() - started

                for name, base_estimator in estimators(
                    "classification", seed, set(args.models)
                ).items():
                    # Inner search owns its extractor, so every inner validation schema is
                    # learned only from that split's training graphs.
                    inner_model = GraphClassifier(
                        feature_extractor=GraphFeatureExtractor(
                            features=preset, n_jobs=args.n_jobs, random_state=seed
                        ),
                        estimator=clone(base_estimator),
                    )
                    started = time.perf_counter()
                    search = GridSearchCV(
                        inner_model,
                        parameter_grid(name),
                        cv=inner_splits,
                        scoring="accuracy",
                        n_jobs=1,
                        refit=False,
                    ).fit(train_graphs, y[train_index])
                    tuning_seconds = time.perf_counter() - started
                    params = search.cv_results_["params"][
                        int(np.argmax(search.cv_results_["mean_test_score"]))
                    ]
                    selected = clone(base_estimator).set_params(
                        **{key.removeprefix("estimator__"): value for key, value in params.items()}
                    )
                    prediction, fit_seconds, predict_seconds = timed_fit_predict(
                        selected, X_train, y[train_index], X_test, proba=False
                    )
                    records.append(
                        {
                            "dataset": args.dataset,
                            "split": "nested_stratified_group_cv",
                            "repeat": repeat,
                            "fold": fold,
                            "preset": preset,
                            "model": name,
                            "accuracy": accuracy_score(y[test_index], prediction),
                            "selected_params": params,
                            "n_features": X_train.shape[1],
                            "extraction_seconds": extraction_seconds,
                            "tuning_seconds": tuning_seconds,
                            "fit_seconds": fit_seconds,
                            "prediction_seconds": predict_seconds,
                            "duplicate_audit": audit,
                            "diagnostics": extractor.diagnostics_,
                        }
                    )
    write_records(records, args.output)


if __name__ == "__main__":
    main()
