#!/usr/bin/env python
"""Benchmark single-target OGB graph property datasets on their official split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from common import duplicate_audit, estimators, structural_hashes, timed_fit_predict, write_records

from tabpfn_graph import GraphFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbg-molhiv")
    parser.add_argument("--root", type=Path, default=Path("data/ogb"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/ogb.json"))
    parser.add_argument("--presets", nargs="+", default=["fast", "balanced"])
    parser.add_argument("--models", nargs="+", default=["tabpfn", "random_forest", "xgboost"])
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name=args.dataset, root=str(args.root))
    split = dataset.get_idx_split()
    indices = {name: np.asarray(value).reshape(-1) for name, value in split.items()}
    if dataset.num_tasks != 1:
        raise ValueError("v0.1 benchmark supports only single-target OGB datasets")
    graphs = [dataset[index] for index in range(len(dataset))]
    y = np.asarray(dataset.data.y).reshape(-1)
    task = "regression" if "regression" in dataset.task_type else "classification"
    evaluator = Evaluator(args.dataset)
    train_graphs = [graphs[index] for index in indices["train"]]
    test_graphs = [graphs[index] for index in indices["test"]]
    audit = duplicate_audit(structural_hashes(train_graphs), structural_hashes(test_graphs))
    records = []

    for preset in args.presets:
        extractor = GraphFeatureExtractor(
            features=preset, n_jobs=args.n_jobs, random_state=args.random_state
        )
        started = time.perf_counter()
        extractor.fit(train_graphs)
        X_train = extractor.transform(train_graphs)
        X_valid = extractor.transform([graphs[index] for index in indices["valid"]])
        X_test = extractor.transform(test_graphs)
        extraction_seconds = time.perf_counter() - started
        # Validation extraction is deliberate: it verifies official-split schema stability and
        # is available for user-added model selection without touching the test labels.
        assert tuple(X_train.columns) == tuple(X_valid.columns) == tuple(X_test.columns)

        for name, estimator in estimators(task, args.random_state, set(args.models)).items():
            prediction, fit_seconds, predict_seconds = timed_fit_predict(
                estimator,
                X_train,
                y[indices["train"]],
                X_test,
                proba=task == "classification",
            )
            truth = y[indices["test"]]
            valid = np.isfinite(truth)
            metric = evaluator.eval(
                {
                    "y_true": truth[valid].reshape(-1, 1),
                    "y_pred": prediction[valid].reshape(-1, 1),
                }
            )
            records.append(
                {
                    "dataset": args.dataset,
                    "split": "official",
                    "preset": preset,
                    "model": name,
                    "metric": metric,
                    "n_features": X_train.shape[1],
                    "extraction_seconds": extraction_seconds,
                    "fit_seconds": fit_seconds,
                    "prediction_seconds": predict_seconds,
                    "duplicate_audit": audit,
                    "diagnostics": extractor.diagnostics_,
                }
            )
    write_records(records, args.output)


if __name__ == "__main__":
    main()
