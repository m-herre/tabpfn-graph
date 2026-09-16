#!/usr/bin/env python
"""One-seed OGBG-MOLHIV sanity benchmark with local TabPFN 3.5.

This intentionally is not a leaderboard submission protocol. It uses the official
scaffold split and evaluator, no hyperparameter tuning, and one random seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from tabpfn_graph import GraphFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ogb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=["fast", "balanced"], default="balanced")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=8)
    return parser.parse_args()


def write_result(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate(evaluator: Any, truth: np.ndarray, prediction: np.ndarray) -> float:
    result = evaluator.eval({"y_true": truth.reshape(-1, 1), "y_pred": prediction.reshape(-1, 1)})
    return float(result["rocauc"])


def main() -> None:
    args = parse_args()
    started_total = time.perf_counter()

    # OGB 1.3.6 predates PyTorch's weights_only=True default. The processed file
    # was created locally from OGB's official download, so explicitly use the
    # legacy loader for this trusted artifact.
    original_torch_load = torch.load

    def trusted_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_torch_load  # type: ignore[assignment]

    import ogb
    import tabpfn
    import torch_geometric
    from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(args.root))
    split = {name: np.asarray(index).reshape(-1) for name, index in dataset.get_idx_split().items()}
    graphs = [dataset[index] for index in range(len(dataset))]
    y = np.asarray(dataset.data.y).reshape(-1)
    train_graphs = [graphs[index] for index in split["train"]]
    valid_graphs = [graphs[index] for index in split["valid"]]
    test_graphs = [graphs[index] for index in split["test"]]

    result: dict[str, Any] = {
        "protocol": {
            "dataset": "ogbg-molhiv",
            "dataset_version": getattr(dataset, "version", "unknown"),
            "split": "official scaffold",
            "metric": "ROC-AUC",
            "seed": args.seed,
            "preset": args.preset,
            "tuning": "none",
            "repetitions": 1,
            "train_graphs": len(train_graphs),
            "validation_graphs": len(valid_graphs),
            "test_graphs": len(test_graphs),
        },
        "environment": {
            "python": platform.python_version(),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "ogb": ogb.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "official_reference": {
            "source": "https://ogb.stanford.edu/docs/leader_graphprop/",
            "GIN": {"test_mean": 0.7558, "test_std": 0.0140},
            "GCN_virtual_node": {"test_mean": 0.7599, "test_std": 0.0119},
            "GIN_virtual_node": {"test_mean": 0.7707, "test_std": 0.0149},
        },
        "results": {},
    }

    extractor = GraphFeatureExtractor(
        features=args.preset,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        backend="networkx",
    )
    started = time.perf_counter()
    extractor.fit(train_graphs)
    X_train = extractor.transform(train_graphs)
    X_valid = extractor.transform(valid_graphs)
    X_test = extractor.transform(test_graphs)
    extraction_seconds = time.perf_counter() - started
    if not (tuple(X_train.columns) == tuple(X_valid.columns) == tuple(X_test.columns)):
        raise AssertionError("feature schemas differ across official splits")
    if X_train.select_dtypes(exclude="number").shape[1]:
        raise TypeError("this quick benchmark expects the OGB feature table to be fully numeric")
    result["features"] = {
        "columns": X_train.shape[1],
        "extraction_seconds": extraction_seconds,
        "schema_sha256": hashlib.sha256("\n".join(X_train.columns).encode()).hexdigest(),
    }
    write_result(result, args.output)

    evaluator = Evaluator("ogbg-molhiv")
    forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    started = time.perf_counter()
    forest.fit(X_train, y[split["train"]])
    forest_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    forest_valid = forest.predict_proba(X_valid)[:, 1]
    forest_test = forest.predict_proba(X_test)[:, 1]
    forest_predict_seconds = time.perf_counter() - started
    result["results"]["random_forest"] = {
        "validation_rocauc": evaluate(evaluator, y[split["valid"]], forest_valid),
        "test_rocauc": evaluate(evaluator, y[split["test"]], forest_test),
        "fit_seconds": forest_fit_seconds,
        "prediction_seconds": forest_predict_seconds,
    }
    write_result(result, args.output)

    checkpoint = Path.home() / ".cache/tabpfn/tabpfn-v3.5-20260909.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"expected local v3.5 checkpoint at {checkpoint}")
    classifier = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        model_path=checkpoint,
        device="cuda",
        n_estimators=args.n_estimators,
        random_state=args.seed,
        fit_mode="fit_preprocessors",
        show_progress_bar=True,
    )
    started = time.perf_counter()
    classifier.fit(X_train, y[split["train"]])
    tabpfn_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tabpfn_valid = classifier.predict_proba(X_valid)[:, 1]
    tabpfn_test = classifier.predict_proba(X_test)[:, 1]
    tabpfn_predict_seconds = time.perf_counter() - started
    result["results"]["tabpfn_3_5"] = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "n_estimators": args.n_estimators,
        "validation_rocauc": evaluate(evaluator, y[split["valid"]], tabpfn_valid),
        "test_rocauc": evaluate(evaluator, y[split["test"]], tabpfn_test),
        "fit_seconds": tabpfn_fit_seconds,
        "prediction_seconds": tabpfn_predict_seconds,
    }
    result["total_seconds"] = time.perf_counter() - started_total
    write_result(result, args.output)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
