# Reproducible evaluation

Do not use one favorable random split. The scripts under `benchmarks/` emit JSON records and
separate feature extraction, fitting, and prediction wall time.

## Small TU classification

`benchmark_tu.py` downloads a requested TU dataset through PyG. Because most TU datasets do not
provide an official split, it uses repeated nested stratified group cross-validation. The outer
and inner folds group identical NetworkX WL graph hashes so exact/WL-equivalent duplicate
candidates cannot cross a fold. Hyperparameter selection happens only in each outer training
fold. The script reports the duplicate-candidate audit alongside scores.

```bash
python benchmarks/benchmark_tu.py --dataset MUTAG --presets fast balanced comprehensive
```

## OGB classification and regression

`benchmark_ogb.py` uses OGB's official split and `Evaluator` without reshuffling:

```bash
python benchmarks/benchmark_ogb.py --dataset ogbg-molhiv --presets fast balanced
python benchmarks/benchmark_ogb.py --dataset ogbg-molesol --presets fast balanced
```

`ogbg-molhiv` is the classification example. `ogbg-molesol` is single-target regression with
its official split and RMSE evaluator. Other single-target OGB graph-property datasets may work,
but multilabel/multi-output tasks are outside v0.1.

Each preset creates one feature matrix that all estimators share. The comparison contains local
TabPFN, RandomForest, and XGBoost when installed. Feature fitting uses training graphs only;
validation and test tables reuse that schema. Train/test duplicate WL-hash candidates are
reported. A WL collision is only an audit candidate, not proof of isomorphism; investigate
candidates with an attribute-aware isomorphism check when conclusions depend on them.

## Reporting checklist

- Publish dataset version, package lockfile, hardware, seeds, and complete command. Record the
  NetworkX version: WL hashes are not comparable across NetworkX feature releases.
- Record `extractor.diagnostics_` alongside scores. It states whether edge weights were
  ignored, whether the batch was mixed-directed, how heterogeneous the graph sizes were, and
  how much of the WL vocabulary was pruned — each of which changes what the table means.
- Report every requested repetition/fold, not only the best one.
- Keep extraction, estimator fit, and prediction timings separate.
- Report `fast`/`balanced`/`comprehensive` ablations using the same splits, and state the
  `edge_weight`, `wl_node_attributes`, `laplacian`, and `undefined` settings: each changes the
  descriptors themselves, not just how many there are.
- When `allow_approximate=True`, report `approx__active` counts. Approximate descriptors are
  estimators, and `paths__diameter` becomes a lower bound.
- Include failed preflight/timeout/OOM runs rather than silently dropping them.
- Preserve official/domain-aware splits. If none exist, use nested group-aware CV.
- Audit exact hashes, WL-equivalent candidates, and confirmed isomorphisms across folds.

