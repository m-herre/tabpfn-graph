# License and model access

The `tabpfn-graph` source code is distributed under Apache-2.0. That license does not relicense
TabPFN model checkpoints, hosted services, benchmark datasets, or optional dependencies.

The first local `GraphClassifier()` or `GraphRegressor()` fit may download a checkpoint and may
require authentication or acceptance of terms from Prior Labs/Hugging Face. Review the checkpoint
license for the exact model version selected by your installed `tabpfn` package, especially for
commercial use. Importing `tabpfn_graph`, constructing an estimator, extracting features, and
using a custom estimator do not download a checkpoint.

Hosted TabPFN semantic-text behavior can differ from the local pandas-string transformation. A
hosted client must be supplied via `estimator=` and is governed by that service's credentials,
privacy policy, model behavior, and terms.

