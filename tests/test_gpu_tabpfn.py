from __future__ import annotations

import os

import networkx as nx
import pytest

from tabpfn_graph import GraphClassifier


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("TABPFN_GRAPH_RUN_GPU") != "1",
    reason="set TABPFN_GRAPH_RUN_GPU=1 to allow checkpoint access and GPU inference",
)
def test_real_local_tabpfn_gpu_end_to_end():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tabpfn = pytest.importorskip("tabpfn")
    graphs = [
        nx.path_graph(3),
        nx.cycle_graph(3),
        nx.path_graph(4),
        nx.cycle_graph(4),
        nx.path_graph(5),
        nx.cycle_graph(5),
    ]
    estimator = tabpfn.TabPFNClassifier(
        device="cuda", inference_config={"TRANSFORM_TEXT": True}, random_state=0
    )
    model = GraphClassifier(estimator=estimator, features="fast").fit(graphs, [0, 1, 0, 1, 0, 1])
    assert model.predict(graphs).shape == (len(graphs),)
