"""ex015c（経路が正解の枝から外れる段階の分析）のテスト．"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex015c_divergence_analysis"))
spec = importlib.util.spec_from_file_location("ex015c_model", os.path.join(DIRECTORY, "model.py"))
ex015c_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex015c_model)

NODES = ["physical_entity.n.01", "object.n.01", "whole.n.02", "living_thing.n.01", "organism.n.01", "animal.n.01",
         "artifact.n.01", "excavation.n.03", "ditch.n.01", "kangaroo.n.01"]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 9), (2, 6), (6, 7), (7, 8)]


def _hier():
    """
    テスト用の小さな階層（physical_entity → object → whole → {living thing … kangaroo，artifact → excavation → ditch}）を作る．

    Returns:
        types.SimpleNamespace: nodes，index，child を持つ階層．
    """
    child = torch.zeros(len(NODES), len(NODES), dtype=torch.bool)
    for a, b in EDGES:
        child[a, b] = True
    return types.SimpleNamespace(nodes=NODES, index={n: k for k, n in enumerate(NODES)}, child=child)


def test_divergence_at_top():
    """kangaroo を artifact → excavation → ditch と命名した場合，whole で外れ（深さ0），正しい子は living thing で順位2．"""
    p = np.array([1.0, 0.9, 0.85, 0.3, 0.3, 0.3, 0.55, 0.5, 0.4, 0.2])
    path = ["physical_entity.n.01", "object.n.01", "whole.n.02", "artifact.n.01", "excavation.n.03", "ditch.n.01"]
    r = ex015c_model.divergence(_hier(), p, path, "kangaroo.n.01")
    assert r["status"] == "diverged" and r["top"] and r["step"] == 3
    assert r["parent"] == "whole.n.02" and r["chosen"] == "artifact.n.01" and r["correct"] == "living_thing.n.01"
    assert r["rank"] == 2 and np.isclose(r["p_correct"], 0.3) and r["wrong_len"] == 3


def test_fine_coarse_unreachable():
    """GTまで到達すれば fine，上位語で止まれば coarse，GTが根の下に無ければ unreachable．"""
    h, p = _hier(), np.ones(len(NODES))
    base = ["physical_entity.n.01", "object.n.01", "whole.n.02", "living_thing.n.01", "organism.n.01", "animal.n.01"]
    assert ex015c_model.divergence(h, p, base + ["kangaroo.n.01"], "kangaroo.n.01")["status"] == "fine"
    assert ex015c_model.divergence(h, p, base, "kangaroo.n.01")["status"] == "coarse"
    assert ex015c_model.divergence(h, p, base, "poster.n.01")["status"] == "unreachable"


def test_step_table():
    """各段階の上位の下位語は確率の順に並ぶ．"""
    p = np.array([1.0, 0.9, 0.85, 0.3, 0.3, 0.3, 0.55, 0.5, 0.4, 0.2])
    steps = ex015c_model.step_table(_hier(), p, ["physical_entity.n.01", "object.n.01", "whole.n.02"], k=2)
    assert [w for w, _ in steps[2]["top"]] == ["artifact", "living thing"]
