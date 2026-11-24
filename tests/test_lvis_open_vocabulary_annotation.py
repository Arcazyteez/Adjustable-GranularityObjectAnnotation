"""ex015（再学習なしの粒度可変アノテーションのやり直し）のテスト．"""
import importlib.util
import os
import sys

import numpy as np
import torch

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex015_lvis_open_vocabulary_annotation"))
sys.path.insert(0, PROGRAMS)
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.insert(0, DIRECTORY)


def _load(name):
    """
    ex015のモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(f"ex015_{name}", os.path.join(DIRECTORY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


ex015_data = _load("data")
ex015_model = _load("model")
ex015_train = _load("train")
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)

VOCAB = [{"synset": "beagle.n.01"}, {"synset": "basset.n.01"}, {"synset": "hound.n.01"}, {"synset": "car.n.01"}]


def _p(row):
    return torch.tensor([row], device=ex015_model.DEVICE)


def test_backoff_climbs_from_argmax_word():
    """backoff は最も確率の高い語から，確率が τ 以上になるまで上位語へさかのぼることを確認する．"""
    h = ex015_model.Hierarchy(VOCAB)
    out = h.choose(_p([0.40, 0.35, 0.05, 0.20]), [0.3, 0.7], "backoff")
    assert h.nodes[int(out[0.3][0][0])] == "beagle.n.01"
    assert h.nodes[int(out[0.7][0][0])] == "hound.n.01"


def test_deepest_can_pick_low_probability_branch_but_backoff_does_not():
    """deepest は argmax と別の枝の深い語を選び得るが，backoff は argmax の上位語のみから選ぶことを確認する．"""
    vocab = [{"synset": "beagle.n.01"}, {"synset": "car.n.01"}]
    h = ex015_model.Hierarchy(vocab)
    p = _p([0.35, 0.65])
    assert h.nodes[int(h.choose(p, [0.3], "backoff")[0.3][0][0])] == "car.n.01"
    assert h.choose(p, [0.3], "deepest")[0.3][0][0] in (h.index["beagle.n.01"], h.index["car.n.01"])


def test_greedy_stops_when_own_word_dominates():
    """greedy は，下位語の確率の最大値が自身の語の確率以下の場合にその語で止まることを確認する．"""
    h = ex015_model.Hierarchy(VOCAB)
    out = h.choose(_p([0.20, 0.15, 0.60, 0.05]), [0.05], "greedy")
    assert h.nodes[int(out[0.05][0][0])] == "hound.n.01"
    out = h.choose(_p([0.70, 0.10, 0.10, 0.10]), [0.05, 0.95], "greedy")
    assert h.nodes[int(out[0.05][0][0])] == "beagle.n.01"
    assert h.nodes[int(out[0.95][0][0])] == "whole.n.02"  # car と hound の共通の上位語（確率1.0）


def test_greedy_respects_start():
    """greedy の起点を与えた場合，起点より上位の語を出力しないことを確認する．"""
    h = ex015_model.Hierarchy(VOCAB)
    start = torch.tensor([h.index["hound.n.01"]], device=ex015_model.DEVICE)
    out = h.choose(_p([0.10, 0.10, 0.10, 0.70]), [0.9], "greedy", start=start)
    assert h.nodes[int(out[0.9][0][0])] == "hound.n.01"


def test_vocab_variant_excludes_person_roles_only():
    """人の役割を表す語を除く場合も person.n.01 自身は残ることを確認する．"""
    vocab = [{"synset": "person.n.01", "person": False}, {"synset": "skier.n.01", "person": True},
             {"synset": "car.n.01", "person": False}]
    assert [v["synset"] for v in ex015_train.vocab_variant(vocab, True)] == ["person.n.01", "car.n.01"]
    assert len(ex015_train.vocab_variant(vocab, False)) == 3


def _image():
    """評価のテスト用の画像（YOLO11xのボックス1件，加えた領域2件，GT2件）．"""
    return {"image_id": "1", "n_yolo": 1,
            "boxes": [{"box": [0, 0, 10, 10], "source": "yolo", "score": 0.9, "label": "dog"},
                      {"box": [20, 20, 30, 30], "source": "owlv2", "score": 0.5},
                      {"box": [50, 50, 60, 60], "source": "owlv2", "score": 0.01}],
            "gts": [{"box": [0, 0, 10, 10], "synset": "beagle.n.01", "kind": "coco"},
                    {"box": [20, 20, 30, 30], "synset": "basset.n.01", "kind": "noncoco"}],
            "iou": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            "checked_synsets": frozenset({"car.n.01"}), "category_synset": {}}


def test_evaluate_counts_named_and_fine():
    """細分化（GTの語またはそれより細かい語）と命名（正しい上位語を含む）を数えることを確認する．"""
    img = _image()
    m = ex015_train.evaluate([img], [["dog.n.01", "basset.n.01", "car.n.01"]], 10, 0.0)
    assert m["coco_named"] == 1.0 and m["coco_fine"] == 0.0
    assert m["noncoco_named"] == 1.0 and m["noncoco_fine"] == 1.0
    assert m["n_false"] == 1 and m["fed_hP"] < m["hP"]


def test_evaluate_objectness_rejection():
    """物体らしさの閾値未満の加えた領域を出力しないことを確認する．"""
    img = _image()
    m = ex015_train.evaluate([img], [["dog.n.01", "basset.n.01", "car.n.01"]], 10, 0.1)
    assert m["n_false"] == 0 and m["fed_hP"] == m["hP"]
    m = ex015_train.evaluate([img], [["dog.n.01", "basset.n.01", "car.n.01"]], 0, 0.0)
    assert m["noncoco_detected"] == 0.0


def test_assemble_pads_unnamed_regions():
    """加えた領域の出力が足りない場合に最上位の語で埋めることを確認する．"""
    img = _image()
    out = ex015_train.assemble([img], [["basset.n.01"]], [["dog.n.01"]])
    assert out == [["dog.n.01", "basset.n.01", "physical_entity.n.01"]]
