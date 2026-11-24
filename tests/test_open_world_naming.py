"""ex011（COCO外の物体の命名の改善）の棄却・停止・評価のテスト．"""
import importlib.util
import os
import sys

import numpy as np

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex011_open_world_naming"))
sys.path.insert(0, PROGRAMS)
sys.path.insert(0, DIRECTORY)
_spec = importlib.util.spec_from_file_location("ex011_train", os.path.join(DIRECTORY, "train.py"))
ex011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex011)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def test_containment_only_counts_larger_boxes():
    """小さいボックスが大きいボックスに含まれる割合のみを数えることを確認する．"""
    ratio = ex011.containment(np.array([[0, 0, 100, 100], [10, 10, 20, 20]], dtype=np.float32))
    assert ratio[0] == 0 and ratio[1] == 1.0


def test_output_node_stops_at_low_softmax():
    """加えた候補では，深さ0は常に出力し，ソフトマックスが τ_s 未満となる前の段階で止まることを確認する．"""
    chain = [{"synset": "animal.n.01", "soft": 0.2, "sig": 0.5}, {"synset": "mammal.n.01", "soft": 0.8, "sig": 0.3},
             {"synset": "dog.n.01", "soft": 0.4, "sig": 0.1}]
    box = {"source": "owlv2"}
    assert ex011.output_node(box, chain, "topdown", 0.5, True) == ("mammal.n.01", 0.3)
    assert ex011.output_node(box, chain, "topdown", 0.9, True) == ("animal.n.01", 0.5)
    assert ex011.output_node(box, chain, "topdown", 0.9, False) == ("dog.n.01", 0.1)


def test_evaluate_counts_unmatched_gt_in_recall_only():
    """出力が無いGTは hR の分母のみに含まれ，正しい粗い語の出力は部分的に正解となることを確認する．"""
    gts = [{"box": [0, 0, 10, 10], "kind": "noncoco", "synset": "camel.n.01"},
           {"box": [50, 50, 60, 60], "kind": "noncoco", "synset": "camel.n.01"}]
    boxes = [{"box": [0, 0, 10, 10], "score": 0.9, "source": "owlv2"}]
    image = {"image_id": "1", "boxes": boxes, "gts": gts, "iou": np.array([[1.0, 0.0]]), "contain": np.zeros(1)}
    chains = [[[{"synset": "mammal.n.01", "word": "mammal", "soft": 1.0, "sig": 0.9}]]]
    params = {"k": 1, "o": 0.0, "gamma": 0.0, "contain": False, "tau_s": 0.0}
    metrics = ex011.evaluate([image], chains, "topdown", params, True)
    assert metrics["hP"] == 1.0 and 0 < metrics["hR"] < 0.5
    assert metrics["noncoco_detected"] == 0.5 and metrics["noncoco_named"] == 0.5
