"""ex013（COCO外の語彙の物体の粒度可変アノテーション）の絞り込み，停止，評価のテスト．"""
import importlib.util
import os
import sys

import numpy as np

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex013_open_vocabulary_granular"))
sys.path.insert(0, PROGRAMS)
sys.path.insert(0, DIRECTORY)
_spec = importlib.util.spec_from_file_location("ex013_train", os.path.join(DIRECTORY, "train.py"))
ex013 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex013)
ex013_model, ex013_data = sys.modules["model"], sys.modules["data"]
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def test_pruned_children_remove_role_words():
    """絞り込みにより，animal の下位語から役割を表す語（pet，herbivore）が除かれ，chordate が残ることを確認する．"""
    names = {s for s, _ in ex013_model.children("animal.n.01", True)}
    assert "chordate.n.01" in names
    assert "pet.n.01" not in names and "herbivore.n.01" not in names


def test_output_synset_softmax_stop():
    """ソフトマックスが τ 未満の段階と，候補が1つで親より類似度が低い段階で止まることを確認する．"""
    chain = [{"synset": "animal.n.01", "soft": 0.5, "cos": 0.1, "n_candidates": 6},
             {"synset": "chordate.n.01", "soft": 0.8, "cos": 0.12, "n_candidates": 7},
             {"synset": "vertebrate.n.01", "soft": 1.0, "cos": 0.05, "n_candidates": 1}]
    assert ex013.output_synset(chain, True, 0.7) == ("chordate.n.01", 1)
    assert ex013.output_synset(chain, True, 0.9) == ("animal.n.01", 0)
    assert ex013.output_synset(chain, False, 0.0) == ("vertebrate.n.01", 2)


def test_evaluate_coarse_answer_gets_partial_credit():
    """正しい粗い語は命名として正解かつ部分的な再現率を得て，GTより細かい語は適合率から除かれることを確認する．"""
    gts = [{"box": [0, 0, 10, 10], "synset": "dog.n.01", "kind": "coco"}]
    boxes = [{"box": [0, 0, 10, 10], "score": 0.9, "source": "owlv2"}]
    image = {"image_id": "1", "boxes": boxes, "gts": gts, "iou": np.array([[1.0]])}
    cfg = {"add": 1, "stop": False}
    coarse = ex013.evaluate([image], [[[{"synset": "canine.n.02", "word": "canine"}]]], cfg, 0.0)
    assert coarse["hP"] == 1.0 and 0 < coarse["hR"] < 1 and coarse["coco_named"] == 1.0 and coarse["coco_fine"] == 0.0
    finer = ex013.evaluate([image], [[[{"synset": "puppy.n.01", "word": "puppy"}]]], cfg, 0.0)
    assert finer["hP"] == 1.0 and finer["hR"] == 1.0 and finer["coco_fine"] == 1.0
