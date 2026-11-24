"""ex012（進捗報告2026Bの粒度可変アノテーション）のアルゴリズムと評価指標のテスト．"""
import importlib.util
import os
import sys

import numpy as np
import torch

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex012_wordnet_granular_annotation"))
sys.path.insert(0, PROGRAMS)
sys.path.insert(0, DIRECTORY)
_spec = importlib.util.spec_from_file_location("ex012_train", os.path.join(DIRECTORY, "train.py"))
ex012 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex012)
ex012_model = sys.modules["model"]
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


class FakeEmbedder:
    """単語ごとに決めた類似度を返すテスト用の F．"""

    def __init__(self, scores):
        """
        Args:
            scores (dict[str, float]): 単語 → 類似度．
        """
        self.scores = scores

    def text_features(self, words):
        """
        各単語の「特徴ベクトル」として，類似度を1次元のベクトルで返す．

        Args:
            words (list[str]): 単語．

        Returns:
            torch.Tensor: 類似度（V×1）．
        """
        return torch.tensor([[self.scores.get(w, 0.0)] for w in words])


def test_granular_annotation_descends_until_threshold():
    """類似度が θ 以上の間だけ下位語をたどり，θ 未満で終了することを確認する．"""
    embedder = FakeEmbedder({"animal": 0.3, "chordate": 0.28, "vertebrate": 0.1})
    chain = ex012_model.granular_annotation(embedder, torch.tensor([1.0]), ["animal.n.01", "person.n.01"], 0.2)
    assert [c["word"] for c in chain] == ["animal", "chordate"]
    assert ex012_model.granular_annotation(embedder, torch.tensor([1.0]), ["animal.n.01"], 0.5) == []


def test_evaluate_label_accuracy_and_box_error():
    """ラベルの一致率，クラスを含めた適合率・再現率，ボックスのずれが正しく計算されることを確認する．"""
    gts = {"1": (np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32), ["dog", "cat"])}
    preds = {"1": [{"box": [0, 0, 10, 10], "label": "dog"}, {"box": [21, 20, 31, 30], "label": "dog"},
                   {"box": [50, 50, 60, 60], "label": "cat"}]}
    m = ex012.evaluate(preds, gts)
    assert m["n_matched"] == 2 and m["label_accuracy"] == 0.5
    assert abs(m["precision"] - 1 / 3) < 1e-9 and m["recall"] == 0.5
    assert m["coord_error_px"] == 0.25  # 2組の平均（0 と 0.5）


def test_proposed_predictions_rejects_below_theta():
    """類似度が θ 未満のボックスは出力しないことを確認する．"""
    boxes = {"1": [{"box": [0, 0, 1, 1], "score": 0.9}, {"box": [2, 2, 3, 3], "score": 0.8}]}
    sims = {"1": np.array([[0.3, 0.1], [0.05, 0.1]])}
    preds = ex012.proposed_predictions(boxes, sims, ["dog", "cat"], 0.2)
    assert [p["label"] for p in preds["1"]] == ["dog"]
