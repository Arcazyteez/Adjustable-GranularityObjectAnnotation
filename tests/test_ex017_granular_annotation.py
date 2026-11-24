"""ex017（order_027）の粒度可変アノテーションと評価関数のテスト．"""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

EX017_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex017_coco_granular_annotation")
sys.path.insert(0, EX017_DIR)

from data import crop_box, hyponym_vocab, root_vocab  # noqa: E402
from model import granular_annotation  # noqa: E402
from train import greedy_match, metrics_func  # noqa: E402

# 他の実験のテストも data，model，train という同名のモジュールを読み込むため，読み込み後に取り除く
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.remove(EX017_DIR)


class DummyEmbedder:
    """単語ごとに固定のスコアを返す埋め込みモデル（テスト用）．"""

    def __init__(self, table, default=0.0):
        """
        Args:
            table (dict[str, float]): 単語 → スコア．
            default (float): 表に無い単語のスコア．
        """
        self.table = table
        self.default = default

    def scores(self, image_feats, words, mode="cosine"):
        """
        Args:
            image_feats (torch.Tensor): 使わない（1×D）．
            words (list[str]): 候補語彙．
            mode (str): 使わない．

        Returns:
            np.ndarray: スコア（1×V）．
        """
        return np.array([[self.table.get(w, self.default) for w in words]], dtype=np.float32)


def test_hyponym_vocab_dog_has_unique_words():
    """dog.n.01 の直接の下位語が単語の重複なく得られる．"""
    vocab = hyponym_vocab("dog.n.01")
    words = [w for w, _ in vocab]
    assert len(words) == len(set(words)) > 5
    assert hyponym_vocab(None) == []


def test_root_vocab_sizes():
    """論文と進捗報告の初期の語彙の大きさ．"""
    assert len(root_vocab("coco80")) == 80
    assert len(root_vocab("coco80_extra")) == 102
    assert len(root_vocab("wordnet_tops")) == 6


def test_granular_annotation_descends_and_stops_at_theta():
    """θ 以上の間は下位語へ進み，θ 未満で止まる（Alg. の break）．"""
    vocab = [("dog", "dog.n.01"), ("cat", "cat.n.01")]
    hypo = [w for w, _ in hyponym_vocab("dog.n.01")]
    emb = DummyEmbedder({"dog": 0.9, "cat": 0.1, hypo[0]: 0.5}, default=0.0)
    feat = torch.zeros(1, 4)
    chain = granular_annotation(emb, feat, vocab, theta=0.3)
    assert [s["word"] for s in chain] == ["dog", hypo[0]]
    assert chain[1]["depth"] == 1 and chain[1]["n_candidates"] == len(hypo)
    assert granular_annotation(emb, feat, vocab, theta=0.95) == []
    assert len(granular_annotation(emb, feat, vocab, theta=-np.inf, max_depth=0)) == 1


def test_greedy_match_respects_order_and_threshold():
    """スコアの高い予測が先にGTを取り，IoUが閾値未満の組は対応しない．"""
    iou = np.array([[0.6, 0.0], [0.9, 0.4]])
    assert greedy_match(iou, [1, 0]) == {1: 0}
    assert greedy_match(iou, [0, 1]) == {0: 0}


def test_metrics_func_label_agreement_and_theta():
    """ラベルの一致率，適合率・再現率，θ による棄却．"""
    record = {"image_id": 1, "gt_boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32),
              "gt_labels": ["dog", "cat"], "boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32),
              "top": [["dog", "cat"], ["dog", "cat"]], "scores": np.array([0.9, 0.2], np.float32),
              "det_scores": np.array([0.99, 0.98], np.float32)}
    m, counts = metrics_func([record])
    assert m["n_matched"] == 2 and m["top1_acc"] == 0.5 and m["top5_acc"] == 1.0
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["mean_iou"] == pytest.approx(1.0)
    m2, _ = metrics_func([record], theta=0.5)
    assert m2["n_pred"] == 1 and m2["precision"] == 1.0 and m2["recall"] == 0.5
    assert counts.shape == (1, 6)
    m3, _ = metrics_func([record], theta=1.0)
    assert m3["n_pred"] == 0 and m3["recall"] == 0.0


def test_crop_box_has_positive_size():
    """画像外や大きさ0のボックスでも1ピクセル以上の切り抜き画像になる．"""
    im = Image.new("RGB", (20, 10))
    assert crop_box(im, [5, 5, 5, 5]).size == (1, 1)
    assert crop_box(im, [-3, -3, 50, 50]).size == (20, 10)
