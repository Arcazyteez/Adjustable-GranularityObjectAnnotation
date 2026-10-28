"""評価指標（programs/common/metrics.py）のテスト．"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs"))
from common.metrics import bootstrap_ci, box_iou, match_boxes, precision_recall_at_k  # noqa: E402


def test_precision_recall_at_k():
    """上位k語とGTの重なりからP@k，R@kが正しく計算されることを確認する．"""
    p, r = precision_recall_at_k(["dog", "cat", "car", "bus", "cup"], {"dog", "car", "tree"}, 5)
    assert p == 2 / 5 and r == 2 / 3
    assert precision_recall_at_k(["cat"], {"dog"}, 1) == (0.0, 0.0)


def test_box_iou():
    """同一ボックスでIoU=1，半分重なるボックスでIoU=1/3となることを確認する．"""
    iou = box_iou(np.array([[0, 0, 2, 2]]), np.array([[0, 0, 2, 2], [1, 0, 3, 2], [5, 5, 6, 6]]))
    np.testing.assert_allclose(iou[0], [1.0, 1 / 3, 0.0])


def test_match_boxes_one_to_one():
    """1つのGTに複数の予測が重なっても1対1にしか対応付けられないことを確認する．"""
    preds = np.array([[0, 0, 10, 10], [0, 0, 9, 10], [20, 20, 30, 30]])
    gts = np.array([[0, 0, 10, 10]])
    matches = match_boxes(preds, gts, 0.5)
    assert len(matches) == 1 and matches[0][:2] == (0, 0)


def test_bootstrap_ci_contains_mean():
    """ブートストラップ信頼区間が標本平均を含むことを確認する．"""
    mean, lo, hi = bootstrap_ci(np.random.default_rng(0).random(200), 500, seed=0)
    assert lo <= mean <= hi
