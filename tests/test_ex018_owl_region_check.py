"""ex018（order_028）のOWLとYOLOの領域の重なりの判定と集計のテスト．"""
import os
import sys

import numpy as np

EX018_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex018_owl_region_check")
sys.path.insert(0, EX018_DIR)

from model import box_iou, covered  # noqa: E402
from train import budget_threshold, image_record, metrics_func  # noqa: E402

# 他の実験のテストも data，model，train という同名のモジュールを読み込むため，読み込み後に取り除く
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.remove(EX018_DIR)


def test_covered_and_empty_inputs():
    """IoU 0.5 以上の重なりの判定と，空の入力の形状．"""
    a = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    b = np.array([[0, 0, 10, 12]], np.float32)
    assert covered(a, b).tolist() == [True, False]
    assert covered(a, np.zeros((0, 4))).tolist() == [False, False]
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 1)


def test_budget_threshold_matches_mean_count():
    """閾値以上のボックス数の平均が予算に一致する．"""
    dets = {1: {"scores": [0.9, 0.5, 0.1]}, 2: {"scores": [0.8, 0.2]}}
    thr = budget_threshold(dets, [1, 2], 2)
    assert sum((np.array(d["scores"]) >= thr).sum() for d in dets.values()) == 4
    assert budget_threshold(dets, [1, 2], 10) == 0.1


def test_metrics_four_way_split():
    """GTの4分割（両方，YOLOのみ，OWLのみ，どちらも無し）と条件付きの検出率．"""
    gt = (np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]], np.float32),
          ["coco", "coco", "noncoco", "noncoco"], ["dog", "cat", "hat", "shoe"])
    yolo = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [90, 90, 95, 95]], np.float32)
    owl = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], np.float32)
    m = metrics_func([image_record(gt, yolo, owl)])
    assert (m["coco"]["both"], m["coco"]["yolo_only"]) == (1, 1)
    assert (m["noncoco"]["owl_only"], m["noncoco"]["neither"]) == (1, 1)
    assert m["coco"]["owl_given_yolo_hit"] == 0.5 and m["noncoco"]["owl_given_yolo_miss"] == 0.5
    assert m["yolo_box_covered_by_owl"] == 1 / 3 and m["yolo_tp_box_covered_by_owl"] == 0.5
    assert m["owl_box_not_by_yolo"] == 0.5 and m["owl_only_box_matches_gt"] == 1.0
