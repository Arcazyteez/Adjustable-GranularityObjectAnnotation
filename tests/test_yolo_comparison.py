"""YOLOとの比較（programs/ex005_coco_val500_yolo_comparison/train.py）の集計関数のテスト．"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "ex005_coco_val500_yolo_comparison"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs"))
from train import compare_common, match_instances, summarize  # noqa: E402


def test_match_instances_and_summarize():
    """IoU 0.5以上で対応付けられた検出のみが評価され，正命名率がGT総数で割られることを確認する．"""
    samples = [{"image_id": 1, "gt_boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), "gt_names": ["dog", "cat"]}]
    detections = [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "scores": [0.9, 0.8],
                   "top5": [["dog", "cat", "a", "b", "c"], ["cat", "dog", "a", "b", "c"]],
                   "top5_conf": [[0.9, 0.1, 0, 0, 0], [0.8, 0.1, 0, 0, 0]]}]
    instances = match_instances(detections, samples)
    assert len(instances) == 1 and instances[0]["gt_index"] == 0 and instances[0]["top1_correct"]
    metrics = summarize(instances, n_detections=2, n_gt=2)
    assert metrics["top1_accuracy"] == 1.0 and metrics["annotation_recall"] == 0.5


def test_compare_common_counts_discordant_pairs():
    """共通GTのみを比較し，片方だけが正解した件数とMcNemarのp値が計算されることを確認する．"""
    instances = [{"image_id": 1, "gt_index": g, "gt_name": "person" if g == 0 else "dog", "top1_correct": True}
                 for g in range(3)]
    proposed = {(1, 0): {"top1_correct": False}, (1, 1): {"top1_correct": True}}
    result = compare_common(instances, proposed)
    assert result["n_common"] == 2 and result["yolo_only"] == 1 and result["proposed_only"] == 0
    assert result["yolo_top1_person"] == 1.0 and result["proposed_top1_person"] == 0.0
    assert 0.0 < result["mcnemar_p"] <= 1.0
