"""ex021（order_033）の棄却のスコアと重複・部分の除去のテスト．"""
import os
import sys

import numpy as np

EX021_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex021_coco_owl_rejection")
sys.path.insert(0, EX021_DIR)

from model import dedup_mask, pairwise_overlap, rejection_score  # noqa: E402

# 他の実験のテストも data，model，train という同名のモジュールを読み込むため，読み込み後に取り除く
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.remove(EX021_DIR)


def test_rejection_score():
    """各スコアの定義（積と単独）．"""
    conf, obj, mass = np.array([0.5, 0.2]), np.array([0.4, 1.0]), np.array([0.9, 0.1])
    assert np.allclose(rejection_score("joint", conf, obj, mass), [0.2, 0.2])
    assert np.allclose(rejection_score("mass_joint", conf, obj, mass), [0.36, 0.1])
    assert np.allclose(rejection_score("objectness", conf, obj, mass), obj)


def test_overlap_and_dedup():
    """同じラベルの重複は NMS で，同じラベルの内側の小さいボックスは包含で除き，違うラベルは残す．"""
    boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100], [10, 10, 30, 30], [10, 10, 30, 30]], float)
    iou, contain = pairwise_overlap(boxes)
    assert iou[0, 1] > 0.5 and contain[2, 0] == 1.0
    labels, scores = ["person", "person", "person", "tie"], np.array([0.9, 0.8, 0.7, 0.6])
    assert dedup_mask(boxes, labels, scores, "none").tolist() == [True] * 4
    assert dedup_mask(boxes, labels, scores, "nms").tolist() == [True, False, True, True]
    assert dedup_mask(boxes, labels, scores, "nms_contain").tolist() == [True, False, False, True]


def test_e2_grid_names_unique():
    """order_037 の E2_factorial の条件の組（144通り）の手法名が重複しない．"""
    import itertools
    for _name in ("data", "model", "train"):
        sys.modules.pop(_name, None)
    sys.path.insert(0, EX021_DIR)
    cwd = os.getcwd()
    os.chdir(EX021_DIR)
    try:
        import train
        names = {train.e2_name(f"{d}@{b}" if b else d, e, dd, s, l)
                 for (d, b), e, s, dd, l in itertools.product(train.E2_DETECTORS, train.E2_EMBEDDERS, train.E2_SCORES,
                                                              train.E2_DEDUPS, train.LABEL_TYPES)}
        assert len(names) == 4 * 3 * 3 * 2 * 2
        assert train.e2_name("OWLv2@50", "AlphaCLIP-ViT-B16_crop", "nms_contain", "mass_joint", "mass") \
            == "OWLv2@50_Alpha-crop_nms_contain_mass_joint_masslabel"
    finally:
        os.chdir(cwd)
        for _name in ("data", "model", "train"):
            sys.modules.pop(_name, None)
        sys.path.remove(EX021_DIR)


def test_e3_grid_names_unique():
    """E3_detector（order_039）の手法名が，検出器 × 分類器 × 判定の全ての組で重複しないことを確かめる．"""
    import itertools
    import os
    import sys
    path = os.path.join(os.path.dirname(__file__), "..", "programs", "ex021_coco_owl_rejection")
    for name in ("data", "model", "train"):
        sys.modules.pop(name, None)
    sys.path.insert(0, path)
    try:
        import train
    finally:
        sys.path.remove(path)
    names = {f"{f'{d}@{b}' if b else d}_{train.E2_SHORT[e]}_{j}"
             for (d, b), e, j in itertools.product(train.E3_DETECTORS, train.E2_EMBEDDERS, train.E3_JUDGES)}
    assert len(names) == len(train.E3_DETECTORS) * len(train.E2_EMBEDDERS) * len(train.E3_JUDGES) == 24
    assert train.E3_JUDGES["proposed"] == ("nms_contain", "mass_joint", "mass")
