"""ex019（order_029）の Alpha-CLIP の入力の作成，OWLv2 の予算の閾値，対応のあるブートストラップのテスト．"""
import os
import sys

import numpy as np
from PIL import Image

EX019_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex019_coco_owl_alphaclip_wordnet")
sys.path.insert(0, EX019_DIR)

from data import make_alpha_input, square_window  # noqa: E402
from train import budget_thresholds, paired_bootstrap  # noqa: E402

# 他の実験のテストも data，model，train という同名のモジュールを読み込むため，読み込み後に取り除く
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.remove(EX019_DIR)


def test_alpha_input_shapes_and_mask_area():
    """全ての入力で 224×224 のRGBとαが得られ，αはボックスの部分だけ1になる．"""
    image = Image.new("RGB", (400, 200), (255, 0, 0))
    box = [100, 50, 200, 150]
    for mode in ("crop", "context1.5", "context2", "full"):
        rgb, alpha = make_alpha_input(image, box, mode)
        assert rgb.size == (224, 224) and alpha.shape == (224, 224)
        assert set(np.unique(alpha)) <= {0.0, 1.0} and alpha.sum() > 0
    _, alpha = make_alpha_input(image, box, "context2")
    assert abs(alpha.mean() - 0.25) < 0.02            # 一辺2倍の窓ではボックスは面積の1/4
    _, alpha = make_alpha_input(image, box, "full")
    assert abs(alpha.mean() - (100 * 100) / (400 * 400)) < 0.01
    assert square_window(image, box, "context1.5") == (75.0, 25.0, 150.0)


def test_budget_thresholds_mean_count():
    """dev の閾値で1枚あたりの平均ボックス数が予算と一致する．"""
    dets = {i: {"scores": np.linspace(1, 0.01, 100)} for i in range(4)}
    th = budget_thresholds(dets)
    for b, t in th.items():
        assert np.mean([(d["scores"] >= t).sum() for d in dets.values()]) == b


def test_paired_bootstrap_detects_difference():
    """同じ計数では差が0，明らかに良い手法では差が正で p 値が小さい．"""
    rng = np.random.default_rng(0)
    a = np.column_stack([np.full(50, 10), rng.integers(3, 6, 50), np.full(50, 8), rng.integers(3, 6, 50),
                         np.full(50, 12), np.full(50, 10)]).astype(float)
    b = a.copy()
    b[:, 1] += 4
    b[:, 3] += 4
    same = paired_bootstrap(a, a, n=200)
    assert same["f1"]["diff"] == 0.0
    better = paired_bootstrap(a, b, n=200)
    assert better["top1_acc"]["diff"] > 0 and better["f1"]["ci95"][0] > 0 and better["f1"]["p_value"] <= 0.01
