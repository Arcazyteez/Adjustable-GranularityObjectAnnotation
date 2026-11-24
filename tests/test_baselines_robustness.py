"""ex015b（命名の方法の比較，ばらつき，誤りの分解）のテスト．"""
import importlib.util
import os
import sys

import numpy as np

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex015b_baselines_robustness"))
sys.path.insert(0, PROGRAMS)
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.insert(0, DIRECTORY)


def _load(name):
    """
    ex015bのモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(f"ex015b_{name}", os.path.join(DIRECTORY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


ex015b_data = _load("data")
ex015b_model = _load("model")
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def _parts():
    """
    2枚の画像の評価の量を作る．

    Returns:
        list[dict]: per_image_outcomes と同じ形式．
    """
    keys = ["inter", "pred", "fed_extra", "strict_extra", "named", "fine", "noncoco", "added", "label"]
    a = {k: np.array([1.0, 0.0]) for k in keys} | {"score": np.array([0.9, 0.1])}
    b = {k: np.array([0.0]) for k in keys} | {"score": np.array([0.5]), "strict_extra": np.array([1.0]),
                                              "pred": np.array([0.0])}
    return [a, b]


def test_label_ap_counts_only_correct_labels():
    """ラベルの正誤による AP が，正しい出力の順位と GT の数から計算されることを確認する．"""
    flat = {"fine": np.array([1.0, 0.0, 1.0]), "named": np.array([1.0, 1.0, 1.0]), "pred": np.array([1.0, 1.0, 1.0]),
            "score": np.array([0.9, 0.8, 0.1])}
    ap = ex015b_model.label_ap(flat, 4)
    # exact：再現率 0.25（適合率1），0.5（適合率 2/3）
    assert abs(ap["AP_exact"] - (0.25 * 1.0 + 0.25 * 2 / 3)) < 1e-9
    assert abs(ap["AP_named"] - 0.75) < 1e-9


def test_metrics_from_parts_and_bootstrap_identity():
    """同じ構成どうしのブートストラップの差が0になり，指標が計算できることを確認する．"""
    parts = _parts()
    rows = np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    m = ex015b_model.metrics_from_parts(parts, rows)
    assert set(m) == {"AP_fed", "AP_strict", "AP_exact", "AP_named"}
    boot = ex015b_model.bootstrap_difference(parts, parts, rows, 20, 0)
    assert all(abs(v["diff"]) < 1e-12 and abs(v["lo"]) < 1e-12 and abs(v["hi"]) < 1e-12 for v in boot.values())


def test_containment():
    """領域の面積のうちGTのボックスに含まれる割合を確認する．"""
    c = ex015b_model._containment([0, 0, 10, 10], [{"box": [0, 0, 20, 20]}, {"box": [5, 0, 20, 10]}])
    assert np.allclose(c, [1.0, 0.5])
