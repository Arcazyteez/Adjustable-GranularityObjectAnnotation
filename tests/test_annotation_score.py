"""ex015a（アノテーションのスコア）のテスト．"""
import importlib.util
import os
import sys

import numpy as np

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex015a_annotation_score"))
sys.path.insert(0, PROGRAMS)
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.insert(0, DIRECTORY)


def _load(name):
    """
    ex015aのモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(f"ex015a_{name}", os.path.join(DIRECTORY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


ex015a_data = _load("data")
ex015a_model = _load("model")
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def test_match_greedy_prefers_high_score():
    """スコアの高い出力が先にGTに対応付けられることを確認する．"""
    iou = np.array([[0.6], [0.9]])
    assert ex015a_model.match_greedy(iou, np.array([0.9, 0.1])).tolist() == [0, -1]
    assert ex015a_model.match_greedy(iou, np.array([0.1, 0.9])).tolist() == [-1, 0]
    assert ex015a_model.match_greedy(np.array([[0.3]]), np.array([1.0])).tolist() == [-1]


def test_average_precision_perfect_and_monotone():
    """全て正しい場合のAPが再現率の最大値となり，適合率を右側の最大値で単調にすることを確認する．"""
    assert abs(ex015a_model.average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])) - 1.0) < 1e-9
    ap = ex015a_model.average_precision(np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.5, 0.67]))
    assert abs(ap - (0.5 * 1.0 + 0.5 * 0.67)) < 1e-9


def test_expected_calibration_error_ignores_unknown():
    """校正誤差がラベル不明の出力を除いて計算されることを確認する．"""
    s = np.array([0.95, 0.95, 0.05, 0.5])
    y = np.array([1.0, 1.0, 0.0, -1.0])
    ece, table = ex015a_model.expected_calibration_error(s, y)
    assert abs(ece - 0.05) < 1e-9 and sum(t["n"] for t in table) == 3


def test_calibrator_learns_monotone_relation():
    """ロジスティック回帰が，正解と相関する特徴に正の重みを付けることを確認する．"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2)).astype(np.float32)
    y = (x[:, 0] + 0.1 * rng.normal(size=500) > 0).astype(np.float64)
    cal = ex015a_model.LogisticCalibrator().fit(x, y)
    assert cal.w[0] > 1.0 and abs(cal.w[1]) < 0.5
    p = cal.predict(np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32))
    assert p[0] > 0.9 and p[1] < 0.1


def test_raw_scores():
    """検出のスコアと命名の確率の組み合わせを確認する．"""
    recs = [{"det": 0.8, "q": 0.5, "source": "yolo"}, {"det": 0.2, "q": float("nan"), "source": "owlv2"},
            {"det": 0.4, "q": 0.5, "source": "owlv2"}]
    assert np.allclose(ex015a_model.raw_score(recs, "det_x_name"), [0.4, 0.2, 0.2])
    assert np.allclose(ex015a_model.raw_score(recs, "det_x_name_added"), [0.8, 0.2, 0.2])


def test_curve_counts_named_objects():
    """曲線の再現率・命名の割合が，スコアの順に出力を加えた場合の累積となることを確認する．"""
    flat = {"inter": np.array([2.0, 0.0]), "pred": np.array([2.0, 0.0]), "fed_extra": np.array([0.0, 3.0]),
            "strict_extra": np.array([0.0, 3.0]), "named": np.array([1.0, 0.0]), "fine": np.array([1.0, 0.0]),
            "noncoco": np.array([1.0, 0.0]), "added": np.array([1.0, 1.0]), "label": np.array([1.0, 0.0]),
            "score": np.array([0.9, 0.1])}
    cur = ex015a_model.curve(flat, {"gt_size": 4, "noncoco_n": 1, "coco_n": 1})
    assert np.allclose(cur["hR"], [0.5, 0.5]) and np.allclose(cur["noncoco_named"], [1.0, 1.0])
    assert abs(cur["fed_hP"][1] - 0.4) < 1e-9 and np.allclose(cur["added_precision"], [1.0, 0.5])
