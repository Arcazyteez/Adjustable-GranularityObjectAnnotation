"""ex016（全体らしさの特徴による物体の一部・背景の領域の棄却）のテスト．"""
import importlib.util
import os
import sys

import numpy as np

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex016_wholeness_rejection"))
sys.path.insert(0, PROGRAMS)
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.insert(0, DIRECTORY)


def _load(name):
    """
    ex016のモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(f"ex016_{name}", os.path.join(DIRECTORY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


ex016_data = _load("data")
ex016_model = _load("model")
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def test_meronym_relations():
    """WordNetの部分と全体の関係：wheel は car の部分，hand は person の部分，car は person の部分でない．"""
    assert ex016_data.is_meronym("wheel.n.01", ["car.n.01"])
    assert ex016_data.is_meronym("hand.n.01", ["person.n.01"])
    assert not ex016_data.is_meronym("car.n.01", ["person.n.01"])
    assert ex016_data.is_part_word("handle.n.01")
    assert not ex016_data.is_part_word("dog.n.01")


def test_containment_matrix():
    """含まれる割合：小さいボックスは大きいボックスに全て含まれ，逆は面積の比になる．"""
    c, area = ex016_data.containment_matrix(np.array([[0, 0, 10, 10], [2, 2, 4, 4]], dtype=np.float64))
    assert np.isclose(c[1, 0], 1.0) and np.isclose(c[0, 1], 0.04) and c[0, 0] == 0
    assert np.allclose(area, [100, 4])


def test_wholeness_features():
    """car のYOLO11xのボックス内の wheel の領域は meronym が1，YOLO11xのボックスの特徴は全て0．"""
    img = {"boxes": [{"box": [0, 0, 100, 100]}, {"box": [10, 60, 30, 90]}, {"box": [200, 200, 300, 300]}], "n_yolo": 1}
    recs = [{"synset": "car.n.01", "chain": [("car.n.01", 1.0)], "source": "yolo"},
            {"synset": "wheel.n.01", "chain": [("wheel.n.01", 0.5)], "source": "owlv2"},
            {"synset": "dog.n.01", "chain": [("dog.n.01", 0.5)], "source": "owlv2"}]
    f = ex016_data.wholeness_features(img, recs)
    assert f.shape == (3, len(ex016_data.WHOLENESS_NAMES))
    assert np.all(f[0] == 0)
    assert f[1, 0] == 1.0 and f[1, 4] == 1.0 and f[1, 5] == 1.0
    assert f[2, 0] == 0.0 and f[2, 4] == 0.0 and np.isclose(f[2, 3], 0.0)


def test_rule_score():
    """rule：meronym が1の出力のみスコアを0にする．"""
    x = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    whole = np.zeros((40, 6), dtype=np.float32)
    whole[:5, 4] = 1
    y = (x[:, 0] > 0).astype(np.float64)
    cals = ex016_model.fit_scorers(x, whole, y)
    s = ex016_model.predict_scores(cals, x, whole)
    assert set(s) == set(ex016_model.SCORERS)
    assert np.all(s["rule"][:5] == 0) and np.allclose(s["rule"][5:], s["base"][5:])
