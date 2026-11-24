"""ex020（order_031，order_032）の粒度の変更，VG.pdf の式(8)の評価の集合，部分木の確率の合計のテスト．"""
import os
import sys

import numpy as np
import torch

EX020_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex020_coco_owl_alphaclip_granularity")
sys.path.insert(0, EX020_DIR)

from data import eval_set, fine_labels, ordered_gt_path  # noqa: E402
from model import TreeNamer, truncate  # noqa: E402
from train import instance_score  # noqa: E402

# 他の実験のテストも data，model，train という同名のモジュールを読み込むため，読み込み後に取り除く
for _name in ("data", "model", "train"):
    sys.modules.pop(_name, None)
sys.path.remove(EX020_DIR)

CHAIN = [{"depth": 0, "word": "person", "synset": "person.n.01", "score": 0.9},
         {"depth": 1, "word": "adult", "synset": "adult.n.01", "score": 0.6},
         {"depth": 2, "word": "man", "synset": "man.n.01", "score": 0.3}]


class FakeEmbedder:
    """語ごとに固定のテキスト特徴を返す埋め込みモデル（logit_scale は exp(0) = 1）．"""

    def __init__(self, table):
        """
        Args:
            table (dict[str, list[float]]): 語 → 特徴．
        """
        self.table = table
        self.model = type("M", (), {"logit_scale": torch.tensor(0.0)})()

    def text_features(self, words):
        """
        Args:
            words (list[str]): 語．

        Returns:
            torch.Tensor: 特徴（V×D）．
        """
        return torch.tensor([self.table[w] for w in words], dtype=torch.float32)


def toy_tree():
    """
    VG.pdf の図1と同じ形の木（clothing の下に shoe と boot，light の下に lamp）．

    Returns:
        dict: data.build_tree と同じ形式の木．
    """
    nodes = ["frame:clothing", "frame:light", "shoe.n.01", "boot.n.01", "lamp.n.01"]
    parent = {"frame:clothing": "root", "frame:light": "root", "shoe.n.01": "frame:clothing", "boot.n.01": "frame:clothing",
              "lamp.n.01": "frame:light"}
    depth = {"frame:clothing": 1, "frame:light": 1, "shoe.n.01": 2, "boot.n.01": 2, "lamp.n.01": 2}
    words = {"frame:clothing": None, "frame:light": None, "shoe.n.01": "shoe", "boot.n.01": "boot", "lamp.n.01": "lamp"}
    return {"nodes": nodes, "parent": parent, "depth": depth, "words": words, "synset": {n: n for n in nodes}}


def test_truncate_by_theta_and_depth():
    """θ_f と最大の深さで出力の列の先頭部分が得られ，深さ0は常に残る．"""
    assert len(truncate(CHAIN)) == 3
    assert len(truncate(CHAIN, max_depth=0)) == 1
    assert len(truncate(CHAIN, theta_f=0.5)) == 2
    assert len(truncate(CHAIN, theta_f=0.99)) == 1


def test_eval_set_and_instance_score():
    """式(8)の集合は深さ4以上の上位語と COCO のクラスを含み，出力の判定が正しい．"""
    assert ordered_gt_path("person.n.01", "man.n.01")[-1] == "man.n.01"
    assert fine_labels("person", ["man.n.01", "dog.n.01"]) == ["man.n.01"]
    s = eval_set("man.n.01")
    assert {"man.n.01", "adult.n.01", "person.n.01"} <= s and "causal_agent.n.01" not in s
    assert instance_score("man.n.01", ["man.n.01"], "person.n.01")["status"] == "exact"
    assert instance_score("adult.n.01", ["man.n.01"], "person.n.01")["status"] == "finer"
    assert instance_score("person.n.01", ["man.n.01"], "person.n.01")["status"] == "coarse_correct"
    assert instance_score("dog.n.01", ["man.n.01"], "person.n.01")["status"] == "wrong"
    r = instance_score(None, ["man.n.01"], "person.n.01")
    assert r["status"] == "none" and r["n_pred"] == 0 and r["n_gt"] == len(s)


def test_subtree_mass_matches_figure1():
    """VG.pdf の図1：shoe 0.30，boot 0.25，lamp 0.45 で τ = 0.5 のとき clothing（部分木の合計 0.55）を出力する．"""
    p = np.array([0.30, 0.25, 0.45])
    emb = FakeEmbedder({"shoe": [1.0, 0.0, 0.0], "boot": [0.0, 1.0, 0.0], "lamp": [0.0, 0.0, 1.0]})
    namer = TreeNamer(emb, toy_tree(), [])
    cos = np.log(p)[None, :]                                   # logit_scale = 1 なので softmax(cos) = p
    assert namer.subtree_mass(cos, 1.0, 0.5) == ["frame:clothing"]
    assert namer.subtree_mass(cos, 1.0, 0.4) == ["lamp.n.01"]  # 葉の lamp（0.45）が深い
    assert namer.subtree_mass(cos, 1.0, 1.01) == [None]
    assert namer.flat(cos, 0.0) == ["lamp.n.01"] and namer.flat(cos, 0.5) == [None]
