"""SigLIPとWordNetの同じ深さの語彙による深さ0の決定（programs/ex008_coco_val_siglip_wordnet_root）のテスト．"""
import importlib.util
import os
import sys

import numpy as np
import torch
from nltk.corpus import wordnet as wn

EX008_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex008_coco_val_siglip_wordnet_root")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs"))
sys.path.insert(0, EX008_DIR)
# 他の実験のテストも data，model，train という名前のモジュールを読み込むため，別名で読み込む
_spec = importlib.util.spec_from_file_location("ex008_train", os.path.join(EX008_DIR, "train.py"))
ex008_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex008_train)
ex008_data, ex008_model = sys.modules["data"], sys.modules["model"]
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(EX008_DIR)]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def test_same_depth_synsets_are_cousins():
    """dogと同じ深さの語にwolfやfoxが含まれ，dogの上位語・下位語が含まれないことを確認する．"""
    dog = wn.synset("dog.n.01")
    cousins = ex008_data.same_depth_synsets(dog, 2)
    names = {s.name() for s in cousins}
    assert "wolf.n.01" in names and "dog.n.01" in names
    assert "canine.n.02" not in names and "puppy.n.01" not in names


def test_wordnet_root_vocab_has_coco_first_and_no_hypernyms():
    """語彙の先頭80語がCOCOのカテゴリで，COCO外の語にCOCOの上位語・下位語や物理的な実体でない語を含まないことを確認する．"""
    vocab = ex008_data.build_wordnet_root_vocab(2, 1)
    assert all(v["coco"] is not None for v in vocab[:80]) and all(v["coco"] is None for v in vocab[80:])
    synsets = {v["synset"] for v in vocab}
    assert "animal.n.01" not in synsets and "penguin.n.01" not in synsets  # birdの上位語・下位語
    # COCOのカテゴリ（traffic light等）以外の語は物理的な実体に限る
    assert all(ex008_data.PHYSICAL_ENTITY in ex008_data.ancestors(v["synset"]) for v in vocab[80:])
    assert len({v["word"] for v in vocab}) == len(vocab)


def test_normalize_sigmoid_is_independent():
    """シグモイドは他の語のロジットに依存せず，ソフトマックスは和が1になることを確認する．"""
    logits = torch.tensor([2.0, 0.0, -1.0])
    sig = ex008_model.normalize(logits, "sigmoid")
    assert torch.isclose(ex008_model.normalize(logits[:1], "sigmoid")[0], sig[0])
    assert torch.isclose(ex008_model.normalize(logits, "softmax").sum(), torch.tensor(1.0))


def test_decide_root_cmp_uses_vlm_only_when_higher():
    """hybrid_cmp はVLMの最大確信度がYOLOの検出スコアより高い場合のみVLMの語を用いることを確認する．"""
    vocab = [{"word": "dog", "synset": "dog.n.01", "coco": "dog"}, {"word": "wolf", "synset": "wolf.n.01", "coco": None}]
    ins = {"yolo_word": "dog", "det_score": 0.6}
    low = ex008_train.decide_root("hybrid_cmp_wn2", ins, np.array([0.1, 0.5]), vocab)
    high = ex008_train.decide_root("hybrid_cmp_wn2", ins, np.array([0.1, 0.7]), vocab)
    assert low["root_word"] == "dog" and low["root_source"] == "yolo" and not low["switched"]
    assert high["root_word"] == "wolf" and high["root_synset"] == "wolf.n.01" and high["switched"]
    assert high["vlm_conf_yolo_word"] == 0.1
