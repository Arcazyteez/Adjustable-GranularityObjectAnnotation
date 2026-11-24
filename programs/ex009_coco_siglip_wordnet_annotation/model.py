"""
ex009のモデル定義：物体領域の検出（DETR，YOLO11x）とSigLIPによるカテゴリのアノテーション，WordNetによる粒度変更．

SigLIPは候補語彙 V 上のロジット l_v = t cos(x, w_v) + b を出力する．各段階で選ぶ語は argmax l_v であり，
確信度はシグモイド σ(l_v)（語ごとに独立）とソフトマックス（候補語彙の中で正規化）の両方を記録する．
"""
import os
import sys

import torch
from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.detectors import DETRDetector  # noqa: E402
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm, vlm_logits  # noqa: E402
from common.wordnet_utils import hyponym_candidates  # noqa: E402
from data import import_experiment_module  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLM_NAME = "SigLIP-ViT-B16"
NORMALIZATIONS = ["sigmoid", "softmax"]


def load_detr(seed=0, threshold=0.5):
    """
    クラス非依存の物体領域の検出に用いるDETR（ResNet-50）をインスタンス化する．

    Args:
        seed (int): 乱数シード．
        threshold (float): 検出スコアの閾値．

    Returns:
        DETRDetector: detect(image) を持つ検出器．
    """
    set_seed(seed)
    return DETRDetector(threshold=threshold, device=DEVICE)


def load_yolo(seed=0, conf=0.25):
    """
    COCOの80クラスで学習済みのYOLO11xをインスタンス化する（ex005の YOLODetector）．

    Args:
        seed (int): 乱数シード．
        conf (float): 検出スコアの閾値．

    Returns:
        YOLODetector: detect(image) を持つ検出器．
    """
    return import_experiment_module("ex005_coco_val500_yolo_comparison", "model").load_model("yolo11x", conf, seed)


def load_model(seed=0):
    """
    カテゴリのアノテーションに用いるSigLIP-ViT-B/16をインスタンス化する．

    Args:
        seed (int): 乱数シード．

    Returns:
        SigLIPModel: image_features，text_features を持つVLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return load_vlm(VLM_NAME, DEVICE)


def normalize(logits, normalization):
    """
    ロジットを確信度に変換する．

    Args:
        logits (torch.Tensor): ロジット（... × V）．
        normalization (str): "sigmoid" または "softmax"．

    Returns:
        torch.Tensor: 確信度（... × V）．
    """
    return logits.sigmoid() if normalization == "sigmoid" else logits.softmax(dim=-1)


def hierarchical_chain(vlm, image_feature, root_word, root_synset_name):
    """
    深さ0の語から出発し，下位語集合が空になるまでWordNetの下位語をたどる（各段階で argmax のロジットの語を選ぶ）．

    Args:
        vlm (SigLIPModel): VLM．
        image_feature (torch.Tensor): クロップのL2正規化済みの画像特徴（D）．
        root_word (str): 深さ0の語．
        root_synset_name (str): 深さ0の語のSynset名．

    Returns:
        list[dict]: 各深さの {"depth", "word", "synset", "conf_sigmoid", "conf_softmax", "n_candidates"}
            （深さ0の確信度は None）．
    """
    synset = wn.synset(root_synset_name)
    chain = [{"depth": 0, "word": root_word, "synset": root_synset_name, "conf_sigmoid": None, "conf_softmax": None,
              "n_candidates": None}]
    while True:
        words, synsets = hyponym_candidates(synset)
        if not words:
            return chain
        logits = vlm_logits(vlm, image_feature, words)
        best = int(logits.argmax())
        synset = synsets[best]
        chain.append({"depth": len(chain), "word": words[best], "synset": synset.name(),
                      "conf_sigmoid": float(logits[best].sigmoid()),
                      "conf_softmax": float(logits.softmax(dim=-1)[best]), "n_candidates": len(words)})
