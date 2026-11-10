"""
ex007のモデル定義：物体領域検出のDETR，閉域検出のYOLO11x（ex005の YOLODetector），VLMのCLIP-ViT-B/32．

WordNetの下位語をたどる粒度可変アノテーション（論文の式(3)）は common.wordnet_utils.hierarchical_annotation と
同じ処理であるが，ここではクロップの画像特徴を1回だけ計算し，各段階の候補語彙のテキスト特徴との内積で
スコアを求めることで，全インスタンスの下位語の展開を高速に行う．
"""
import os
import sys

import torch
from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.detectors import DETRDetector  # noqa: E402
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402
from common.wordnet_utils import hyponym_candidates  # noqa: E402
from data import DETR_BASE_THRESHOLD, YOLO_BASE_CONF, import_experiment_module  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_detr(seed=0):
    """
    物体領域検出に用いるDETR（ResNet-50，閾値0.5）をインスタンス化する．

    Args:
        seed (int): 乱数シード．

    Returns:
        DETRDetector: detect(image) を持つ検出器．
    """
    set_seed(seed)
    return DETRDetector(threshold=DETR_BASE_THRESHOLD, device=DEVICE)


def load_yolo(seed=0):
    """
    閉域検出に用いるYOLO11x（閾値0.25，ex005と同じ設定）をインスタンス化する．

    Args:
        seed (int): 乱数シード．

    Returns:
        YOLODetector: detect(image) を持つ検出器．
    """
    ex005_model = import_experiment_module("ex005_coco_val500_yolo_comparison", "model")
    return ex005_model.load_model("yolo11x", YOLO_BASE_CONF, seed)


def load_model(name="CLIP-ViT-B32", seed=0):
    """
    カテゴリのアノテーションに用いるVLMをインスタンス化する．

    Args:
        name (str): VLMの名前．
        seed (int): 乱数シード．

    Returns:
        CLIPModel: image_features，text_features，score を持つVLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return load_vlm(name, DEVICE)


@torch.no_grad()
def scores_from_features(vlm, image_feature, vocab):
    """
    1つのクロップの画像特徴から，候補語彙上のスコア（論文の式(2)のソフトマックス）を計算する．

    Args:
        vlm (CLIPModel): CLIP．
        image_feature (torch.Tensor): L2正規化済みの画像特徴（D）．
        vocab (list[str]): 候補語彙（長さV）．

    Returns:
        torch.Tensor: スコア（V），和は1．
    """
    logits = vlm.model.logit_scale.exp() * image_feature @ vlm.text_features(vocab).T
    return logits.softmax(dim=-1)


def hierarchical_annotation_from_feature(vlm, image_feature, root_word, root_synset_name, root_conf):
    """
    深さ0の単語から出発し，下位語集合が空になるまでWordNetの下位語をたどる（論文の式(3)）．

    Args:
        vlm (CLIPModel): CLIP．
        image_feature (torch.Tensor): クロップのL2正規化済みの画像特徴（D）．
        root_word (str): 深さ0の単語．
        root_synset_name (str): 深さ0の単語に対応するSynset名．
        root_conf (float): 深さ0の確信度．

    Returns:
        list[dict]: 各深さの {"depth", "word", "synset", "conf", "n_candidates"} のリスト．
    """
    synset = wn.synset(root_synset_name)
    chain = [{"depth": 0, "word": root_word, "synset": root_synset_name, "conf": float(root_conf), "n_candidates": None}]
    while True:
        words, synsets = hyponym_candidates(synset)
        if not words:
            return chain
        scores = scores_from_features(vlm, image_feature, words)
        best = int(scores.argmax())
        synset = synsets[best]
        chain.append({"depth": len(chain), "word": words[best], "synset": synset.name(),
                      "conf": float(scores[best]), "n_candidates": len(words)})
