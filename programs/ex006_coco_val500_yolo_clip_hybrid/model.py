"""
ex006のモデル定義：深さ0のカテゴリを出力するYOLO（ex005の YOLODetector）と，
深さ1以降の細分化を行うVLM（CLIP-ViT-B/32）．
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402
from data import import_experiment_module  # noqa: E402


def load_model(yolo_name, conf, seed=0):
    """
    学習済みのYOLOをインスタンス化する（ex005と同じ設定）．

    Args:
        yolo_name (str): YOLOのモデル名．
        conf (float): 検出スコアの閾値．
        seed (int): 乱数シード．

    Returns:
        YOLODetector: detect(image) を持つ検出器．
    """
    ex005_model = import_experiment_module("ex005_coco_val500_yolo_comparison", "model")
    return ex005_model.load_model(yolo_name, conf, seed)


def load_vlm_model(name="CLIP-ViT-B32", seed=0):
    """
    深さ1以降の細分化と，COCO外の語への切替の判定に用いるVLMをインスタンス化する．

    Args:
        name (str): VLMの名前．
        seed (int): 乱数シード．

    Returns:
        CLIPModel: score(images, vocab) を持つVLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return load_vlm(name, "cuda" if torch.cuda.is_available() else "cpu")
