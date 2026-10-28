"""ex003のモデル定義：カテゴリアノテーションに用いるVLM f．"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402

# CLIP-ViT-B32が論文の代表VLM．他の2手法はパイプライン全体での比較用に追加で評価する
METHODS = ["CLIP-ViT-B32", "CLIP-RN50", "ResNet50-IN1k"]


def load_model(name, seed=0):
    """
    学習済みVLMをインスタンス化する．

    Args:
        name (str): METHODSのいずれか．
        seed (int): 乱数シード．

    Returns:
        ResNet50IN1k | CLIPModel: score(images, vocab) を持つモデル．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return load_vlm(name, "cuda" if torch.cuda.is_available() else "cpu")
