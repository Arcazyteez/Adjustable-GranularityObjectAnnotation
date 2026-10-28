"""ex001のモデル定義：比較する3種類のVLM f（ResNet50-IN1k，CLIP-RN50，CLIP-ViT-B/32）．"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402

METHODS = ["ResNet50-IN1k", "CLIP-RN50", "CLIP-ViT-B32"]


def load_model(name, seed=0):
    """
    学習済みVLMをインスタンス化する（学習は行わないため重みは公開済みのものを用いる）．

    Args:
        name (str): METHODSのいずれか．
        seed (int): 乱数シード．

    Returns:
        ResNet50IN1k | CLIPModel: score(images, vocab) → np.ndarray（N×V）を持つモデル．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return load_vlm(name, "cuda" if torch.cuda.is_available() else "cpu")
