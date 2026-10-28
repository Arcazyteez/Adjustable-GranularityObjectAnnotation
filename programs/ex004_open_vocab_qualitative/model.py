"""ex004のモデル定義：提案パイプラインを構成するDETRとVLM（CLIP-ViT-B/32）．"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.detectors import DETRDetector  # noqa: E402
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402


def load_model(vlm_name="CLIP-ViT-B32", detr_threshold=0.5, seed=0):
    """
    パイプラインのモデル（検出器とVLM）をインスタンス化する．

    Args:
        vlm_name (str): VLMの名前．
        detr_threshold (float): DETRの検出スコア閾値．
        seed (int): 乱数シード．

    Returns:
        tuple[DETRDetector, CLIPModel]: 検出器とVLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DETRDetector(threshold=detr_threshold, device=device), load_vlm(vlm_name, device)
