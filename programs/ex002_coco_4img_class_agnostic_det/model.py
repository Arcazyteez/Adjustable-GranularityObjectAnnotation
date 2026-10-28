"""ex002のモデル定義：比較する物体領域検出器（DETR，OWL-ViT，Grounding DINO）．"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.detectors import DETRDetector, GroundingDINODetector, OWLViTDetector  # noqa: E402
from common.paths import set_seed  # noqa: E402


def load_model(name, params, queries, seed=0):
    """
    学習済みの検出器をインスタンス化する．

    Args:
        name (str): "DETR"，"OWL-ViT"，"Grounding DINO" のいずれか．
        params (dict): 検出器の閾値等のハイパーパラメータ．
        queries (list[str]): オープンボキャブラリ検出器に与えるテキストクエリ（DETRでは未使用）．
        seed (int): 乱数シード．

    Returns:
        DETRDetector | OWLViTDetector | GroundingDINODetector: detect(image) を持つ検出器．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "DETR":
        return DETRDetector(device=device, **params)
    if name == "OWL-ViT":
        return OWLViTDetector(queries, device=device, **params)
    return GroundingDINODetector(queries, device=device, **params)
