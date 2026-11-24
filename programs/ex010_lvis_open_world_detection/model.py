"""
ex010のモデル定義：COCO外の物体を含む領域の候補を出力する4つの検出器と，命名・粒度変更に用いるSigLIP．

すべての検出器はクラス非依存のボックスとスコアを出力し，IoU 0.5のクラス非依存NMSの後，スコアの上位 MAX_BOXES 件を残す．
- yolo_low:  YOLO11x（閾値0.01）．スコアは80クラスの最大スコア
- detr_low:  DETR（閾値0.01）．スコアは「物体なし」以外のクラスの最大確率
- owlv2:     OWLv2（google/owlv2-base-patch16-ensemble）の物体らしさ（objectness）のスコア．テキストに依存しない
- gdino:     Grounding DINO（tiny）にテキスト「object.」を与えたスコア
"""
import os
import sys

import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.detectors import DETRDetector, GroundingDINODetector, class_agnostic_nms  # noqa: E402
from common.paths import set_seed  # noqa: E402
from data import import_experiment_module  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BOXES = 100
NMS_IOU = 0.5
DETECTORS = ["yolo_low", "detr_low", "owlv2", "gdino"]


def ex009_model():
    """
    ex009の model モジュール（SigLIP，WordNetの下位語の展開，DETR，YOLO11x）を返す．

    Returns:
        module: ex009の model．
    """
    return import_experiment_module("ex009_coco_siglip_wordnet_annotation", "model")


def top_boxes(boxes, scores):
    """
    クラス非依存NMSを行い，スコアの上位 MAX_BOXES 件を返す．

    Args:
        boxes (np.ndarray | torch.Tensor): xyxy形式のボックス（K×4）．
        scores (np.ndarray | torch.Tensor): スコア（K）．

    Returns:
        tuple[np.ndarray, np.ndarray]: ボックス（≤MAX_BOXES×4）とスコア（スコアの降順）．
    """
    boxes, scores = torch.as_tensor(np.asarray(boxes, dtype=np.float32)), torch.as_tensor(np.asarray(scores, dtype=np.float32))
    if len(scores) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes, scores = class_agnostic_nms(boxes, scores, NMS_IOU)
    order = np.argsort(-np.asarray(scores), kind="stable")[:MAX_BOXES]
    return np.asarray(boxes)[order], np.asarray(scores)[order]


class OWLv2Objectness:
    """OWLv2の物体らしさのスコアによる，クラス非依存の領域の候補の検出．"""

    def __init__(self, checkpoint="google/owlv2-base-patch16-ensemble", device=DEVICE):
        """
        Args:
            checkpoint (str): Hugging Face Hubのチェックポイント名．
            device (str): 推論デバイス．
        """
        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint).eval().to(device)
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        1枚の画像の領域の候補を検出する．OWLv2は画像を右下に0埋めした正方形として扱うため，ボックスを長辺の長さで復元する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックスとスコア．
        """
        inputs = self.processor(text=[["a photo of an object"]], images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        scores = out.objectness_logits[0].sigmoid().cpu().numpy()
        cx, cy, w, h = out.pred_boxes[0].cpu().numpy().T * max(image.size)
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = np.clip(boxes, 0, [image.size[0], image.size[1], image.size[0], image.size[1]])
        return top_boxes(boxes, scores)


class YOLOProposals:
    """YOLO11xの低い閾値の検出を，クラス非依存の領域の候補として用いる．"""

    def __init__(self, seed=0, conf=0.01):
        """
        Args:
            seed (int): 乱数シード．
            conf (float): 検出スコアの閾値．
        """
        self.detector = ex009_model().load_yolo(seed, conf)

    def detect(self, image):
        """
        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックスとスコア．
        """
        det = self.detector.detect(image)
        return top_boxes(det["boxes"], det["scores"])


class DETRProposals:
    """DETRの低い閾値の検出を，クラス非依存の領域の候補として用いる．"""

    def __init__(self, seed=0, threshold=0.01):
        """
        Args:
            seed (int): 乱数シード．
            threshold (float): 検出スコアの閾値．
        """
        set_seed(seed)
        self.detector = DETRDetector(threshold=threshold, device=DEVICE)

    def detect(self, image):
        """
        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックスとスコア．
        """
        return top_boxes(*self.detector.detect(image))


class GDINOProposals:
    """Grounding DINO（tiny）にテキスト「object.」を与えた検出を，クラス非依存の領域の候補として用いる．"""

    def __init__(self, box_threshold=0.05):
        """
        Args:
            box_threshold (float): ボックススコアの閾値．
        """
        self.detector = GroundingDINODetector(["object"], box_threshold=box_threshold, text_threshold=box_threshold,
                                              device=DEVICE)

    def detect(self, image):
        """
        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックスとスコア．
        """
        boxes, scores = self.detector.detect(image)
        return top_boxes(boxes, scores)


def load_detector(name, seed=0):
    """
    名前から領域の候補の検出器をインスタンス化する．

    Args:
        name (str): DETECTORS のいずれか．
        seed (int): 乱数シード．

    Returns:
        object: detect(image) -> (boxes, scores) を持つ検出器．
    """
    if name == "yolo_low":
        return YOLOProposals(seed)
    if name == "detr_low":
        return DETRProposals(seed)
    if name == "owlv2":
        return OWLv2Objectness()
    return GDINOProposals()
