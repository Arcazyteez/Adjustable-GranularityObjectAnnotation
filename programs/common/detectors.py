"""
物体領域検出器（クラス非依存のボックスのみを出力する）．

- DETRDetector: COCO学習済みDETR（ResNet-50）．予測カテゴリは使わず，スコアが閾値以上のボックスのみ返す．
- OWLViTDetector / GroundingDINODetector: テキストクエリを入力とするオープンボキャブラリ検出器（比較対象）．
  予測ラベルは捨て，クラス間でNMSをかけてクラス非依存のボックス集合とする．
"""
import numpy as np
import torch
from torchvision.ops import nms
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, DetrForObjectDetection,
                          DetrImageProcessor, OwlViTForObjectDetection, OwlViTProcessor)


def class_agnostic_nms(boxes, scores, iou_threshold=0.5):
    """
    ラベルを無視したNMSで重複ボックスを除去する．

    Args:
        boxes (torch.Tensor): xyxy形式のボックス（K×4）．
        scores (torch.Tensor): スコア（K）．
        iou_threshold (float): NMSのIoU閾値．

    Returns:
        tuple[np.ndarray, np.ndarray]: 残ったボックス（K'×4）とスコア（K'）．
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    keep = nms(boxes.float(), scores.float(), iou_threshold)
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()


class DETRDetector:
    """カテゴリに依存しないボックス検出器として用いるDETR．"""

    name = "DETR"

    def __init__(self, threshold=0.9, device="cuda", checkpoint="facebook/detr-resnet-50"):
        """
        モデルを読み込む．

        Args:
            threshold (float): 検出スコアの閾値．
            device (str): 推論デバイス．
            checkpoint (str): Hugging Face Hubのチェックポイント名．
        """
        self.processor = DetrImageProcessor.from_pretrained(checkpoint, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(checkpoint, revision="no_timm").eval().to(device)
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        1回の推論でボックスを検出する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックス（K×4）とスコア（K）．
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[image.size[::-1]])[0]
        order = result["scores"].argsort(descending=True)
        return result["boxes"][order].cpu().numpy(), result["scores"][order].cpu().numpy()


class OWLViTDetector:
    """テキストクエリを用いるOWL-ViT（比較対象）．"""

    name = "OWL-ViT"

    def __init__(self, queries, threshold=0.1, device="cuda", checkpoint="google/owlvit-base-patch32"):
        """
        モデルを読み込む．

        Args:
            queries (list[str]): テキストクエリとするカテゴリ名．
            threshold (float): 検出スコアの閾値．
            device (str): 推論デバイス．
            checkpoint (str): Hugging Face Hubのチェックポイント名．
        """
        self.processor = OwlViTProcessor.from_pretrained(checkpoint)
        self.model = OwlViTForObjectDetection.from_pretrained(checkpoint).eval().to(device)
        self.queries = [f"a photo of a {q}" for q in queries]
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        テキストクエリを与えて検出し，クラス非依存のボックスに変換する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックス（K×4）とスコア（K）．
        """
        inputs = self.processor(text=[self.queries], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=torch.tensor([image.size[::-1]]))[0]
        return class_agnostic_nms(result["boxes"], result["scores"])


class GroundingDINODetector:
    """テキストクエリを用いるGrounding DINO（比較対象）．"""

    name = "Grounding DINO"

    def __init__(self, queries, box_threshold=0.35, text_threshold=0.25, chunk_size=40, device="cuda",
                 checkpoint="IDEA-Research/grounding-dino-tiny"):
        """
        モデルを読み込む．

        Args:
            queries (list[str]): テキストクエリとするカテゴリ名．
            box_threshold (float): ボックススコアの閾値．
            text_threshold (float): テキスト対応スコアの閾値．
            chunk_size (int): 1回の推論に含めるカテゴリ数（テキスト長の上限256トークン対策）．
            device (str): 推論デバイス．
            checkpoint (str): Hugging Face Hubのチェックポイント名．
        """
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).eval().to(device)
        self.prompts = [". ".join(queries[i:i + chunk_size]) + "." for i in range(0, len(queries), chunk_size)]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        カテゴリのチャンクごとに検出し，結果を統合してクラス非依存のボックスに変換する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[np.ndarray, np.ndarray]: xyxy形式のボックス（K×4）とスコア（K）．
        """
        boxes, scores = [], []
        for prompt in self.prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]])[0]
            boxes.append(result["boxes"])
            scores.append(result["scores"])
        return class_agnostic_nms(torch.cat(boxes), torch.cat(scores))
