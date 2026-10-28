"""
ex005のモデル定義：比較対象の閉域物体検出モデルYOLO（Ultralytics）．

YOLOは検出と同時にCOCOの80クラス（YOLO自体の語彙）のいずれかを出力する．
Top-5正解率を計算するため，NMSで残った各ボックスの80クラスのスコアを生出力から取り出す．
"""
import contextlib
import os
import sys

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import predict as yolo_predict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import DATASET_DIR, set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402

WEIGHT_DIR = os.path.join(DATASET_DIR, "ex005_coco_val500_yolo_comparison", "weights")


@contextlib.contextmanager
def capture_class_scores(stash):
    """
    YOLOのNMSを一時的に置き換え，残ったボックスの全クラスのスコアを stash に追加する．

    Args:
        stash (list): 画像ごとのクラススコア（np.ndarray，K×80）を追加するリスト．

    Returns:
        contextlib の文脈（with文で用いる）．
    """
    original = yolo_predict.ops.non_max_suppression

    def wrapped(preds, *args, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        kwargs["return_idxs"] = True
        output, keep = original(preds, *args, **kwargs)
        n_classes = raw.shape[1] - 4
        for b, idx in enumerate(keep):
            # 検出が0件の場合，keep は浮動小数点型の空テンソルとなるため整数型に変換する
            stash.append(raw[b, 4:4 + n_classes, idx.long().reshape(-1)].T.float().cpu().numpy())
        return output

    yolo_predict.ops.non_max_suppression = wrapped
    try:
        yield
    finally:
        yolo_predict.ops.non_max_suppression = original


class YOLODetector:
    """COCOの80クラスで学習済みのYOLOによる検出とカテゴリ予測．"""

    def __init__(self, name="yolo11x", conf=0.25, iou=0.7, device="cuda"):
        """
        モデルを読み込む（重みは初回に自動ダウンロードされる）．

        Args:
            name (str): モデル名（"yolo11n"，"yolo11m"，"yolo11x"，"yolov8x" 等）．
            conf (float): 検出スコアの閾値．
            iou (float): NMSのIoU閾値（Ultralyticsの既定値0.7）．
            device (str): 推論デバイス．
        """
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        self.model = YOLO(os.path.join(WEIGHT_DIR, f"{name}.pt"))
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        self.name, self.conf, self.iou, self.device = name, conf, iou, device

    @torch.no_grad()
    def detect(self, image):
        """
        1回の推論でボックス・検出スコア・クラスの上位5件を出力する．

        Args:
            image (PIL.Image.Image): 入力画像（RGB）．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"top5"（K×5のクラス名），"top5_conf"（K×5）．
        """
        stash = []
        with capture_class_scores(stash):
            result = self.model.predict(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        class_scores = stash[0] if stash else np.zeros((0, len(self.names)), np.float32)
        order = np.argsort(-class_scores, axis=1, kind="stable")[:, :5]
        return {
            "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
            "scores": result.boxes.conf.cpu().numpy().tolist(),
            "top5": [[self.names[j] for j in row] for row in order],
            "top5_conf": np.take_along_axis(class_scores, order, axis=1).tolist(),
        }


def load_model(name, conf, seed=0):
    """
    学習済みのYOLOをインスタンス化する．

    Args:
        name (str): YOLOのモデル名．
        conf (float): 検出スコアの閾値．
        seed (int): 乱数シード．

    Returns:
        YOLODetector: detect(image) を持つ検出器．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return YOLODetector(name, conf, device="cuda" if torch.cuda.is_available() else "cpu")


def load_vlm_model(name="CLIP-ViT-B32", seed=0):
    """
    YOLOのボックスを分類するためのVLMをインスタンス化する．

    Args:
        name (str): VLMの名前．
        seed (int): 乱数シード．

    Returns:
        CLIPModel: score(images, vocab) を持つVLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return load_vlm(name, "cuda" if torch.cuda.is_available() else "cpu")
