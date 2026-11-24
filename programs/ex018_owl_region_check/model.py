"""
ex018：OWLv2による領域の検出と，検出結果どうしの重なりの計算（order_028）．

- OWLv2（google/owlv2-base-patch16-ensemble）の1回の推論から，2種類の検出結果を得る．
  - "query"：COCOの80クラスのテキストクエリに対するスコア（クエリ間の最大値）
  - "objectness"：クラスに依存しない物体らしさのスコア（新しいパイプラインの検出器の候補）
- いずれもクラス非依存の NMS（IoU 0.5）の後，スコアの上位 MAX_BOXES 件を低い閾値で保存し，閾値は評価時に変える．
"""
import numpy as np
import torch
import torchvision
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "a photo of a {}"
MAX_BOXES = 300        # 1枚あたりに保存するボックスの上限
MIN_SCORE = 0.001      # 保存するスコアの下限
NMS_IOU = 0.5


def class_agnostic_nms(boxes, scores, iou_threshold=NMS_IOU):
    """
    クラスを区別しない NMS で重複するボックスを除き，スコアの上位 MAX_BOXES 件を返す．

    Args:
        boxes (torch.Tensor): xyxy形式のボックス（K×4）．
        scores (torch.Tensor): スコア（K）．
        iou_threshold (float): 重複とみなす IoU．

    Returns:
        tuple[np.ndarray, np.ndarray]: ボックス（K'×4，float32）とスコア（K'），スコアの降順．
    """
    mask = scores >= MIN_SCORE
    boxes, scores = boxes[mask].float().cpu(), scores[mask].float().cpu()
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)[:MAX_BOXES]
    return boxes[keep].numpy(), scores[keep].numpy()


class OWLv2Detector:
    """OWLv2 のテキストクエリのスコアと物体らしさのスコアによる検出．"""

    def __init__(self, queries, checkpoint="google/owlv2-base-patch16-ensemble", device=DEVICE):
        """
        Args:
            queries (list[str]): テキストクエリとするカテゴリ名．
            checkpoint (str): Hugging Face Hub のチェックポイント名．
            device (str): 推論デバイス．
        """
        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint).eval().to(device)
        self.queries = [PROMPT.format(q) for q in queries]
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        1回の推論で，クエリのスコアと物体らしさのスコアによる検出結果を求める．
        OWLv2は画像を右下に0埋めした正方形として扱うため，ボックスを長辺の長さで復元する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict[str, dict]: "query" と "objectness" → {"boxes"（K×4，xyxy），"scores"（K）}．
        """
        inputs = self.processor(text=[self.queries], images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        cx, cy, w, h = (out.pred_boxes[0] * max(image.size)).T
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        limit = torch.tensor([image.size[0], image.size[1]] * 2, device=boxes.device, dtype=boxes.dtype)
        boxes = torch.minimum(boxes.clamp(min=0), limit)
        result = {}
        for kind, scores in (("query", out.logits[0].sigmoid().max(dim=-1).values),
                             ("objectness", out.objectness_logits[0].sigmoid())):
            b, s = class_agnostic_nms(boxes, scores)
            result[kind] = {"boxes": b, "scores": s}
        return result


def box_iou(a, b):
    """
    2組のボックスの IoU 行列を求める．

    Args:
        a (np.ndarray): xyxy形式のボックス（N×4）．
        b (np.ndarray): xyxy形式のボックス（M×4）．

    Returns:
        np.ndarray: IoU（N×M）．どちらかが空の場合は形状 (N, M) の0行列．
    """
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    return torchvision.ops.box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()


def covered(a, b, iou_threshold=0.5):
    """
    a の各ボックスに，b のいずれかのボックスが IoU 閾値以上で重なるかを判定する（提案領域の再現率の判定）．

    Args:
        a (np.ndarray): 判定するボックス（N×4）．
        b (np.ndarray): 重なりを探すボックス（M×4）．
        iou_threshold (float): IoU の閾値．

    Returns:
        np.ndarray: bool（N）．
    """
    iou = box_iou(a, b)
    if iou.shape[1] == 0:
        return np.zeros(len(iou), bool)
    return iou.max(axis=1) >= iou_threshold
