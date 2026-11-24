"""
ex021：深さ0の棄却の判定（重複・部分の除去と棄却のスコア）（order_033）．

検出器と F は ex019 のもの（OWLv2 objectness，Alpha-CLIP）であり，新たに学習するモデルは無い．
棄却のスコア（1つのボックスの「残す確からしさ」）：
- joint：VLM の確信度（COCO の80クラスの softmax の最大値）× 物体らしさ（ex019 の新しい提案手法）
- objectness：物体らしさのみ
- vlm：VLM の確信度のみ
- mass：VG の木で COCO の各クラスの部分木の確率の合計（VG.pdf の式(7)の m_u，T = 1）の最大値．
  確率が人物の多くの下位語（man，player など）に分散する場合でも，人物の部分木に集まる．
- mass_joint：mass × 物体らしさ
重複・部分の除去は，棄却のスコアの降順に，同じ深さ0のラベルの先のボックスと重なるボックスを除く．
"""
import numpy as np

SCORE_TYPES = ["joint", "objectness", "vlm", "mass", "mass_joint"]
DEDUP_TYPES = ["none", "nms", "nms_contain"]
NMS_IOU = 0.5           # クラスごとの NMS の IoU の下限
CONTAIN_RATIO = 0.8     # 包含の判定：小さいボックスの面積のうち大きいボックスに含まれる割合の下限


def rejection_score(score_type, conf, objectness, mass):
    """
    棄却のスコアを求める．

    Args:
        score_type (str): SCORE_TYPES のいずれか．
        conf (np.ndarray): VLM の確信度（K）．
        objectness (np.ndarray): 検出スコア（K）．
        mass (np.ndarray): COCO のクラスの部分木の確率の合計の最大値（K）．

    Returns:
        np.ndarray: スコア（K）．
    """
    return {"joint": conf * objectness, "objectness": objectness, "vlm": conf, "mass": mass,
            "mass_joint": mass * objectness}[score_type]


def pairwise_overlap(boxes):
    """
    ボックスの組ごとの IoU と，行のボックスの面積のうち列のボックスに含まれる割合を求める．

    Args:
        boxes (np.ndarray): xyxy形式のボックス（K×4）．

    Returns:
        tuple[np.ndarray, np.ndarray]: IoU（K×K）と包含の割合（K×K）．
    """
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-9), inter / np.maximum(area[:, None], 1e-9)


def dedup_mask(boxes, labels, scores, dedup):
    """
    重複・部分のボックスを除く（スコアの降順に，同じラベルで先に残したボックスと重なるものを除く）．

    Args:
        boxes (np.ndarray): ボックス（K×4）．
        labels (list[str]): 深さ0のラベル（長さK）．
        scores (np.ndarray): 棄却のスコア（K）．
        dedup (str): DEDUP_TYPES のいずれか．

    Returns:
        np.ndarray: 残すかどうか（K，bool）．
    """
    keep = np.ones(len(boxes), bool)
    if dedup == "none" or len(boxes) < 2:
        return keep
    iou, contain = pairwise_overlap(np.asarray(boxes, np.float64))
    labels = np.asarray(labels)
    kept = []
    for k in np.argsort(-scores, kind="stable"):
        same = [j for j in kept if labels[j] == labels[k]]
        hit = any(iou[k, j] >= NMS_IOU for j in same)
        if dedup == "nms_contain":
            hit = hit or any(contain[k, j] >= CONTAIN_RATIO for j in same)
        if hit:
            keep[k] = False
        else:
            kept.append(k)
    return keep
