"""評価指標（Precision/Recall@k，IoU，ボックスの対応付け，統計的検定）．"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon


def precision_recall_at_k(ranked_words, gt_words, k):
    """
    上位k語に対するPrecisionとRecallを計算する．

    Args:
        ranked_words (list[str]): スコアの降順に並べた単語のリスト．
        gt_words (set[str]): 正解単語の集合（空でないこと）．
        k (int): 上位語数．

    Returns:
        tuple[float, float]: (Precision@k, Recall@k)．
    """
    hit = len(set(ranked_words[:k]) & gt_words)
    return hit / k, hit / len(gt_words)


def box_iou(boxes_a, boxes_b):
    """
    2つのボックス集合のIoU行列を計算する．

    Args:
        boxes_a (np.ndarray): xyxy形式のボックス（N×4）．
        boxes_b (np.ndarray): xyxy形式のボックス（M×4）．

    Returns:
        np.ndarray: IoU行列（N×M）．
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(1, -1, 4)
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    予測ボックスとGTボックスをIoU最大となるよう1対1で対応付ける（ハンガリアン法）．

    Args:
        pred_boxes (np.ndarray): 予測ボックス（N×4）．
        gt_boxes (np.ndarray): GTボックス（M×4）．
        iou_threshold (float): 対応とみなすIoUの下限．

    Returns:
        list[tuple[int, int, float]]: (予測インデックス, GTインデックス, IoU) のリスト．
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    iou = box_iou(pred_boxes, gt_boxes)
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]


def bootstrap_ci(values, n_bootstrap=1000, seed=0, alpha=0.05):
    """
    画像単位の指標値の平均に対するブートストラップ信頼区間を計算する．

    Args:
        values (array-like): 画像単位の指標値（長さN）．
        n_bootstrap (int): リサンプリング回数B．
        seed (int): 乱数シード．
        alpha (float): 有意水準（0.05で95%信頼区間）．

    Returns:
        tuple[float, float, float]: (平均, 下限, 上限)．
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(values_a, values_b):
    """
    対応のある2標本に対するWilcoxon符号付き順位検定のp値を計算する．

    Args:
        values_a (array-like): 手法Aの画像単位の指標値（長さN）．
        values_b (array-like): 手法Bの画像単位の指標値（長さN）．

    Returns:
        float: 両側検定のp値（全差分が0の場合は1.0）．
    """
    diff = np.asarray(values_a) - np.asarray(values_b)
    if np.all(diff == 0):
        return 1.0
    return float(wilcoxon(values_a, values_b).pvalue)
