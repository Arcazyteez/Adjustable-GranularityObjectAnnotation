"""
ex003のデータ定義：DETRの検出ボックスとGTボックスの対応付けによる評価インスタンスの作成（論文4.4節）．

seed=42でサンプリングした500枚に対しDETRを1回だけ実行し，IoU 0.5以上で1対1に対応付けられた
（検出ボックス，GTカテゴリ）の組を評価インスタンスとする．検出結果は閾値ごとにdatasets/以下へキャッシュする．
"""
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, gt_instances, load_coco, load_image, make_crop, sample_image_ids  # noqa: E402
from common.detectors import DETRDetector  # noqa: E402
from common.metrics import match_boxes  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402

DATA_DIR = os.path.join(DATASET_DIR, "ex003_coco_val500_pipeline")


def detect_all(coco, image_ids, threshold, device="cuda"):
    """
    全画像に対しDETRを1回ずつ実行し，結果をキャッシュする．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        image_ids (list[int]): 画像IDのリスト．
        threshold (float): DETRの検出スコア閾値．
        device (str): 推論デバイス．

    Returns:
        dict: 画像ID（str）→ {"boxes": list[list[float]]（K×4）, "scores": list[float]（K）}．
    """
    path = os.path.join(DATA_DIR, f"detr_threshold{threshold}_n{len(image_ids)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    detector = DETRDetector(threshold=threshold, device=device)
    detections = {}
    for image_id in tqdm(image_ids, desc=f"DETR th={threshold}", leave=False):
        boxes, scores = detector.detect(load_image(coco, image_id))
        detections[str(image_id)] = {"boxes": boxes.tolist(), "scores": scores.tolist()}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(detections, f)
    return detections


def build_instances(threshold, crop_mode="pad", n_images=500, seed=42, iou_threshold=0.5, device="cuda"):
    """
    検出ボックスとGTを対応付け，評価インスタンス（クロップとGTカテゴリ）を作成する．

    Args:
        threshold (float): DETRの検出スコア閾値．
        crop_mode (str): クロップの前処理（"center" または "pad"）．
        n_images (int): サンプリングする画像数．
        seed (int): サンプリングの乱数シード．
        iou_threshold (float): 対応付けのIoU閾値．
        device (str): 推論デバイス．

    Returns:
        tuple[list[dict], list[str], dict]: インスタンスのリスト
            {"image_id", "box", "det_score", "iou", "gt_name", "crop"}，COCO 80クラス名，
            統計 {"n_detections", "n_gt"}．
    """
    coco = load_coco("instances")
    image_ids = sample_image_ids(coco, n_images, seed)
    detections = detect_all(coco, image_ids, threshold, device)
    instances, n_det, n_gt = [], 0, 0
    for image_id in tqdm(image_ids, desc="match & crop", leave=False):
        det = detections[str(image_id)]
        gt_boxes, gt_names = gt_instances(coco, image_id)
        n_det, n_gt = n_det + len(det["boxes"]), n_gt + len(gt_names)
        matches = match_boxes(det["boxes"], gt_boxes, iou_threshold)
        if not matches:
            continue
        image = load_image(coco, image_id)
        for p, g, iou in matches:
            box = det["boxes"][p]
            instances.append({"image_id": image_id, "box": box, "det_score": det["scores"][p], "iou": iou,
                              "gt_name": gt_names[g], "crop": make_crop(image, box, crop_mode)})
    return instances, category_names(coco), {"n_detections": n_det, "n_gt": n_gt}
