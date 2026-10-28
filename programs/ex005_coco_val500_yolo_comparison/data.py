"""
ex005のデータ定義：YOLOとの比較に用いる画像集合と，提案手法（ex003）の評価インスタンスのGT対応付け．

- coco500: ex003と同じ，seed=42でサンプリングしたCOCO val2017の500枚
- loc4: ex002で選定した4画像（GTインスタンス10件）
- open_vocab: ex004で取得したCOCO外の語を含む実画像6枚
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import gt_instances, load_coco, load_image, sample_image_ids  # noqa: E402
from common.metrics import match_boxes  # noqa: E402
from common.paths import DATASET_DIR, OUTPUT_DIR  # noqa: E402

EX002_SELECTED = os.path.join(DATASET_DIR, "ex002_coco_4img_class_agnostic_det", "selected_seed{seed}.json")
EX003_DETR_CACHE = os.path.join(DATASET_DIR, "ex003_coco_val500_pipeline", "detr_threshold{threshold}_n{n}.json")
EX003_OUTPUT = os.path.join(OUTPUT_DIR, "ex003_coco_val500_pipeline")
EX004_DATA_DIR = os.path.join(DATASET_DIR, "ex004_open_vocab_qualitative")
OPEN_VOCAB_NAMES = ["penguin", "helicopter", "camel", "kangaroo", "lighthouse_1", "lighthouse_2"]


def load_coco_samples(n_images=500, seed=42):
    """
    COCOの評価画像とGTを読み込む．

    Args:
        n_images (int): サンプリングする画像数．
        seed (int): サンプリングの乱数シード．

    Returns:
        list[dict]: 画像ごとの {"image_id", "image", "gt_boxes"（K×4）, "gt_names"（K）}．
    """
    coco = load_coco("instances")
    samples = []
    for image_id in sample_image_ids(coco, n_images, seed):
        boxes, names = gt_instances(coco, image_id)
        samples.append({"image_id": image_id, "image": load_image(coco, image_id), "gt_boxes": boxes, "gt_names": names})
    return samples


def load_loc4_samples(seed=42):
    """
    ex002で選定した4画像とGTを読み込む．

    Args:
        seed (int): ex002の選定に用いたシード．

    Returns:
        list[dict]: 画像ごとの {"image_id", "image", "gt_boxes", "gt_names"}．
    """
    with open(EX002_SELECTED.format(seed=seed)) as f:
        image_ids = json.load(f)
    coco = load_coco("instances")
    samples = []
    for image_id in image_ids:
        boxes, names = gt_instances(coco, image_id)
        samples.append({"image_id": image_id, "image": load_image(coco, image_id), "gt_boxes": boxes, "gt_names": names})
    return samples


def load_open_vocab_samples():
    """
    ex004で取得したCOCO外の語を含む実画像を読み込む．

    Returns:
        list[dict]: 画像ごとの {"name", "image"}．
    """
    return [{"name": name, "image": Image.open(os.path.join(EX004_DATA_DIR, f"{name}.jpg")).convert("RGB")}
            for name in OPEN_VOCAB_NAMES]


def load_proposed_instances(condition, n_images=500, seed=42, iou_threshold=0.5):
    """
    提案手法（ex003，CLIP-ViT-B/32）の評価インスタンスに，対応するGTのインデックスを付与して読み込む．

    ex003と同じ手順（DETRの検出キャッシュとハンガリアン法）で対応付けを再計算し，
    instances.json の並び順と一致させる．

    Args:
        condition (str): ex003の条件名（例: "detr_threshold0.97_crop_pad"）．
        n_images (int): 画像数．
        seed (int): シード．
        iou_threshold (float): 対応付けのIoU閾値．

    Returns:
        dict: (画像ID, GTインデックス) → {"gt_name", "pred", "top1_correct", "top5_correct"}．
    """
    threshold = condition.split("detr_threshold")[1].split("_")[0]
    with open(EX003_DETR_CACHE.format(threshold=threshold, n=n_images)) as f:
        detections = json.load(f)
    with open(os.path.join(EX003_OUTPUT, "CLIP-ViT-B32", condition, str(seed), "instances.json")) as f:
        records = json.load(f)
    coco = load_coco("instances")
    keys = []
    for image_id in sample_image_ids(coco, n_images, seed):
        gt_boxes, _ = gt_instances(coco, image_id)
        keys += [(image_id, g) for _, g, _ in match_boxes(detections[str(image_id)]["boxes"], gt_boxes, iou_threshold)]
    assert len(keys) == len(records), "ex003の評価インスタンスと対応付けの数が一致しない"
    return {key: {"gt_name": r["gt_name"], "pred": r["top5"][0], "top1_correct": r["top1_correct"],
                  "top5_correct": r["top5_correct"]}
            for key, r in zip(keys, records)}
