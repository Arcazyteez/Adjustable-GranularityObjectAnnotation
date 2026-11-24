"""
ex012のデータ定義：進捗報告（2026B）の粒度可変アノテーションの評価に用いるデータ．

- 画像: COCO 2017 val の5000枚を seed で並べ替え，前半2500枚を検証用（dev，θ の選択），後半2500枚をテスト用（test，報告）とする
- 検出結果: 物体認識モデル D（YOLO11x，DETR）を全5000枚に1回ずつ適用し，ボックス・スコア・クラスをキャッシュする
- 定性評価の画像: テスト用の画像のうち，GTの物体が2個以上6個以下の画像から seed で選ぶ
"""
import json
import os
import random
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, gt_instances, load_coco, load_image  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402

DATA_DIR = os.path.join(DATASET_DIR, "ex012_wordnet_granular_annotation")
SPLITS = ["dev", "test"]


def load_dataset():
    """
    COCO 2017 val のアノテーションとクラス名を読み込む．

    Returns:
        tuple[COCO, list[str]]: COCOオブジェクトとクラス名のリスト（80）．
    """
    coco = load_coco("instances")
    return coco, category_names(coco)


def make_splits(coco, seed=42):
    """
    5000枚を検証用とテスト用に2等分する．

    Args:
        coco (COCO): COCOオブジェクト．
        seed (int): 乱数シード．

    Returns:
        dict[str, list[int]]: "dev"，"test" → 画像IDのリスト（昇順）．
    """
    ids = sorted(coco.getImgIds())
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    return {"dev": sorted(ids[:half]), "test": sorted(ids[half:])}


def cached_detections(name, coco, image_ids, detector_factory):
    """
    物体認識モデルを全画像に1回ずつ適用し，結果をキャッシュする．

    Args:
        name (str): 物体認識モデルの名前（"yolo" または "detr"）．
        coco (COCO): COCOオブジェクト．
        image_ids (list[int]): 画像ID．
        detector_factory (callable): キャッシュが無い場合に detect(image) を持つ検出器を返す関数．

    Returns:
        dict: 画像ID（str）→ {"boxes"（K×4，xyxy），"scores"（K），"labels"（K，COCOのクラス名）}．
    """
    path = os.path.join(DATA_DIR, f"{name}_val2017_n{len(image_ids)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    detector = detector_factory()
    out = {str(i): detector.detect(load_image(coco, i)) for i in tqdm(image_ids, desc=name, leave=False)}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def ground_truth(coco, image_ids):
    """
    画像ごとのGT（iscrowd=0）を返す．

    Args:
        coco (COCO): COCOオブジェクト．
        image_ids (list[int]): 画像ID．

    Returns:
        dict[str, tuple[np.ndarray, list[str]]]: 画像ID → (ボックス（K×4，xyxy），クラス名のリスト)．
    """
    return {str(i): gt_instances(coco, i) for i in image_ids}


def qualitative_images(coco, image_ids, n=6, seed=42):
    """
    定性評価に用いる画像を選ぶ．

    Args:
        coco (COCO): COCOオブジェクト．
        image_ids (list[int]): 候補の画像ID（テスト用）．
        n (int): 選ぶ枚数．
        seed (int): 乱数シード．

    Returns:
        list[int]: 画像ID．
    """
    candidates = [i for i in image_ids if 2 <= len(gt_instances(coco, i)[1]) <= 6]
    return sorted(random.Random(seed).sample(candidates, n))
