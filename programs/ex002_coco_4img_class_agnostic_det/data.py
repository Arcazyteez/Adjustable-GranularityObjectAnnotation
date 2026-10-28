"""
ex002のデータ定義：位置検出精度の比較に用いる4画像（GTインスタンス計10件）の選定（論文4.3節）．

論文では画像の選び方が明記されていないため，seed=42でサンプリングした500枚の順序で
GT（iscrowd=0）が2〜3件の画像を候補とし，先頭から見て合計がちょうど10件となる最初の4枚の組を採用する．
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, gt_instances, load_coco, load_image, sample_image_ids  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402

DATA_DIR = os.path.join(DATASET_DIR, "ex002_coco_4img_class_agnostic_det")


def select_images(coco, n_select=4, n_instances=10, seed=42, n_pool=500):
    """
    合計GTインスタンス数が n_instances となる n_select 枚の画像を決定的に選ぶ．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        n_select (int): 選ぶ画像数．
        n_instances (int): GTインスタンス数の合計．
        seed (int): サンプリングの乱数シード．
        n_pool (int): サンプリングする画像数．

    Returns:
        list[int]: 選ばれた画像IDのリスト（長さ n_select）．
    """
    pool = [i for i in sample_image_ids(coco, n_pool, seed) if 2 <= len(gt_instances(coco, i)[1]) <= 3]
    for combo in itertools.combinations(pool[:20], n_select):
        if sum(len(gt_instances(coco, i)[1]) for i in combo) == n_instances:
            return list(combo)
    raise RuntimeError("条件を満たす画像の組が見つからない")


def load_dataset(seed=42):
    """
    評価用の画像とGTを読み込む（選定結果はdatasets/以下にJSONで保存する）．

    Args:
        seed (int): サンプリングの乱数シード．

    Returns:
        tuple[list[dict], list[str]]: 画像ごとの {"image_id", "image", "gt_boxes", "gt_names"} のリストと，
            テキストクエリに用いるCOCO 80クラス名．
    """
    coco = load_coco("instances")
    path = os.path.join(DATA_DIR, f"selected_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            image_ids = json.load(f)
    else:
        image_ids = select_images(coco, seed=seed)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(image_ids, f)
    samples = []
    for image_id in image_ids:
        boxes, names = gt_instances(coco, image_id)
        samples.append({"image_id": image_id, "image": load_image(coco, image_id), "gt_boxes": boxes,
                        "gt_names": names})
    return samples, category_names(coco)
