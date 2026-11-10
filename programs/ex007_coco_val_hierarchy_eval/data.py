"""
ex007のデータ定義：評価数で揃えた比較，誤りの伝播，累積確信度による細分化の停止，LVISによる深さ1以降の評価に用いるデータ．

画像の分割（COCO val2017の5000枚，いずれも重複しない）:
- dev500: ex003〜ex006と同じ，seed=42でサンプリングした500枚（τ等の決定に用いる検証用）
- test500: dev500以外の画像を2等分した後半（B）から，seed=42でサンプリングした500枚（テスト用）
- lvis_dev: dev500と前半（A）のうちLVIS v1 valのアノテーションが存在する画像（検証用）
- lvis_test: 後半（B）のうちLVIS v1 valのアノテーションが存在する画像（テスト用，test500を含む）

検出は，DETRは閾値0.5，YOLO11xは閾値0.25で全5000枚に対して1回だけ実行してキャッシュし，
より高い閾値の結果はスコアで絞り込んで得る（DETRはNMSを用いず，YOLOのNMSは貪欲法のため，絞り込みと再実行は一致する）．
"""
import importlib.util
import json
import os
import random
import sys

import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import gt_instances, load_coco, load_image, sample_image_ids  # noqa: E402
from common.metrics import box_iou, match_boxes  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402

PROGRAM_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(DATASET_DIR, "ex007_coco_val_hierarchy_eval")
LVIS_PATH = os.path.join(DATASET_DIR, "lvis", "lvis_v1_val.json")
DETR_BASE_THRESHOLD = 0.5
YOLO_BASE_CONF = 0.25


def import_experiment_module(experiment, module):
    """
    他の実験ディレクトリのモジュール（data.py，model.py等）を，名前の衝突を避けて読み込む．

    Args:
        experiment (str): 実験ディレクトリ名（例: "ex005_coco_val500_yolo_comparison"）．
        module (str): モジュール名（例: "model"）．

    Returns:
        module: 読み込んだモジュール．
    """
    path = os.path.join(PROGRAM_DIR, experiment, f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"{experiment}_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def make_splits(coco, lvis_image_ids, seed=42):
    """
    COCO val2017の画像を，重複しない検証用・テスト用に分割する．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        lvis_image_ids (set[int]): LVIS v1 valにアノテーションが存在する画像IDの集合．
        seed (int): 乱数シード．

    Returns:
        dict[str, list[int]]: 分割名（"dev500"，"test500"，"lvis_dev"，"lvis_test"）→ 画像IDのリスト．
    """
    dev500 = sample_image_ids(coco, 500, seed)
    rest = sorted(set(coco.getImgIds()) - set(dev500))
    random.Random(seed).shuffle(rest)
    half_a, half_b = sorted(rest[:len(rest) // 2]), sorted(rest[len(rest) // 2:])
    test500 = random.Random(seed).sample(half_b, 500)
    return {
        "dev500": dev500,
        "test500": test500,
        "lvis_dev": [i for i in sorted(dev500 + half_a) if i in lvis_image_ids],
        "lvis_test": [i for i in half_b if i in lvis_image_ids],
    }


def cached_json(path, compute):
    """
    JSONのキャッシュが存在すれば読み込み，存在しなければ計算して保存する．

    Args:
        path (str): キャッシュのパス．
        compute (callable): 引数なしでJSONに保存できる値を返す関数．

    Returns:
        object: キャッシュまたは計算結果．
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    value = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f)
    return value


def detect_detr(coco, image_ids, detector_factory):
    """
    全画像に対しDETR（閾値0.5）を1回ずつ実行し，結果をキャッシュする．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        image_ids (list[int]): 画像IDのリスト．
        detector_factory (callable): 引数なしでDETRDetectorを返す関数（キャッシュが無い場合のみ呼ぶ）．

    Returns:
        dict: 画像ID（str）→ {"boxes": K×4, "scores": K}．
    """
    def compute():
        detector = detector_factory()
        out = {}
        for image_id in tqdm(image_ids, desc="DETR", leave=False):
            boxes, scores = detector.detect(load_image(coco, image_id))
            out[str(image_id)] = {"boxes": boxes.tolist(), "scores": scores.tolist()}
        return out
    return cached_json(os.path.join(DATA_DIR, f"detr_threshold{DETR_BASE_THRESHOLD}_n{len(image_ids)}.json"), compute)


def detect_yolo(coco, image_ids, detector_factory):
    """
    全画像に対しYOLO11x（閾値0.25）を1回ずつ実行し，結果をキャッシュする．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        image_ids (list[int]): 画像IDのリスト．
        detector_factory (callable): 引数なしでYOLODetectorを返す関数（キャッシュが無い場合のみ呼ぶ）．

    Returns:
        dict: 画像ID（str）→ {"boxes": K×4, "scores": K, "top5": K×5, "top5_conf": K×5}．
    """
    def compute():
        detector = detector_factory()
        return {str(i): detector.detect(load_image(coco, i)) for i in tqdm(image_ids, desc="YOLO", leave=False)}
    return cached_json(os.path.join(DATA_DIR, f"yolo11x_conf{YOLO_BASE_CONF}_n{len(image_ids)}.json"), compute)


def load_lvis():
    """
    LVIS v1 valのアノテーションを画像ごとに読み込む．

    Returns:
        dict[int, list[dict]]: 画像ID → {"box"（xyxy，長さ4），"synset"（WordNetのSynset名）} のリスト．
    """
    with open(LVIS_PATH) as f:
        lvis = json.load(f)
    synset_of = {c["id"]: c["synset"] for c in lvis["categories"]}
    per_image = {}
    for a in lvis["annotations"]:
        x, y, w, h = a["bbox"]
        per_image.setdefault(a["image_id"], []).append({"box": [x, y, x + w, y + h], "synset": synset_of[a["category_id"]]})
    return per_image


def gt_path_synsets(coco_name, lvis_synset_name):
    """
    COCOのカテゴリのSynset以下，LVISのSynset以上にあるSynsetの集合（GTの経路）を求める．
    LVISのSynsetがCOCOのカテゴリのSynsetの真の下位語でない場合は None を返す．

    Args:
        coco_name (str): COCOのカテゴリ名．
        lvis_synset_name (str): LVISのカテゴリのSynset名．

    Returns:
        list[str] | None: 経路上のSynset名のリスト（COCOのSynsetとLVISのSynsetを含む）．
    """
    try:
        target = wn.synset(lvis_synset_name)
    except Exception:  # WordNet 3.0に存在しないSynset名
        return None
    root = wn.synset(COCO_TO_SYNSET[coco_name])
    if target == root:
        return None
    nodes = set()
    for path in target.hypernym_paths():
        if root in path:
            nodes.update(path[path.index(root):])
    return sorted(s.name() for s in nodes) if nodes else None


def attach_lvis(instance, lvis_anns, iou_threshold=0.5):
    """
    インスタンスに，検出ボックスとIoUが最大で，COCOのGTカテゴリより細かいLVISのアノテーションを付与する．

    Args:
        instance (dict): "box"，"gt_name" を持つインスタンス（"lvis_synset"，"gt_path" を追記する）．
        lvis_anns (list[dict]): 同じ画像のLVISのアノテーション．
        iou_threshold (float): 対応とみなすIoUの下限．

    Returns:
        bool: 付与できたか．
    """
    candidates = [(a, gt_path_synsets(instance["gt_name"], a["synset"])) for a in lvis_anns]
    candidates = [(a, p) for a, p in candidates if p]
    if not candidates:
        return False
    ious = box_iou(np.array([instance["box"]]), np.array([a["box"] for a, _ in candidates]))[0]
    best = int(np.argmax(ious))
    if ious[best] < iou_threshold:
        return False
    instance["lvis_synset"], instance["gt_path"] = candidates[best][0]["synset"], candidates[best][1]
    return True


def build_instances(coco, image_ids, detections, min_score, iou_threshold=0.5):
    """
    スコアが min_score を超える検出ボックスとCOCOのGTを対応付け，評価インスタンスを作成する．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        image_ids (list[int]): 画像IDのリスト．
        detections (dict): detect_detr / detect_yolo の戻り値．
        min_score (float): 検出スコアの閾値．
        iou_threshold (float): 対応付けのIoU閾値．

    Returns:
        tuple[list[dict], int]: インスタンス {"image_id", "gt_index", "gt_name", "box", "det_score", ("yolo_word")}
            のリストと，GTインスタンスの総数．
    """
    instances, n_gt = [], 0
    for image_id in image_ids:
        det = detections[str(image_id)]
        keep = [i for i, s in enumerate(det["scores"]) if s > min_score]  # DETR，YOLOとも閾値を超えるボックスを残す
        gt_boxes, gt_names = gt_instances(coco, image_id)
        n_gt += len(gt_names)
        boxes = np.array([det["boxes"][i] for i in keep], dtype=np.float32).reshape(-1, 4)
        for p, g, iou in match_boxes(boxes, gt_boxes, iou_threshold):
            k = keep[p]
            ins = {"image_id": image_id, "gt_index": g, "gt_name": gt_names[g], "box": det["boxes"][k],
                   "det_score": det["scores"][k], "iou": iou}
            if "top5" in det:
                ins["yolo_word"] = det["top5"][k][0]
            instances.append(ins)
    return instances, n_gt
