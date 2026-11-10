"""
ex006のデータ定義：YOLOとCLIPを組み合わせた粒度可変アノテーション（ハイブリッド構成）の評価データ．

- coco500: ex005でYOLO11xの検出とGTを対応付けた評価インスタンス（seed=42でサンプリングしたCOCO val2017の500枚）
- proposed: 比較対象の提案手法（ex003，DETR + CLIP-ViT-B/32）の評価インスタンスと階層アノテーションの結果
- open_vocab，coco_qualitative: ex004の定性評価に用いた実画像6枚とCOCO画像（論文の図4，図7）
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_coco, load_image  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402

PROGRAM_DIR = os.path.join(os.path.dirname(__file__), "..")
EX003_OUTPUT = os.path.join(OUTPUT_DIR, "ex003_coco_val500_pipeline", "CLIP-ViT-B32")
EX005_OUTPUT = os.path.join(OUTPUT_DIR, "ex005_coco_val500_yolo_comparison")


def import_experiment_module(experiment, module):
    """
    他の実験ディレクトリのモジュール（data.py，model.py等）を，名前の衝突を避けて読み込む．

    Args:
        experiment (str): 実験ディレクトリ名（例: "ex005_coco_val500_yolo_comparison"）．
        module (str): モジュール名（例: "data"）．

    Returns:
        module: 読み込んだモジュール．
    """
    path = os.path.join(PROGRAM_DIR, experiment, f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"{experiment}_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def load_yolo_instances(yolo_name, conf, seed=42):
    """
    ex005で対応付けたYOLOの評価インスタンスを読み込む．

    Args:
        yolo_name (str): YOLOのモデル名（例: "yolo11x"）．
        conf (float): 検出スコアの閾値．
        seed (int): シード．

    Returns:
        list[dict]: {"image_id", "gt_index", "gt_name", "box"（4），"det_score", "top5"（5），"top5_conf"（5），...}．
    """
    with open(os.path.join(EX005_OUTPUT, yolo_name, f"coco500_conf{conf}", str(seed), "instances.json")) as f:
        return json.load(f)


def load_n_gt(yolo_name, conf, seed=42):
    """
    評価画像500枚のGTインスタンス数を，ex005の評価結果から読み込む．

    Args:
        yolo_name (str): YOLOのモデル名．
        conf (float): 検出スコアの閾値．
        seed (int): シード．

    Returns:
        int: GTインスタンス数の合計．
    """
    with open(os.path.join(EX005_OUTPUT, yolo_name, f"coco500_conf{conf}", str(seed), "metrics.json")) as f:
        return json.load(f)["n_gt"]


def load_images(image_ids):
    """
    COCO val2017の画像を読み込む．

    Args:
        image_ids (Iterable[int]): 画像IDの集合．

    Returns:
        dict: 画像ID → PIL.Image.Image（RGB）．
    """
    coco = load_coco("instances")
    return {image_id: load_image(coco, image_id) for image_id in sorted(set(image_ids))}


def load_proposed_chains(condition, n_images=500, seed=42):
    """
    提案手法（ex003，CLIP-ViT-B/32）の階層アノテーションの結果を，GTインスタンスをキーとして読み込む．

    GTの対応付けはex005の load_proposed_instances と同じ手順で再計算し，instances.json の並び順と一致させる．

    Args:
        condition (str): ex003の条件名（例: "detr_threshold0.97_crop_pad"）．
        n_images (int): 画像数．
        seed (int): シード．

    Returns:
        dict: (画像ID, GTインデックス) → {"gt_name", "top1_correct", "chain"（深さ0が正解のときのみ，それ以外はNone）}．
    """
    ex005_data = import_experiment_module("ex005_coco_val500_yolo_comparison", "data")
    keys = list(ex005_data.load_proposed_instances(condition, n_images, seed))
    with open(os.path.join(EX003_OUTPUT, condition, str(seed), "instances.json")) as f:
        records = json.load(f)
    return {key: {"gt_name": r["gt_name"], "top1_correct": r["top1_correct"], "chain": r.get("chain")}
            for key, r in zip(keys, records)}


def load_qualitative_samples():
    """
    ex004の定性評価に用いた画像を読み込む．

    Returns:
        dict: 画像集合の名前（"open_vocab"，"coco_qualitative"）→ 画像ごとの {"name", "image", "meta"} のリスト．
    """
    ex004_data = import_experiment_module("ex004_open_vocab_qualitative", "data")
    open_vocab, _ = ex004_data.load_open_vocab_images()
    coco, _ = ex004_data.load_coco_qualitative_images()
    return {"open_vocab": open_vocab, "coco_qualitative": coco}
