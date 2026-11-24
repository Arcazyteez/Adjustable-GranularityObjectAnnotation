"""
ex009のデータ定義：COCO 2017 valでの提案手法（DETR + SigLIP + WordNet），ハイブリッド手法（YOLO11x + SigLIP + WordNet），
比較対象（YOLO11x単体）の評価データ．

画像の分割，DETR（閾値0.5）とYOLO11x（閾値0.25）の5000枚の検出結果，LVISのアノテーションはex007と同じものを用いる．
- dev500，lvis_dev: 検証用（累積確信度の閾値 τ の決定）
- test500，lvis_test: テスト用（結果の報告）
"""
import functools
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_coco  # noqa: E402

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS = ["dev500", "test500", "lvis_dev", "lvis_test"]


def import_experiment_module(experiment, module):
    """
    他の実験ディレクトリのモジュールを，名前の衝突を避けて読み込む．

    Args:
        experiment (str): 実験ディレクトリ名．
        module (str): モジュール名（例: "data"）．

    Returns:
        module: 読み込んだモジュール．
    """
    path = os.path.join(PROGRAM_DIR, experiment, f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"{experiment}_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.lru_cache(maxsize=None)
def ex007_data():
    """
    ex007の data モジュール（分割，検出のキャッシュ，LVISの読み込みと対応付け）を読み込む．

    Returns:
        module: ex007の data．
    """
    return import_experiment_module("ex007_coco_val_hierarchy_eval", "data")


def load_context(seed, detr_factory, yolo_factory):
    """
    COCO，LVIS，画像の分割，検出結果をまとめて読み込む．

    Args:
        seed (int): 分割の乱数シード．
        detr_factory (callable): キャッシュが無い場合にDETRを返す関数．
        yolo_factory (callable): キャッシュが無い場合にYOLO11xを返す関数．

    Returns:
        dict: "coco"，"lvis"（画像ID → アノテーション），"splits"（分割名 → 画像ID），"detections"（"detr"，"yolo"）．
    """
    d7 = ex007_data()
    coco = load_coco("instances")
    lvis = d7.load_lvis()
    all_ids = sorted(coco.getImgIds())
    return {"coco": coco, "lvis": lvis, "splits": d7.make_splits(coco, set(lvis), seed),
            "detections": {"detr": d7.detect_detr(coco, all_ids, detr_factory),
                           "yolo": d7.detect_yolo(coco, all_ids, yolo_factory)}}


def build_split_instances(ctx, split, detector, threshold, iou_threshold=0.5):
    """
    分割の画像について，検出ボックスとCOCOのGTを対応付けた評価インスタンスを作る．
    LVISの分割では，COCOのGTカテゴリより細かいLVISのSynsetが対応付いたインスタンスのみを残す．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．
        detector (str): "detr" または "yolo"．
        threshold (float): 検出スコアの閾値．
        iou_threshold (float): 対応付けのIoU閾値．

    Returns:
        tuple[list[dict], int]: インスタンスのリストとGTインスタンスの総数．
    """
    d7 = ex007_data()
    instances, n_gt = d7.build_instances(ctx["coco"], ctx["splits"][split], ctx["detections"][detector],
                                         threshold, iou_threshold)
    if split.startswith("lvis"):
        instances = [i for i in instances if d7.attach_lvis(i, ctx["lvis"].get(i["image_id"], []), iou_threshold)]
    return instances, n_gt
