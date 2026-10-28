"""
ex002の実行スクリプト：物体領域検出モデルのクラス非依存の位置検出精度の比較（論文4.3節，表3）．

学習は行わず，各検出器のボックスとGTボックスの一致度を評価する．
評価指標: 平均提案数，Recall@IoU0.5（IoU 0.5以上の提案が存在するGTの割合），平均最良IoU（GTごとの最良IoUの平均）．

出力: outputs/ex002_coco_4img_class_agnostic_det/{手法}/{閾値設定}/{seed}/
    config.json, per_gt.csv, detections.json, metrics.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.metrics import box_iou  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from data import load_dataset  # noqa: E402
from model import load_model  # noqa: E402

# ===== 実験条件（order_001） =====
EXPERIMENT = "ex002_coco_4img_class_agnostic_det"
SEEDS = [42]
# 手法 → 閾値設定のリスト（OWL-ViT，Grounding DINOは各実装の推奨値，DETRは論文に記載がないため複数設定）
CONDITIONS = [
    ("DETR", {"threshold": 0.5}),
    ("DETR", {"threshold": 0.7}),
    ("DETR", {"threshold": 0.9}),
    ("OWL-ViT", {"threshold": 0.1}),
    ("Grounding DINO", {"box_threshold": 0.35, "text_threshold": 0.25}),
]


def condition_name(params):
    """
    ハイパーパラメータの辞書からディレクトリ名を作る．

    Args:
        params (dict): ハイパーパラメータ．

    Returns:
        str: ディレクトリ名（例: "threshold0.9"）．
    """
    return "_".join(f"{k}{v}" for k, v in params.items())


def evaluate_detector(detector, samples):
    """
    全画像で検出を行い，GTごとの最良IoUと画像ごとの提案数を計算する．

    Args:
        detector: detect(image) を持つ検出器．
        samples (list[dict]): load_dataset の戻り値の画像リスト．

    Returns:
        tuple[pd.DataFrame, list[dict]]: GTごとの {"image_id", "gt_name", "best_iou"} と，
            画像ごとの検出結果 {"image_id", "boxes", "scores"}．
    """
    rows, detections = [], []
    for sample in samples:
        boxes, scores = detector.detect(sample["image"])
        iou = box_iou(sample["gt_boxes"], boxes) if len(boxes) else np.zeros((len(sample["gt_boxes"]), 1))
        for name, best in zip(sample["gt_names"], iou.max(axis=1)):
            rows.append({"image_id": sample["image_id"], "gt_name": name, "best_iou": float(best)})
        detections.append({"image_id": sample["image_id"], "boxes": boxes.tolist(), "scores": scores.tolist()})
    return pd.DataFrame(rows), detections


def run_condition(method, params, samples, queries, seed, target_dir):
    """
    1つの手法・閾値設定・シードについて評価を実行し保存する．

    Args:
        method (str): 手法名．
        params (dict): 閾値等のハイパーパラメータ．
        samples (list[dict]): 評価画像．
        queries (list[str]): テキストクエリ（COCO 80クラス名）．
        seed (int): シード．
        target_dir (str): 保存先ディレクトリ．

    Returns:
        dict: 評価指標．
    """
    detector = load_model(method, params, queries, seed)
    df, detections = evaluate_detector(detector, samples)
    metrics = {
        "mean_proposals": float(np.mean([len(d["boxes"]) for d in detections])),
        "recall_iou0.5": float((df["best_iou"] >= 0.5).mean()),
        "mean_best_iou": float(df["best_iou"].mean()),
        "n_images": len(samples),
        "n_gt": len(df),
    }
    os.makedirs(target_dir, exist_ok=True)
    df.to_csv(os.path.join(target_dir, "per_gt.csv"), index=False)
    with open(os.path.join(target_dir, "detections.json"), "w") as f:
        json.dump(detections, f)
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump({"method": method, "params": params, "seed": seed, "queries": queries,
                   "image_ids": [s["image_id"] for s in samples]}, f, indent=4)
    with open(os.path.join(target_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(method, params, metrics)
    del detector
    torch.cuda.empty_cache()
    return metrics


def main():
    """全条件の評価を実行する（完了済みの条件はスキップ）．"""
    for seed in SEEDS:
        samples, queries = load_dataset(seed)
        print("images:", [s["image_id"] for s in samples], "GT:", sum(len(s["gt_names"]) for s in samples))
        for method, params in CONDITIONS:
            target_dir = os.path.join(OUTPUT_DIR, EXPERIMENT, method.replace(" ", "_"), condition_name(params), str(seed))
            if os.path.exists(os.path.join(target_dir, "metrics.json")):
                continue
            run_condition(method, params, samples, queries, seed, target_dir)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
