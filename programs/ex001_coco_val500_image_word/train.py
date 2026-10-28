"""
ex001の実行スクリプト：ResNet50-IN1k vs CLIPによる画像単位の単語推定の評価（論文4.2節，表1・表2，図2）．

本論文の手法は学習を行わない推論パイプラインであるため，本スクリプトは学習ではなく評価を実行する．
各手法・各タスクについて画像単位の P@1, R@1, P@5, R@5 を計算し，
ブートストラップ95%信頼区間（B=1000）とWilcoxon符号付き順位検定のp値を保存する．

出力: outputs/ex001_coco_val500_image_word/{手法}/{タスク}/{seed}/
    config.json, per_image.csv, metrics.json
      outputs/ex001_coco_val500_image_word/wilcoxon.json
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.metrics import bootstrap_ci, precision_recall_at_k, wilcoxon_p  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from data import build_task_data, load_images  # noqa: E402
from model import METHODS, load_model  # noqa: E402

# ===== 実験条件（order_001） =====
EXPERIMENT = "ex001_coco_val500_image_word"
N_IMAGES = 500
SEEDS = [42]  # 画像サンプリングのシード（論文と同じ42）
TASKS = ["A", "B"]
KS = [1, 5]
N_BOOTSTRAP = 1000
METRIC_NAMES = [f"{m}@{k}" for k in KS for m in ("P", "R")]


def evaluate_scores(scores, vocab, image_ids, gt):
    """
    スコア行列から画像単位のPrecision/Recall@kを計算する．

    Args:
        scores (np.ndarray): スコア行列（N×V）．
        vocab (list[str]): 候補語彙（長さV）．
        image_ids (list[int]): 画像IDのリスト（長さN）．
        gt (dict): 画像ID（str）→ 正解語のリスト．

    Returns:
        pd.DataFrame: 画像ごとの指標値（GTが空の画像は除外）．列は image_id と METRIC_NAMES と top5．
    """
    rows = []
    for image_id, score in zip(image_ids, scores):
        gt_words = set(gt[str(image_id)])
        if not gt_words:
            continue
        ranked = [vocab[i] for i in np.argsort(-score, kind="stable")[:max(KS)]]
        row = {"image_id": image_id, "top5": "|".join(ranked[:5])}
        for k in KS:
            row[f"P@{k}"], row[f"R@{k}"] = precision_recall_at_k(ranked, gt_words, k)
        rows.append(row)
    return pd.DataFrame(rows)


def run_condition(model, images, data, task, seed, target_dir):
    """
    1つの手法・タスク・シードについて評価を実行し保存する．

    Args:
        model: score(images, vocab) を持つVLM．
        images (list[PIL.Image.Image]): 画像のリスト（長さN）．
        data (dict): build_task_data の戻り値．
        task (str): "A" または "B"．
        seed (int): シード．
        target_dir (str): 保存先ディレクトリ．

    Returns:
        pd.DataFrame: 画像単位の指標値．
    """
    vocab, gt = data[task]["vocab"], data[task]["gt"]
    start = time.time()
    df = evaluate_scores(model.score(images, vocab), vocab, data["image_ids"], gt)
    metrics = {}
    for name in METRIC_NAMES:
        mean, lo, hi = bootstrap_ci(df[name].values, N_BOOTSTRAP, seed=seed)
        metrics[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}
    os.makedirs(target_dir, exist_ok=True)
    df.to_csv(os.path.join(target_dir, "per_image.csv"), index=False)
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump({"method": model.name, "task": task, "seed": seed, "n_images": N_IMAGES,
                   "n_evaluated_images": len(df), "vocab_size": len(vocab), "n_bootstrap": N_BOOTSTRAP,
                   "elapsed_sec": time.time() - start}, f, indent=4)
    with open(os.path.join(target_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(model.name, task, {k: round(v["mean"], 3) for k, v in metrics.items()})
    return df


def run_wilcoxon(seed):
    """
    全手法ペア・全タスク・全指標についてWilcoxon符号付き順位検定を行い保存する．

    Args:
        seed (int): シード．

    Returns:
        list[dict]: 検定結果（task, metric, method_a, method_b, p_value）．
    """
    results = []
    for task, (a, b), metric in itertools.product(TASKS, itertools.combinations(METHODS, 2), METRIC_NAMES):
        df_a = pd.read_csv(os.path.join(OUTPUT_DIR, EXPERIMENT, a, task, str(seed), "per_image.csv"))
        df_b = pd.read_csv(os.path.join(OUTPUT_DIR, EXPERIMENT, b, task, str(seed), "per_image.csv"))
        merged = df_a.merge(df_b, on="image_id", suffixes=("_a", "_b"))
        results.append({"task": task, "metric": metric, "method_a": a, "method_b": b,
                        "p_value": wilcoxon_p(merged[f"{metric}_a"], merged[f"{metric}_b"])})
    with open(os.path.join(OUTPUT_DIR, EXPERIMENT, f"wilcoxon_seed{seed}.json"), "w") as f:
        json.dump(results, f, indent=4)
    return results


def main():
    """全手法・全タスク・全シードの評価を実行する（完了済みの条件はスキップ）．"""
    for seed in SEEDS:
        data = build_task_data(N_IMAGES, seed)
        print(f"seed={seed}: |V_A|={len(data['A']['vocab'])}, |V_B|={len(data['B']['vocab'])}")
        images = None
        for method in METHODS:
            todo = [t for t in TASKS
                    if not os.path.exists(os.path.join(OUTPUT_DIR, EXPERIMENT, method, t, str(seed), "metrics.json"))]
            if not todo:
                continue
            images = images or load_images(data["image_ids"])
            model = load_model(method, seed)
            for task in todo:
                run_condition(model, images, data, task, seed, os.path.join(OUTPUT_DIR, EXPERIMENT, method, task, str(seed)))
            del model
            torch.cuda.empty_cache()
        run_wilcoxon(seed)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
