"""
ex003の実行スクリプト：提案パイプライン全体のカテゴリアノテーション精度とWordNet粒度可変アノテーションの確信度評価
（論文4.4節，4.5節，表4，図3）．

学習は行わない．DETR（1回の推論）→ 局所クロップ → VLMによるゼロショット分類（式(2)，COCO 80クラス）で
Top-1/Top-5正解率を計算し，深さ0の推定が正解のインスタンスに対してWordNetの下位語をたどる
粒度可変アノテーション（式(3)）を行い，深さごとの確信度と候補数を記録する．

出力: outputs/ex003_coco_val500_pipeline/{VLM}/detr_threshold{閾値}_crop_{前処理}/{seed}/
    config.json, instances.json, metrics.json, depth_stats.csv
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET, hierarchical_annotation  # noqa: E402
from data import build_instances  # noqa: E402
from model import METHODS, load_model  # noqa: E402

# ===== 実験条件（order_001） =====
EXPERIMENT = "ex003_coco_val500_pipeline"
N_IMAGES = 500
SEEDS = [42]
DETR_THRESHOLDS = [0.5, 0.7, 0.9, 0.97]  # 論文に記載がないため複数設定で評価する（0.97は論文のインスタンス数1466件に近い設定）
CROP_MODES = ["center", "pad"]  # VLMへの入力前処理（論文に記載なし）．center: VLM標準の中央切り出し，pad: 黒で正方形化
IOU_THRESHOLD = 0.5
HIERARCHY_METHODS = ["CLIP-ViT-B32", "CLIP-RN50"]  # WordNet階層をたどるVLM（ResNet50-IN1kは語彙が閉じているため対象外）


def classify_instances(model, instances, vocab):
    """
    各インスタンスのクロップを候補語彙でゼロショット分類する（式(2)）．

    Args:
        model: score(images, vocab) を持つVLM．
        instances (list[dict]): build_instances のインスタンス．
        vocab (list[str]): 候補語彙（COCO 80クラス，長さV）．

    Returns:
        list[dict]: インスタンスごとの {"top5", "top5_conf", "top1_correct", "top5_correct"}．
    """
    scores = model.score([ins["crop"] for ins in instances], vocab)
    results = []
    for ins, score in zip(instances, scores):
        order = np.argsort(-score, kind="stable")[:5]
        top5 = [vocab[i] for i in order]
        results.append({"top5": top5, "top5_conf": [float(score[i]) for i in order],
                        "top1_correct": top5[0] == ins["gt_name"], "top5_correct": ins["gt_name"] in top5})
    return results


def annotate_hierarchy(model, instances, results):
    """
    深さ0の推定が正解のインスタンスについて，WordNet階層をたどる粒度可変アノテーションを行う．

    Args:
        model: score(images, vocab) を持つVLM．
        instances (list[dict]): インスタンス．
        results (list[dict]): classify_instances の戻り値（"chain" を追記する）．

    Returns:
        None
    """
    for ins, res in tqdm(list(zip(instances, results)), desc="WordNet", leave=False):
        if not res["top1_correct"]:
            continue
        word = res["top5"][0]
        res["chain"] = hierarchical_annotation(model, ins["crop"], word, COCO_TO_SYNSET[word], res["top5_conf"][0])


def depth_statistics(results):
    """
    深さごとのインスタンス数・平均候補数・平均確信度を集計する（表4）．

    Args:
        results (list[dict]): "chain" を含むインスタンスの結果．

    Returns:
        pd.DataFrame: 列 depth, n_instances, mean_candidates, mean_conf, std_conf．
    """
    steps = [step for res in results for step in res.get("chain", [])]
    df = pd.DataFrame(steps)
    df.loc[df["depth"] == 0, "n_candidates"] = 80
    return (df.groupby("depth")
              .agg(n_instances=("conf", "size"), mean_candidates=("n_candidates", "mean"),
                   mean_conf=("conf", "mean"), std_conf=("conf", "std"))
              .reset_index())


def save_condition(target_dir, instances, results, stats, config):
    """
    1条件の結果を保存する．

    Args:
        target_dir (str): 保存先ディレクトリ．
        instances (list[dict]): インスタンス．
        results (list[dict]): インスタンスごとの結果．
        stats (dict): 検出数・GT数の統計．
        config (dict): 実験条件．

    Returns:
        dict: 評価指標．
    """
    metrics = {
        "n_instances": len(results),
        "top1_accuracy": float(np.mean([r["top1_correct"] for r in results])),
        "top5_accuracy": float(np.mean([r["top5_correct"] for r in results])),
        **stats,
    }
    records = [{k: v for k, v in ins.items() if k != "crop"} | res for ins, res in zip(instances, results)]
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "instances.json"), "w") as f:
        json.dump(records, f)
    with open(os.path.join(target_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    if any("chain" in r for r in results):
        depth_statistics(results).to_csv(os.path.join(target_dir, "depth_stats.csv"), index=False)
    return metrics


def main():
    """全条件（VLM × DETR閾値 × クロップ前処理 × seed）の評価を実行する（完了済みの条件はスキップ）．"""
    for seed, threshold, crop_mode in itertools.product(SEEDS, DETR_THRESHOLDS, CROP_MODES):
        condition = f"detr_threshold{threshold}_crop_{crop_mode}"
        todo = [m for m in METHODS
                if not os.path.exists(os.path.join(OUTPUT_DIR, EXPERIMENT, m, condition, str(seed), "metrics.json"))]
        if not todo:
            continue
        instances, vocab, stats = build_instances(threshold, crop_mode, N_IMAGES, seed, IOU_THRESHOLD)
        for method in todo:
            start = time.time()
            model = load_model(method, seed)
            results = classify_instances(model, instances, vocab)
            if method in HIERARCHY_METHODS:
                annotate_hierarchy(model, instances, results)
            config = {"method": method, "detr_threshold": threshold, "crop_mode": crop_mode,
                      "iou_threshold": IOU_THRESHOLD, "seed": seed,
                      "n_images": N_IMAGES, "elapsed_sec": time.time() - start}
            target_dir = os.path.join(OUTPUT_DIR, EXPERIMENT, method, condition, str(seed))
            print(method, condition, save_condition(target_dir, instances, results, stats, config))
            del model
            torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
