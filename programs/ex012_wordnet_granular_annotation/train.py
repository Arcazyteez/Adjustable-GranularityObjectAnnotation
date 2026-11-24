"""
ex012の実行スクリプト：進捗報告（2026B）の粒度可変アノテーションの定量評価と定性評価（order_013）．

学習は行わない．
- 定量評価: 初期の語彙 W をCOCOの80クラスとし，下位語は生成しない．提案手法（D のボックス + F によるクラス）と，
  Close Vocabularyモデル（D 自身のクラス）を，同じボックスで比較する．
  θ は検証用（dev）でクラスを含めたF値が最大となる値を選び，テスト用（test）で報告する（θ = -1 は棄却なし）．
- 定性評価: 初期の語彙 W をWordNetの上位語（model.WORDNET_TOPS）とし，テスト用の画像に下位語まで予測する．

出力: outputs/ex012_wordnet_granular_annotation/{手法}/{分割}/{seed}/
    metrics.json，predictions.json（test），config.json，theta_curve.csv（提案手法の dev）
    qualitative_{D}_{F}/{seed}/annotations.json，grid.png
    および直下の table_quantitative.csv
"""
import contextlib
import io
import itertools
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.metrics import match_boxes  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from data import SPLITS, cached_detections, ground_truth, load_dataset, make_splits, qualitative_images  # noqa: E402
from model import WORDNET_TOPS, Embedder, coco_similarities, crop, granular_annotation, load_detector  # noqa: E402

# ===== 実験条件（order_013） =====
EXPERIMENT = "ex012_wordnet_granular_annotation"
SEEDS = [42]
DETECTORS = ["yolo", "detr"]
EMBEDDERS = ["CLIP", "SigLIP"]
SCORE_THRESHOLD = {"yolo": 0.25, "detr": 0.5}   # 評価に用いるボックスの検出スコアの下限（Ultralytics，DETRの一般的な値）
IOU_THRESHOLD = 0.5
THETA_QUANTILES = np.linspace(0.0, 0.95, 20)    # θ の候補（dev の最大類似度の分位点）


def target_dir(method, split, seed):
    """
    1条件の保存先を返す．

    Args:
        method (str): 手法名（"closed_{D}" または "proposed_{D}_{F}"）．
        split (str): 分割名．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, split, str(seed))


def box_lists(detections, image_ids, detector):
    """
    検出スコアが下限以上のボックスを画像ごとに返す．

    Args:
        detections (dict): cached_detections の戻り値．
        image_ids (list[int]): 画像ID．
        detector (str): 物体認識モデルの名前．

    Returns:
        dict[str, list[dict]]: 画像ID → {"box"，"score"，"label"（D のクラス）}．
    """
    out = {}
    for i in image_ids:
        d = detections[str(i)]
        out[str(i)] = [{"box": b, "score": s, "label": l} for b, s, l in zip(d["boxes"], d["scores"], d["labels"])
                       if s >= SCORE_THRESHOLD[detector]]
    return out


# ===================== 評価指標 =====================

def coco_map(coco, predictions, image_ids, class_names):
    """
    COCOの mAP（AP@[.5:.95]，AP50）を計算する．

    Args:
        coco (COCO): COCOオブジェクト．
        predictions (dict): 画像ID → {"box"，"score"，"label"} のリスト．
        image_ids (list[int]): 評価する画像ID．
        class_names (list[str]): COCOのクラス名．

    Returns:
        dict: AP，AP50．
    """
    name_to_id = {coco.cats[c]["name"]: c for c in coco.getCatIds()}
    results = [{"image_id": int(i), "category_id": name_to_id[p["label"]], "score": float(p["score"]),
                "bbox": [p["box"][0], p["box"][1], p["box"][2] - p["box"][0], p["box"][3] - p["box"][1]]}
               for i, preds in predictions.items() for p in preds]
    if not results:
        return {"AP": 0.0, "AP50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
        evaluator.params.imgIds = [int(i) for i in image_ids]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1])}


def evaluate(predictions, gts):
    """
    予測とCOCOのラベルから予測精度を計算する．予測とGTはIoU 0.5以上でハンガリアン法により1対1に対応付ける．

    Args:
        predictions (dict): 画像ID → {"box"，"label"，...} のリスト．
        gts (dict): 画像ID → (GTのボックス（K×4），クラス名のリスト)．

    Returns:
        dict: n_pred，n_gt，n_matched，label_accuracy（対応したボックスのうちクラスが一致した割合），
            loc_precision / loc_recall（クラスを問わない対応の割合），precision / recall / f1（クラスも一致したもの），
            mean_iou，center_error（中心のずれ / GTの対角線の長さ），coord_error_px（座標の誤差の平均，ピクセル）．
    """
    n_pred = n_gt = n_matched = n_correct = 0
    ious, centers, coords = [], [], []
    for image_id, (gt_boxes, gt_names) in gts.items():
        preds = predictions.get(image_id, [])
        n_pred += len(preds)
        n_gt += len(gt_names)
        if not preds or not gt_names:
            continue
        boxes = np.asarray([p["box"] for p in preds], dtype=np.float32)
        for p, g, iou in match_boxes(boxes, np.asarray(gt_boxes, dtype=np.float32), IOU_THRESHOLD):
            n_matched += 1
            n_correct += preds[p]["label"] == gt_names[g]
            ious.append(iou)
            pb, gb = boxes[p], np.asarray(gt_boxes[g], dtype=np.float32)
            diag = np.hypot(gb[2] - gb[0], gb[3] - gb[1])
            centers.append(np.hypot((pb[0] + pb[2] - gb[0] - gb[2]) / 2, (pb[1] + pb[3] - gb[1] - gb[3]) / 2) / max(diag, 1e-9))
            coords.append(float(np.abs(pb - gb).mean()))
    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_gt if n_gt else 0.0
    return {"n_pred": n_pred, "n_gt": n_gt, "n_matched": n_matched,
            "label_accuracy": n_correct / n_matched if n_matched else float("nan"),
            "loc_precision": n_matched / n_pred if n_pred else 0.0, "loc_recall": n_matched / n_gt if n_gt else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
            "center_error": float(np.mean(centers)) if centers else float("nan"),
            "coord_error_px": float(np.mean(coords)) if coords else float("nan")}


# ===================== 提案手法の予測 =====================

def crop_features(embedder, coco, boxes):
    """
    全ボックスの切り抜き画像の特徴ベクトルを計算する．

    Args:
        embedder (Embedder): F．
        coco (COCO): COCOオブジェクト．
        boxes (dict): 画像ID → ボックスのリスト（box_lists の戻り値）．

    Returns:
        dict[str, torch.Tensor]: 画像ID → 特徴ベクトル（K×D）．
    """
    out = {}
    for image_id, items in tqdm(boxes.items(), desc=f"{embedder.name} features", leave=False):
        if not items:
            out[image_id] = torch.zeros(0)
            continue
        image = load_image(coco, int(image_id))
        out[image_id] = embedder.image_features([crop(image, b["box"]) for b in items])
    return out


def proposed_predictions(boxes, similarities, class_names, theta):
    """
    COCOの80クラスを初期の語彙とし，類似度が θ 以上のボックスに最も類似したクラスを付ける（下位語は生成しない）．

    Args:
        boxes (dict): 画像ID → ボックスのリスト．
        similarities (dict): 画像ID → コサイン類似度（K×80）．
        class_names (list[str]): COCOのクラス名．
        theta (float): 類似と判断する下限値．

    Returns:
        dict: 画像ID → {"box"，"score"（D の検出スコア），"label"，"similarity"} のリスト．
    """
    out = {}
    for image_id, items in boxes.items():
        sims = similarities[image_id]
        preds = []
        for b, s in zip(items, sims):
            j = int(np.argmax(s))
            if s[j] >= theta:
                preds.append({"box": b["box"], "score": b["score"], "label": class_names[j], "similarity": float(s[j])})
        out[image_id] = preds
    return out


def save_condition(method, split, seed, metrics, config, predictions=None, curve=None):
    """
    1条件の結果を保存する．

    Args:
        method (str): 手法名．
        split (str): 分割名．
        seed (int): シード．
        metrics (dict): 指標．
        config (dict): 実験条件．
        predictions (dict | None): 予測．
        curve (pd.DataFrame | None): θ に対する dev の指標．

    Returns:
        None
    """
    out_dir = target_dir(method, split, seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    if predictions is not None:
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
    if curve is not None:
        curve.to_csv(os.path.join(out_dir, "theta_curve.csv"), index=False)


# ===================== 定性評価 =====================

def run_qualitative(coco, embedder, detector, detections, image_ids, theta, seed):
    """
    初期の語彙をWordNetの上位語として，テスト用の画像に粗い語から細かい語までのクラスを予測し，図を保存する．

    Args:
        coco (COCO): COCOオブジェクト．
        embedder (Embedder): F．
        detector (str): 物体認識モデルの名前．
        detections (dict): cached_detections の戻り値．
        image_ids (list[int]): 定性評価の画像ID．
        theta (float): 類似と判断する下限値（定量評価で選んだ値）．
        seed (int): シード．

    Returns:
        None
    """
    out_dir = target_dir(f"qualitative_{detector}_{embedder.name}", "test", seed)
    if os.path.exists(os.path.join(out_dir, "annotations.json")):
        return
    boxes = box_lists(detections, image_ids, detector)
    results = {}
    for image_id, items in boxes.items():
        image = load_image(coco, int(image_id))
        feats = embedder.image_features([crop(image, b["box"]) for b in items]) if items else []
        results[image_id] = [{"box": b["box"], "score": b["score"], "closed_label": b["label"],
                              "chain": granular_annotation(embedder, f, WORDNET_TOPS, theta)} for b, f in zip(items, feats)]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump({"theta": theta, "initial_vocabulary": WORDNET_TOPS, "results": results}, f, indent=1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, image_id in zip(axes.flat, results):
        ax.imshow(load_image(coco, int(image_id)))
        ax.axis("off")
        ax.set_title(f"image {image_id} ({detector} + {embedder.name}, θ={theta:.3f})", fontsize=9)
        for k, r in enumerate(results[image_id]):
            x0, y0, x1, y1 = r["box"]
            color = plt.cm.tab10.colors[k % 10]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
            text = " > ".join(c["word"] for c in r["chain"]) if r["chain"] else "(rejected)"
            ax.text(x0, y0, text, fontsize=6, color="white", va="top", bbox=dict(facecolor=color, alpha=0.75, pad=1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================== 実行 =====================

def main():
    """定量評価（Close Vocabularyモデルと提案手法）と定性評価を実行し，集計表を作成する．"""
    coco, class_names = load_dataset()
    all_ids = sorted(coco.getImgIds())
    for seed in SEEDS:
        splits = make_splits(coco, seed)
        gts = {s: ground_truth(coco, splits[s]) for s in SPLITS}
        detections = {d: cached_detections(d, coco, all_ids, lambda d=d: load_detector(d, seed)) for d in DETECTORS}
        torch.cuda.empty_cache()
        rows, thetas = [], {}
        for detector in DETECTORS:
            boxes = {s: box_lists(detections[detector], splits[s], detector) for s in SPLITS}
            for split in SPLITS:
                preds = {i: [{"box": b["box"], "score": b["score"], "label": b["label"]} for b in items]
                         for i, items in boxes[split].items()}
                metrics = evaluate(preds, gts[split]) | coco_map(coco, preds, splits[split], class_names)
                save_condition(f"closed_{detector}", split, seed, metrics,
                               {"detector": detector, "score_threshold": SCORE_THRESHOLD[detector], "seed": seed},
                               preds if split == "test" else None)
                rows.append({"method": f"closed_{detector}", "detector": detector, "embedder": None, "split": split,
                             "theta": None} | metrics)
        for embedder_name in EMBEDDERS:
            embedder = Embedder(embedder_name, seed)
            for detector in DETECTORS:
                start = time.time()
                boxes = {s: box_lists(detections[detector], splits[s], detector) for s in SPLITS}
                sims = {}
                for split in SPLITS:
                    feats = crop_features(embedder, coco, boxes[split])
                    sims[split] = {i: coco_similarities(embedder, f, class_names) for i, f in feats.items()}
                dev_max = np.concatenate([s.max(axis=1) for s in sims["dev"].values() if len(s)])
                candidates = [-1.0] + [float(np.quantile(dev_max, q)) for q in THETA_QUANTILES[1:]]
                curve = pd.DataFrame([{"theta": t} | evaluate(proposed_predictions(boxes["dev"], sims["dev"], class_names, t),
                                                              gts["dev"]) for t in candidates])
                theta = float(curve.loc[curve["f1"].idxmax(), "theta"])
                thetas[(detector, embedder_name)] = theta
                method = f"proposed_{detector}_{embedder_name}"
                for split, (label, t) in itertools.product(SPLITS, [("none", -1.0), ("selected", theta)]):
                    preds = proposed_predictions(boxes[split], sims[split], class_names, t)
                    metrics = evaluate(preds, gts[split]) | coco_map(coco, preds, splits[split], class_names)
                    name = method if label == "selected" else f"{method}_no_reject"
                    save_condition(name, split, seed, metrics,
                                   {"detector": detector, "embedder": embedder_name, "score_threshold": SCORE_THRESHOLD[detector],
                                    "theta": t, "theta_selection": "dev F1" if label == "selected" else "none",
                                    "initial_vocabulary": "COCO 80 classes", "hyponyms": False, "seed": seed,
                                    "elapsed_sec": time.time() - start},
                                   preds if split == "test" else None, curve if split == "dev" and label == "selected" else None)
                    rows.append({"method": name, "detector": detector, "embedder": embedder_name, "split": split, "theta": t}
                                | metrics)
                print(method, "theta", round(theta, 4), {k: round(v, 4) for k, v in rows[-1].items()
                                                         if k in ["label_accuracy", "precision", "recall", "f1", "AP"]})
            q_ids = qualitative_images(coco, splits["test"], seed=seed)
            for detector in DETECTORS:
                run_qualitative(coco, embedder, detector, detections[detector], q_ids, thetas[(detector, embedder_name)], seed)
            del embedder
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, EXPERIMENT, "table_quantitative.csv"), index=False)


if __name__ == "__main__":
    main()
