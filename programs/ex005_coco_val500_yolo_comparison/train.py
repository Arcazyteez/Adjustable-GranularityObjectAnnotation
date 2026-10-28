"""
ex005の実行スクリプト：閉域物体検出モデルYOLOと提案手法（DETR + CLIP-ViT-B/32）の比較．

学習は行わない．YOLOは自身の語彙（COCOの80クラス）でカテゴリを出力し，提案手法の第1段階と同じ語彙で比較する．
1. coco500: COCO val2017の500枚でのカテゴリアノテーション精度（Top-1/Top-5），共通GTインスタンスでの比較
   （McNemar検定），YOLOのボックスをCLIPで分類する組合せ（YOLO-box + CLIP）
2. loc4: ex002の4画像でのクラス非依存の位置検出精度
3. open_vocab: ex004のCOCO外の語を含む実画像6枚での定性比較

出力: outputs/ex005_coco_val500_yolo_comparison/{手法}/{画像集合}_conf{閾値}/{seed}/
"""
import itertools
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from tqdm import tqdm

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, load_coco, make_crop  # noqa: E402
from common.metrics import box_iou, match_boxes  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from data import load_coco_samples, load_loc4_samples, load_open_vocab_samples, load_proposed_instances  # noqa: E402
from model import load_model, load_vlm_model  # noqa: E402

# ===== 実験条件（order_003） =====
EXPERIMENT = "ex005_coco_val500_yolo_comparison"
SEEDS = [42]
N_IMAGES = 500
YOLO_MODELS = ["yolo11n", "yolo11m", "yolo11x", "yolov8x"]
CONFS = [0.25, 0.5, 0.7]  # 0.25はUltralyticsの既定値
IOU_THRESHOLD = 0.5
PROPOSED_REFS = ["detr_threshold0.5_crop_pad", "detr_threshold0.97_crop_pad"]  # 比較する提案手法（ex003）の条件
CLIP_ON_YOLO_MODELS = ["yolo11x"]  # YOLO-box + CLIP を評価するYOLO
VLM_NAME = "CLIP-ViT-B32"
OPEN_VOCAB_CONDITIONS = [("yolo11x", 0.25)]


def target_dir(method, condition, seed):
    """
    結果の保存先ディレクトリを返す．

    Args:
        method (str): 手法名．
        condition (str): 条件名．
        seed (int): シード．

    Returns:
        str: ディレクトリのパス．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, condition, str(seed))


def save_json(path, obj):
    """
    JSONファイルを保存する（親ディレクトリも作成する）．

    Args:
        path (str): 保存先．
        obj: JSONに変換可能なオブジェクト．

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def match_instances(detections, samples):
    """
    YOLOの検出をGTとIoU 0.5以上で1対1に対応付け，評価インスタンスを作る．

    Args:
        detections (list[dict]): 画像ごとの detect の戻り値．
        samples (list[dict]): load_coco_samples の戻り値．

    Returns:
        list[dict]: {"image_id", "gt_index", "gt_name", "box", "det_score", "iou", "top5", "top5_conf",
            "top1_correct", "top5_correct"} のリスト．
    """
    instances = []
    for det, sample in zip(detections, samples):
        for p, g, iou in match_boxes(det["boxes"], sample["gt_boxes"], IOU_THRESHOLD):
            gt_name = sample["gt_names"][g]
            instances.append({"image_id": sample["image_id"], "gt_index": g, "gt_name": gt_name, "box": det["boxes"][p],
                              "det_score": det["scores"][p], "iou": iou, "top5": det["top5"][p],
                              "top5_conf": det["top5_conf"][p], "top1_correct": det["top5"][p][0] == gt_name,
                              "top5_correct": gt_name in det["top5"][p]})
    return instances


def summarize(instances, n_detections, n_gt):
    """
    インスタンスの正誤から評価指標を計算する．

    Args:
        instances (list[dict]): 評価インスタンス．
        n_detections (int): 検出数の合計．
        n_gt (int): GTインスタンス数の合計．

    Returns:
        dict: n_instances，top1_accuracy，top5_accuracy，正しく命名されたGTの割合（annotation_recall）等．
    """
    top1 = [i["top1_correct"] for i in instances]
    return {"n_instances": len(instances), "n_detections": n_detections, "n_gt": n_gt,
            "top1_accuracy": float(np.mean(top1)), "top5_accuracy": float(np.mean([i["top5_correct"] for i in instances])),
            "annotation_recall": float(np.sum(top1) / n_gt)}


def compare_common(instances, proposed):
    """
    YOLOと提案手法の両方が検出した共通のGTインスタンスでTop-1正解率を比較する（McNemar検定）．

    Args:
        instances (list[dict]): YOLOの評価インスタンス．
        proposed (dict): load_proposed_instances の戻り値．

    Returns:
        dict: 共通インスタンス数，各手法の正解率，不一致の件数，McNemarの正確なp値，人物クラスでの正解率．
    """
    pairs = [(i["top1_correct"], proposed[(i["image_id"], i["gt_index"])]["top1_correct"], i["gt_name"])
             for i in instances if (i["image_id"], i["gt_index"]) in proposed]
    yolo, prop, names = map(np.array, zip(*pairs))
    b, c = int((yolo & ~prop).sum()), int((~yolo & prop).sum())
    person = names == "person"
    return {"n_common": len(pairs), "yolo_top1": float(yolo.mean()), "proposed_top1": float(prop.mean()),
            "both_correct": int((yolo & prop).sum()), "yolo_only": b, "proposed_only": c,
            "mcnemar_p": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0,
            "n_common_person": int(person.sum()), "yolo_top1_person": float(yolo[person].mean()),
            "proposed_top1_person": float(prop[person].mean()),
            "yolo_top1_non_person": float(yolo[~person].mean()), "proposed_top1_non_person": float(prop[~person].mean())}


def classify_with_clip(vlm, instances, samples, vocab):
    """
    YOLOのボックスのクロップ（正方形パディング）をCLIPでゼロショット分類する（YOLO-box + CLIP）．

    Args:
        vlm (CLIPModel): VLM．
        instances (list[dict]): YOLOの評価インスタンス．
        samples (list[dict]): 画像．
        vocab (list[str]): COCOの80クラス．

    Returns:
        list[dict]: top5，top5_conf，top1_correct，top5_correct を差し替えたインスタンス．
    """
    images = {s["image_id"]: s["image"] for s in samples}
    crops = [make_crop(images[i["image_id"]], i["box"], "pad") for i in instances]
    scores = vlm.score(crops, vocab)
    results = []
    for ins, score in zip(instances, scores):
        order = np.argsort(-score, kind="stable")[:5]
        top5 = [vocab[j] for j in order]
        results.append(ins | {"top5": top5, "top5_conf": [float(score[j]) for j in order],
                              "top1_correct": top5[0] == ins["gt_name"], "top5_correct": ins["gt_name"] in top5})
    return results


def run_coco500(samples, vocab, proposed_refs, seed):
    """
    全YOLOモデル・全閾値でCOCO 500枚の評価を行う（完了済みの条件はスキップ）．

    Args:
        samples (list[dict]): COCOの評価画像．
        vocab (list[str]): COCOの80クラス．
        proposed_refs (dict): 条件名 → load_proposed_instances の戻り値．
        seed (int): シード．

    Returns:
        None
    """
    n_gt = sum(len(s["gt_names"]) for s in samples)
    vlm = None
    for name, conf in itertools.product(YOLO_MODELS, CONFS):
        out_dir = target_dir(name, f"coco500_conf{conf}", seed)
        if os.path.exists(os.path.join(out_dir, "metrics.json")):
            continue
        detector = load_model(name, conf, seed)
        detections = [detector.detect(s["image"]) for s in tqdm(samples, desc=f"{name} conf={conf}", leave=False)]
        instances = match_instances(detections, samples)
        metrics = summarize(instances, sum(len(d["boxes"]) for d in detections), n_gt)
        save_json(os.path.join(out_dir, "instances.json"), instances)
        save_json(os.path.join(out_dir, "metrics.json"), metrics)
        save_json(os.path.join(out_dir, "config.json"), {"model": name, "conf": conf, "nms_iou": detector.iou,
                                                         "match_iou": IOU_THRESHOLD, "seed": seed, "n_images": len(samples)})
        for ref, proposed in proposed_refs.items():
            save_json(os.path.join(out_dir, f"common_vs_{ref}.json"), compare_common(instances, proposed))
        print(name, conf, metrics)
        if name in CLIP_ON_YOLO_MODELS:
            vlm = vlm or load_vlm_model(VLM_NAME, seed)
            clip_instances = classify_with_clip(vlm, instances, samples, vocab)
            clip_dir = target_dir(f"{name}+{VLM_NAME}", f"coco500_conf{conf}", seed)
            clip_metrics = summarize(clip_instances, metrics["n_detections"], n_gt)
            save_json(os.path.join(clip_dir, "instances.json"), clip_instances)
            save_json(os.path.join(clip_dir, "metrics.json"), clip_metrics)
            save_json(os.path.join(clip_dir, "config.json"), {"detector": name, "vlm": VLM_NAME, "conf": conf,
                                                              "crop_mode": "pad", "seed": seed})
            print(f"{name}+{VLM_NAME}", conf, clip_metrics)
        del detector
        torch.cuda.empty_cache()


def run_loc4(seed):
    """
    ex002の4画像でYOLOのクラス非依存の位置検出精度を評価する．

    Args:
        seed (int): シード．

    Returns:
        None
    """
    samples = load_loc4_samples(seed)
    for name, conf in itertools.product(YOLO_MODELS, CONFS):
        out_dir = target_dir(name, f"loc4_conf{conf}", seed)
        if os.path.exists(os.path.join(out_dir, "metrics.json")):
            continue
        detector = load_model(name, conf, seed)
        rows, n_boxes = [], []
        for s in samples:
            boxes = np.array(detector.detect(s["image"])["boxes"]).reshape(-1, 4)
            n_boxes.append(len(boxes))
            best = box_iou(s["gt_boxes"], boxes).max(axis=1) if len(boxes) else np.zeros(len(s["gt_names"]))
            rows += [{"image_id": s["image_id"], "gt_name": n, "best_iou": float(b)} for n, b in zip(s["gt_names"], best)]
        df = pd.DataFrame(rows)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "per_gt.csv"), index=False)
        save_json(os.path.join(out_dir, "metrics.json"), {"mean_proposals": float(np.mean(n_boxes)),
                                                          "recall_iou0.5": float((df.best_iou >= 0.5).mean()),
                                                          "mean_best_iou": float(df.best_iou.mean())})


def run_open_vocab(seed):
    """
    COCO外の語を含む実画像6枚にYOLOを適用し，結果の図を保存する．

    Args:
        seed (int): シード．

    Returns:
        None
    """
    samples = load_open_vocab_samples()
    for name, conf in OPEN_VOCAB_CONDITIONS:
        out_dir = target_dir(name, f"open_vocab_conf{conf}", seed)
        if os.path.exists(os.path.join(out_dir, "annotations.json")):
            continue
        detector = load_model(name, conf, seed)
        results = {s["name"]: detector.detect(s["image"]) for s in samples}
        save_json(os.path.join(out_dir, "annotations.json"), results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, s in zip(axes.flat, samples):
            ax.imshow(s["image"])
            ax.axis("off")
            ax.set_title(f"{s['name']} ({len(results[s['name']]['boxes'])} boxes)")
            for i, (box, top5, conf5) in enumerate(zip(*(results[s["name"]][k] for k in ("boxes", "top5", "top5_conf")))):
                color = plt.cm.tab10.colors[i % 10]
                ax.add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color, lw=2))
                ax.text(box[0], box[1], f"{top5[0]} ({conf5[0]:.2f})", fontsize=7, color="white", va="top",
                        bbox=dict(facecolor=color, alpha=0.8, pad=1))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grid.png"), dpi=150)
        plt.close(fig)
        for image_name, r in results.items():
            print(name, image_name, [f"{t[0]}({c[0]:.2f})" for t, c in zip(r["top5"], r["top5_conf"])])


def main():
    """全条件の評価を実行する（完了済みの条件はスキップ）．"""
    for seed in SEEDS:
        vocab = category_names(load_coco("instances"))
        proposed_refs = {ref: load_proposed_instances(ref, N_IMAGES, seed) for ref in PROPOSED_REFS}
        run_coco500(load_coco_samples(N_IMAGES, seed), vocab, proposed_refs, seed)
        run_loc4(seed)
        run_open_vocab(seed)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
