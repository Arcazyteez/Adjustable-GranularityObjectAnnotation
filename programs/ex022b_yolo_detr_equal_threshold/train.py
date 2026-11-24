"""
ex022b：YOLO11x と DETR の検出（物体の領域）を，同じ閾値と同じ箱の数で比較する（order_049）．

ex022 では各検出器の既定の閾値（YOLO11x 0.25，DETR 0.5）で比較した．本実験では次の3通りで比較する．
  1. same_threshold：両方を同じスコアの閾値（0.5，0.6，0.7，0.8，0.9）にする
  2. same_boxes：DETR の閾値を，dev の画像で1枚あたりの箱の数が YOLO11x（0.25 と 0.5）と同じになるように決める
  3. default：従来の既定値（YOLO11x 0.25，DETR 0.5）
新しく検出器は動かさず，ex017 で保存した COCO val2017 の dev・test 各500枚の検出結果（YOLO11x はスコア 0.25 以上，
DETR は 0.5 以上を保存）を数え直す．

指標（クラスを問わず IoU 0.5 以上で照合）：
  boxes_per_image，recall（COCO の物体），recall_person，recall_other，box_no_gt_rate（COCO の正解に当たらない箱の割合），
  noncoco_recall（LVIS v1 の注釈のうち COCO 外の物体を囲んだ割合），n_noncoco

出力：outputs/ex022b_yolo_detr_equal_threshold/{condition}/{detector}_{threshold}/0/metrics.json，
      outputs/ex022b_yolo_detr_equal_threshold/table_{split}.csv，fig_recall_vs_boxes.png

実行：.venv_pytorch/bin/python programs/ex022b_yolo_detr_equal_threshold/train.py
"""
import itertools
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ex022b_yolo_detr_equal_threshold")
EX017_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "ex017_coco_granular_annotation")
SPLITS = ["dev", "test"]
DETECTORS = ["YOLO11x", "DETR"]
SAVED_MIN = {"YOLO11x": 0.25, "DETR": 0.5}        # 保存されている検出結果のスコアの下限
SAME_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT = {"YOLO11x": 0.25, "DETR": 0.5}
MATCH_YOLO = [0.25, 0.5]                           # 箱の数をそろえる YOLO11x の閾値
IOU = 0.5


def load_module(name, path):
    """
    ファイルからモジュールを読み込む．

    Args:
        name (str): モジュールの別名．
        path (str): ファイルのパス．

    Returns:
        module: モジュール．
    """
    import importlib.util
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def covered(a, b, thr=IOU):
    """
    a の各ボックスが，b のいずれかと IoU thr 以上で重なるかを判定する．

    Args:
        a (np.ndarray): ボックス（xyxy），形状 (N, 4)．
        b (np.ndarray): ボックス（xyxy），形状 (M, 4)．
        thr (float): IoU の閾値．

    Returns:
        np.ndarray: 真偽値，形状 (N,)．
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)
    return (iou >= thr).any(axis=1)


def load_data():
    """
    COCO の正解，LVIS の COCO 外の物体，保存済みの検出結果を読み込む．

    Returns:
        dict: "splits"（split → 画像ID），"coco"（画像ID → (G×4 の正解ボックス, 人物かの真偽値 (G,))），
              "noncoco"（画像ID → LVIS の COCO 外の物体のボックス K×4．注釈の無い画像は含まない），
              "dets"（(検出器, split) → 画像ID → (ボックス K×4, スコア K)）．
    """
    e17 = load_module("ex017_data", os.path.join(PROJECT_ROOT, "programs", "ex017_coco_granular_annotation", "data.py"))
    e18 = load_module("ex018_data", os.path.join(PROJECT_ROOT, "programs", "ex018_owl_region_check", "data.py"))
    coco = e17.load_coco()
    splits = json.load(open(os.path.join(EX017_DATASET_DIR, "splits.json")))
    lvis = e18.load_lvis()
    ids = [i for s in SPLITS for i in splits[s]]
    gt = {}
    for i in ids:
        boxes, names = e17.gt_instances(coco, i)
        gt[i] = (np.asarray(boxes, np.float32).reshape(-1, 4), np.array([n == "person" for n in names], dtype=bool))
    noncoco = {}
    for i in ids:
        if i in lvis:
            b, kinds, _ = e18.gt_arrays(lvis, i)
            noncoco[i] = b[np.array([k == "noncoco" for k in kinds], dtype=bool)]
    dets = {}
    for det, split in itertools.product(DETECTORS, SPLITS):
        raw = torch.load(os.path.join(EX017_DATASET_DIR, f"detections_{det}_{split}.pt"), weights_only=False)
        dets[(det, split)] = {i: (np.asarray(d["boxes"], np.float32).reshape(-1, 4), np.asarray(d["scores"], np.float32))
                              for i, d in raw.items()}
    return {"splits": {s: splits[s] for s in SPLITS}, "coco": gt, "noncoco": noncoco, "dets": dets}


def evaluate(data, det, split, threshold):
    """
    1つの検出器と閾値について，検出の指標を求める．

    Args:
        data (dict): load_data の戻り値．
        det (str): 検出器の名前．
        split (str): "dev" または "test"．
        threshold (float): スコアの閾値（保存の下限以上）．

    Returns:
        dict: 指標（boxes_per_image，recall，recall_person，recall_other，box_no_gt_rate，noncoco_recall，n_noncoco，n_images）．
    """
    assert threshold >= SAVED_MIN[det] - 1e-9, (det, threshold)
    ids = data["splits"][split]
    found, person, hits, nc_found = [], [], [], []
    n_box = 0
    for i in ids:
        boxes, scores = data["dets"][(det, split)][i]
        boxes = boxes[scores >= threshold]
        gt_boxes, is_person = data["coco"][i]
        found.append(covered(gt_boxes, boxes))
        person.append(is_person)
        hits.append(covered(boxes, gt_boxes))
        n_box += len(boxes)
        if i in data["noncoco"]:
            nc_found.append(covered(data["noncoco"][i], boxes))
    found, person, hits = np.concatenate(found), np.concatenate(person), np.concatenate(hits)
    nc = np.concatenate(nc_found) if nc_found else np.zeros(0, bool)
    return {"boxes_per_image": n_box / len(ids), "recall": float(found.mean()), "recall_person": float(found[person].mean()),
            "recall_other": float(found[~person].mean()), "box_no_gt_rate": float((~hits).mean()) if len(hits) else 0.0,
            "noncoco_recall": float(nc.mean()) if len(nc) else float("nan"), "n_noncoco": int(len(nc)), "n_images": len(ids)}


def match_threshold(data, det, split, target):
    """
    1枚あたりの箱の数が target に最も近くなる閾値を求める（保存の下限以上）．

    Args:
        data (dict): load_data の戻り値．
        det (str): 検出器の名前．
        split (str): 閾値を決める split（"dev"）．
        target (float): 目標の1枚あたりの箱の数．

    Returns:
        float: 閾値．
    """
    scores = np.sort(np.concatenate([s for _, s in data["dets"][(det, split)].values()]))[::-1]
    k = int(round(target * len(data["splits"][split])))
    k = min(max(k, 1), len(scores))
    return float(max(scores[k - 1], SAVED_MIN[det]))


def conditions(data):
    """
    比較する (条件, 検出器, 閾値) の一覧を作る．

    Args:
        data (dict): load_data の戻り値．

    Returns:
        list[tuple[str, str, float]]: (condition, detector, threshold) のリスト．
    """
    out = [("default", d, DEFAULT[d]) for d in DETECTORS]
    out += [("same_threshold", d, t) for d, t in itertools.product(DETECTORS, SAME_THRESHOLDS)]
    for t in MATCH_YOLO:
        target = evaluate(data, "YOLO11x", "dev", t)["boxes_per_image"]
        out += [(f"same_boxes_as_yolo{t}", "YOLO11x", t), (f"same_boxes_as_yolo{t}", "DETR", match_threshold(data, "DETR", "dev", target))]
    return out


def run(data):
    """
    全ての条件を dev と test で評価し，条件ごとの metrics.json と split ごとの表を保存する．

    Args:
        data (dict): load_data の戻り値．

    Returns:
        pd.DataFrame: 全ての結果（行：条件×検出器×split）．
    """
    rows = []
    for (cond, det, thr), split in tqdm(list(itertools.product(conditions(data), SPLITS)), desc="ex022b", leave=False):
        out_dir = os.path.join(OUTPUT_DIR, cond, f"{det}_{thr:.4f}", "0")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"metrics_{split}.json")
        if os.path.exists(path):
            m = json.load(open(path))
        else:
            m = {"condition": cond, "detector": det, "threshold": thr, "split": split, **evaluate(data, det, split, thr)}
            json.dump(m, open(path, "w"), indent=1)
        rows.append(m)
    df = pd.DataFrame(rows)
    for split in SPLITS:
        df[df.split == split].to_csv(os.path.join(OUTPUT_DIR, f"table_{split}.csv"), index=False)
    return df


def plot(df):
    """
    test の箱の数と再現率（COCO の物体，COCO 外の物体）の関係を，同じ閾値の点を結んで描く．

    Args:
        df (pd.DataFrame): run の戻り値．

    Returns:
        None
    """
    t = df[(df.split == "test") & df.condition.isin(["same_threshold", "default"])].sort_values("threshold")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["recall", "noncoco_recall"], ["Recall of COCO objects", "Recall of non-COCO objects (LVIS)"]):
        for det, color in zip(DETECTORS, ["#4C72B0", "#DD8452"]):
            g = t[t.detector == det].sort_values("boxes_per_image")
            ax.plot(g.boxes_per_image, g[col], "-o", color=color, label=det)
            for _, r in g.iterrows():
                ax.annotate(f"{r.threshold:g}", (r.boxes_per_image, r[col]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("Boxes per image")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_recall_vs_boxes.png"), dpi=150)
    plt.close(fig)


def main():
    """
    ex022b を実行する．

    Returns:
        None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data()
    df = run(data)
    plot(df)
    cols = ["condition", "detector", "threshold", "boxes_per_image", "recall", "recall_person", "recall_other",
            "box_no_gt_rate", "noncoco_recall"]
    for split in SPLITS:
        print(f"== {split}")
        print(df[df.split == split][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
