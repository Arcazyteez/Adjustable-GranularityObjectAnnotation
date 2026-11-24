"""
ex022：YOLO11x と DETR の検出（物体の領域）を COCO の test で比較する（order_038）．

ex018（YOLO11x と OWLv2 の比較）と同じ形式の比較画像を，YOLO11x と DETR について作る．
新しく検出器は動かさず，ex017 で保存した COCO val2017 の test 500枚の検出結果を用いる
（ex019〜ex021 で用いたものと同一）．

- YOLO11x：conf 0.25 以上のボックス（Ultralytics の既定値）
- DETR：スコア 0.5 以上のボックス
- 正解：COCO の注釈（iscrowd=0）．クラスを問わず，IoU 0.5 以上のボックスがあれば「検出できた」とする（ex017 の E2 と同じ定義）

出力：outputs/ex022_yolo_detr_region_check/
  table_gt_status.csv（正解を「両方」「YOLO だけ」「DETR だけ」「両方が見逃し」に分けた集計，人物・人物以外・全体）
  table_detector.csv（検出器ごとのボックス数，再現率，正解に当たらないボックスの割合）
  fig_examples_detr_gain.png，fig_examples_yolo_gain.png（比較画像），fig_gt_status.png（集計の図）

order_039：論文の検出器の比較の表のため，OWLv2 の物体らしさ（ボックス数 20，50．ex019 の検出結果と，ex021 の E3_detector が
dev で決めた閾値）を加えた4つの検出器の表 table_detector_all.csv（ボックス数，再現率（全体，人物，人物以外），
正解に当たらないボックスの割合，検出の時間）を追加した．

実行：.venv_pytorch/bin/python programs/ex022_yolo_detr_region_check/train.py
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EX017_DIR = os.path.join(PROJECT_ROOT, "programs", "ex017_coco_granular_annotation")
EX017_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "ex017_coco_granular_annotation")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ex022_yolo_detr_region_check")

DETECTORS = {"YOLO11x": 0.25, "DETR": 0.5}   # 検出器 → スコアの閾値（ex017〜ex021 と同じ）
IOU_THRESHOLD = 0.5
STATUSES = ["both", "yolo_only", "detr_only", "neither"]
N_EXAMPLES = 6
EX019_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "ex019_coco_owl_alphaclip_wordnet")
EX021_E3_DIR = os.path.join(PROJECT_ROOT, "outputs", "ex021_coco_owl_rejection", "E3_detector")
OWL_BUDGETS = [20, 50]


def load_ex017_data():
    """
    ex017 の data モジュール（COCO の読み込み，分割，正解，画像）を読み込む．

    Returns:
        module: ex017 の data．
    """
    sys.path.insert(0, EX017_DIR)
    cwd = os.getcwd()
    os.chdir(EX017_DIR)
    try:
        import data
    finally:
        os.chdir(cwd)
    return data


def box_iou(a, b):
    """
    2組のボックスの IoU を計算する．

    Args:
        a (np.ndarray): xyxy 形式のボックス，形状 (N, 4)．
        b (np.ndarray): xyxy 形式のボックス，形状 (M, 4)．

    Returns:
        np.ndarray: IoU，形状 (N, M)．
    """
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def covered(a, b, thr=IOU_THRESHOLD):
    """
    a の各ボックスに，b のいずれかのボックスが IoU thr 以上で重なるかを返す．

    Args:
        a (np.ndarray): xyxy 形式のボックス，形状 (N, 4)．
        b (np.ndarray): xyxy 形式のボックス，形状 (M, 4)．
        thr (float): IoU の閾値．

    Returns:
        np.ndarray: 真偽値，形状 (N,)．
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    return (box_iou(a, b) >= thr).any(axis=1)


def load_detections():
    """
    ex017 で保存した test の検出結果を読み込み，閾値以上のボックスだけを残す．

    Returns:
        dict[str, dict[int, dict]]: 検出器 → 画像 ID → {"boxes": (K, 4) の np.ndarray, "labels": 長さ K の list[str]}．
    """
    out = {}
    for name, thr in DETECTORS.items():
        raw = torch.load(os.path.join(EX017_DATASET_DIR, f"detections_{name}_test.pt"), weights_only=False)
        out[name] = {}
        for image_id, det in raw.items():
            keep = np.asarray(det["scores"]) >= thr
            out[name][image_id] = {"boxes": np.asarray(det["boxes"]).reshape(-1, 4)[keep],
                                   "labels": [l for l, k in zip(det["labels"], keep) if k]}
    return out


def image_record(gt_boxes, gt_names, yolo_boxes, detr_boxes):
    """
    1枚の画像について，正解とボックスの対応を判定する．

    Args:
        gt_boxes (np.ndarray): 正解のボックス，形状 (G, 4)．
        gt_names (list[str]): 正解のクラス名，長さ G．
        yolo_boxes (np.ndarray): YOLO11x のボックス，形状 (Ky, 4)．
        detr_boxes (np.ndarray): DETR のボックス，形状 (Kd, 4)．

    Returns:
        dict: "gt_yolo"，"gt_detr"（形状 (G,) の真偽値），"gt_person"（形状 (G,) の真偽値），
            "yolo_hit"，"detr_hit"（各ボックスが正解に当たるか，形状 (Ky,)，(Kd,)）．
    """
    return {"gt_yolo": covered(gt_boxes, yolo_boxes), "gt_detr": covered(gt_boxes, detr_boxes),
            "gt_person": np.array([n == "person" for n in gt_names], dtype=bool),
            "yolo_hit": covered(yolo_boxes, gt_boxes), "detr_hit": covered(detr_boxes, gt_boxes)}


def build_records(data, coco, image_ids, dets):
    """
    全ての画像の判定を作る．

    Args:
        data (module): ex017 の data．
        coco (dict): COCO の注釈．
        image_ids (list[int]): 画像 ID，長さ N．
        dets (dict): load_detections の戻り値．

    Returns:
        list[dict]: image_record の戻り値，長さ N（image_ids と同じ順序）．
    """
    records = []
    for image_id in tqdm(image_ids, desc="records", leave=False):
        gt_boxes, gt_names = data.gt_instances(coco, image_id)
        records.append(image_record(gt_boxes, gt_names, dets["YOLO11x"][image_id]["boxes"],
                                    dets["DETR"][image_id]["boxes"]))
    return records


def gt_status_table(records):
    """
    正解を「両方」「YOLO だけ」「DETR だけ」「両方が見逃し」に分けて数える．

    Args:
        records (list[dict]): build_records の戻り値．

    Returns:
        pd.DataFrame: 行が group（person，non-person，all），列が件数と割合，形状 (3, 10)．
    """
    y = np.concatenate([r["gt_yolo"] for r in records])
    d = np.concatenate([r["gt_detr"] for r in records])
    p = np.concatenate([r["gt_person"] for r in records])
    rows = []
    for group, mask in [("person", p), ("non-person", ~p), ("all", np.ones_like(p))]:
        counts = {"both": int((y & d & mask).sum()), "yolo_only": int((y & ~d & mask).sum()),
                  "detr_only": int((~y & d & mask).sum()), "neither": int((~y & ~d & mask).sum())}
        n = int(mask.sum())
        rows.append({"group": group, "n_gt": n, **counts, **{f"{k}_rate": v / n for k, v in counts.items()},
                     "recall_yolo": float(y[mask].mean()), "recall_detr": float(d[mask].mean())})
    return pd.DataFrame(rows)


def detector_table(records, dets, image_ids):
    """
    検出器ごとのボックス数，再現率，正解に当たらないボックスの割合をまとめる．

    Args:
        records (list[dict]): build_records の戻り値．
        dets (dict): load_detections の戻り値．
        image_ids (list[int]): 画像 ID．

    Returns:
        pd.DataFrame: 行が検出器，形状 (2, 5)．
    """
    rows = []
    for name, key in [("YOLO11x", "yolo"), ("DETR", "detr")]:
        hits = np.concatenate([r[f"{key}_hit"] for r in records])
        gt = np.concatenate([r[f"gt_{key}"] for r in records])
        rows.append({"detector": name, "threshold": DETECTORS[name],
                     "boxes_per_image": len(hits) / len(image_ids), "recall": float(gt.mean()),
                     "box_no_gt_rate": float((~hits).mean())})
    return pd.DataFrame(rows)


def owl_threshold(budget):
    """
    ex021 の E3_detector が dev のスコアから決めた OWLv2 の物体らしさの閾値を読む．

    Args:
        budget (int): 1枚あたりの平均ボックス数．

    Returns:
        float: 閾値．
    """
    path = os.path.join(EX021_E3_DIR, f"OWLv2@{budget}_Alpha-crop_proposed", "summary.json")
    return float(json.load(open(path))["owl_threshold"])


def load_all_detectors():
    """
    YOLO11x，DETR，OWLv2@20，OWLv2@50 の test の検出結果を読み込み，閾値以上のボックスだけを残す．

    Returns:
        dict[str, dict[int, dict]]: 検出器 → 画像 ID → {"boxes": (K, 4) の np.ndarray, "time": float}．
    """
    out = {}
    raw = {name: torch.load(os.path.join(EX017_DATASET_DIR, f"detections_{name}_test.pt"), weights_only=False)
           for name in DETECTORS}
    raw_owl = torch.load(os.path.join(EX019_DATASET_DIR, "detections_OWLv2_test.pt"), weights_only=False)
    specs = [(name, raw[name], thr) for name, thr in DETECTORS.items()]
    specs += [(f"OWLv2@{b}", raw_owl, owl_threshold(b)) for b in OWL_BUDGETS]
    for name, dets, thr in specs:
        out[name] = {i: {"boxes": np.asarray(d["boxes"]).reshape(-1, 4)[np.asarray(d["scores"]) >= thr],
                         "time": float(d["time"]), "threshold": thr} for i, d in dets.items()}
    return out


def detector_table_all(data, coco, image_ids):
    """
    4つの検出器のボックス数，再現率（全体，人物，人物以外），正解に当たらないボックスの割合，時間をまとめる．

    Args:
        data (module): ex017 の data．
        coco (dict): COCO の注釈．
        image_ids (list[int]): 画像 ID，長さ N．

    Returns:
        pd.DataFrame: 行が検出器，形状 (4, 8)．
    """
    dets = load_all_detectors()
    rows = []
    for name, by_image in dets.items():
        found, person, hits = [], [], []
        for i in image_ids:
            gt_boxes, gt_names = data.gt_instances(coco, i)
            found.append(covered(gt_boxes, by_image[i]["boxes"]))
            person.append(np.array([n == "person" for n in gt_names], dtype=bool))
            hits.append(covered(by_image[i]["boxes"], gt_boxes))
        found, person, hits = np.concatenate(found), np.concatenate(person), np.concatenate(hits)
        rows.append({"detector": name, "threshold": next(iter(by_image.values()))["threshold"],
                     "boxes_per_image": len(hits) / len(image_ids), "recall": float(found.mean()),
                     "recall_person": float(found[person].mean()), "recall_other": float(found[~person].mean()),
                     "box_no_gt_rate": float((~hits).mean()),
                     "time_sec": float(np.mean([by_image[i]["time"] for i in image_ids]))})
    return pd.DataFrame(rows)


def draw_boxes(ax, boxes, color, lw=2.0, ls="-", labels=None):
    """
    ボックスとラベルを描く．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        boxes (np.ndarray): xyxy 形式のボックス，形状 (K, 4)．
        color (str): 色．
        lw (float): 線の太さ．
        ls (str): 線の種類．
        labels (list[str] | None): ボックスごとのラベル，長さ K．

    Returns:
        None
    """
    for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=ls))
        if labels:
            ax.text(x0 + 2, y0 + 2, labels[k], color="white", fontsize=7, va="top",
                    bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "lw": 0})


def panel(ax, image, gt_boxes, gt_names, own, other, title):
    """
    1つの検出器の結果を色分けして描く．
    - 青：もう一方の検出器も見つけた正解に当たるボックス
    - 橙：もう一方の検出器が見逃した正解に当たるボックス（正解のクラス名）
    - 灰の細線：正解に当たらないボックス
    - 赤の破線：この検出器が見逃した正解（正解のクラス名）

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        image (PIL.Image.Image): 画像．
        gt_boxes (np.ndarray): 正解のボックス，形状 (G, 4)．
        gt_names (list[str]): 正解のクラス名，長さ G．
        own (dict): この検出器の {"boxes": (K, 4), "labels": 長さ K}．
        other (dict): もう一方の検出器の {"boxes": (K', 4), "labels": 長さ K'}．
        title (str): 題名．

    Returns:
        None
    """
    boxes, labels = own["boxes"], own["labels"]
    gt_found_other = covered(gt_boxes, other["boxes"])
    gt_missed = ~covered(gt_boxes, boxes)
    iou = box_iou(boxes, gt_boxes) if len(gt_boxes) else np.zeros((len(boxes), 0))
    hit_any = (iou >= IOU_THRESHOLD).any(axis=1)
    hit_unique = ((iou >= IOU_THRESHOLD) & ~gt_found_other[None, :]).any(axis=1)
    unique_labels = [gt_names[int(np.argmax(np.where(gt_found_other, -1, iou[k])))]
                     for k in np.where(hit_unique)[0]]
    ax.imshow(image)
    ax.axis("off")
    draw_boxes(ax, boxes[~hit_any], "lightgray", lw=0.8)
    draw_boxes(ax, boxes[hit_any & ~hit_unique], "tab:blue",
               labels=[l for l, h, u in zip(labels, hit_any, hit_unique) if h and not u])
    draw_boxes(ax, gt_boxes[gt_missed], "tab:red", lw=1.5, ls="--",
               labels=[n for n, m in zip(gt_names, gt_missed) if m])
    draw_boxes(ax, boxes[hit_unique], "tab:orange", lw=2.5, labels=unique_labels)
    ax.set_title(f"{title}  ({len(boxes)} boxes, only this: {int(hit_unique.sum())}, missed GT: {int(gt_missed.sum())})",
                 fontsize=9)


def draw_examples(data, coco, image_ids, dets, records, mode, n=N_EXAMPLES):
    """
    比較画像を保存する．mode="detr_gain" は DETR だけが検出した正解が多い画像，
    mode="yolo_gain" は YOLO だけが検出した正解が多い画像を選ぶ（同数なら正解の多い画像）．

    Args:
        data (module): ex017 の data．
        coco (dict): COCO の注釈．
        image_ids (list[int]): 画像 ID，長さ N．
        dets (dict): load_detections の戻り値．
        records (list[dict]): build_records の戻り値，長さ N．
        mode (str): "detr_gain" または "yolo_gain"．
        n (int): 画像の枚数．

    Returns:
        list[int]: 選んだ画像 ID，長さ n．
    """
    if mode == "detr_gain":
        score = [int((~r["gt_yolo"] & r["gt_detr"]).sum()) for r in records]
    else:
        score = [int((r["gt_yolo"] & ~r["gt_detr"]).sum()) for r in records]
    order = sorted(range(len(records)), key=lambda j: (-score[j], -len(records[j]["gt_yolo"]), image_ids[j]))
    chosen = [image_ids[j] for j in order[:n]]
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.2 * n))
    for row, image_id in enumerate(chosen):
        image = data.load_image(coco, image_id)
        gt_boxes, gt_names = data.gt_instances(coco, image_id)
        y, d = dets["YOLO11x"][image_id], dets["DETR"][image_id]
        panel(axes[row, 0], image, gt_boxes, gt_names, y, d, f"YOLO11x (conf 0.25)  id={image_id}")
        panel(axes[row, 1], image, gt_boxes, gt_names, d, y, "DETR (thr 0.5)")
    legend = [mpatches.Patch(color=c, label=t) for c, t in [
        ("tab:blue", "box on GT found by both"), ("tab:orange", "box on GT missed by the other detector"),
        ("tab:red", "GT missed by this detector (dashed)"), ("lightgray", "box not matching any GT")]]
    fig.legend(handles=legend, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(OUTPUT_DIR, f"fig_examples_{mode}.png"), dpi=110)
    plt.close(fig)
    return chosen


def plot_status(table):
    """
    正解の内訳（両方，YOLO だけ，DETR だけ，両方が見逃し）を積み上げ棒グラフにする．

    Args:
        table (pd.DataFrame): gt_status_table の戻り値．

    Returns:
        None
    """
    colors = {"both": "tab:blue", "yolo_only": "tab:green", "detr_only": "tab:orange", "neither": "tab:red"}
    names = {"both": "found by both", "yolo_only": "YOLO11x only", "detr_only": "DETR only", "neither": "missed by both"}
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, row in enumerate(table.itertuples()):
        left = 0.0
        for s in STATUSES:
            v = getattr(row, f"{s}_rate")
            ax.barh(i, v, left=left, color=colors[s], label=names[s] if i == 0 else None)
            if v > 0.03:
                ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
            left += v
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(table["group"], table["n_gt"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of COCO GT (test, IoU ≥ 0.5, class-agnostic)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_gt_status.png"), dpi=150)
    plt.close(fig)


def main():
    """ex022 の全ての処理を実行する．"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_ex017_data()
    coco = data.load_coco()
    image_ids = data.split_image_ids(coco)["test"]
    dets = load_detections()
    records = build_records(data, coco, image_ids, dets)
    status = gt_status_table(records)
    det_table = detector_table(records, dets, image_ids)
    status.to_csv(os.path.join(OUTPUT_DIR, "table_gt_status.csv"), index=False)
    det_table.to_csv(os.path.join(OUTPUT_DIR, "table_detector.csv"), index=False)
    plot_status(status)
    if os.path.exists(EX021_E3_DIR):
        table_all = detector_table_all(data, coco, image_ids)
        table_all.to_csv(os.path.join(OUTPUT_DIR, "table_detector_all.csv"), index=False)
        print(table_all.round(3).to_string(index=False))
    for mode in ("detr_gain", "yolo_gain"):
        print(mode, draw_examples(data, coco, image_ids, dets, records, mode))
    print(det_table.round(3).to_string(index=False))
    print(status.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
