"""
ex018：OWLが画像の物体の領域を検出（クロップ）できるかを，YOLO11xと比較して確認する（order_028）．

確認すること：
1. YOLOが検出できる領域をOWLも検出できるか
   - YOLOのボックス（conf 0.25）のうち，OWLのボックスが IoU 0.5 以上で重なる割合
   - YOLOが検出したLVISのGTのうち，OWLも検出したGTの割合 P(OWL | YOLO検出)
2. YOLOが検出できない領域をOWLが検出できるか
   - YOLOが見逃したLVISのGTのうち，OWLが検出したGTの割合 P(OWL | YOLO見逃し)（COCOの物体とCOCO外の物体で分ける）
OWLの条件：OWL-ViT（COCO 80クラスのクエリ，ex017のキャッシュ，閾値0.1以上のみ），OWLv2のCOCO 80クラスのクエリ，
OWLv2の物体らしさ（クラス非依存）．OWLのボックス数は閾値で変わるため，1枚あたりの平均ボックス数（予算）を
揃えた閾値と，閾値の掃引で比較する．GTを見て閾値を選ばない（予算はボックスのスコアのみから決める）．

出力：outputs/ex018_owl_region_check/{条件}/{動作点}/0/metrics.json，curve.csv，比較画像（fig_examples_*.png）．
"""
import json
import os
import sys
import time

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from data import (DATASET_DIR, OUTPUT_DIR, PROJECT_ROOT, gt_arrays, load_coco, load_ex017_detections,  # noqa: E402
                  load_image, load_lvis, test_image_ids)
from model import OWLv2Detector, covered  # noqa: E402

sys.path.append(os.path.join(PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
from machine_learning_utils import set_seed  # noqa: E402

SEED = 0
YOLO_CONF = 0.25                     # Ultralyticsの既定値（ex017と同じ）
IOU_THRESHOLD = 0.5
BUDGETS = [10, 20, 50]               # 1枚あたりの平均ボックス数を揃える動作点
FIXED_THRESHOLDS = {"OWL-ViT_query": 0.1, "OWLv2_query": 0.1, "OWLv2_objectness": 0.1}
CONDITIONS = ["OWL-ViT_query", "OWLv2_query", "OWLv2_objectness"]
EXAMPLE_CONDITION = "OWLv2_objectness"
EXAMPLE_BUDGET = 20
N_BOOTSTRAP = 1000
KINDS = ["coco", "noncoco", "all"]


def save_json(obj, path):
    """
    辞書をJSONとして保存する．

    Args:
        obj (dict): 保存する辞書．
        path (str): 保存先．
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def owlv2_detections(coco, image_ids):
    """
    OWLv2 の検出結果を求める（キャッシュがあれば読み込む）．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．

    Returns:
        dict[str, dict[int, dict]]: "OWLv2_query"，"OWLv2_objectness" → 画像ID → {"boxes"，"scores"}．
    """
    path = os.path.join(DATASET_DIR, "detections_OWLv2_test.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    set_seed(SEED)
    detector = OWLv2Detector(coco["class_names"])
    result = {"OWLv2_query": {}, "OWLv2_objectness": {}, "time": {}}
    for i in tqdm(image_ids, desc="OWLv2", leave=False):
        start = time.perf_counter()
        det = detector.detect(load_image(coco, i))
        result["time"][i] = time.perf_counter() - start
        result["OWLv2_query"][i] = det["query"]
        result["OWLv2_objectness"][i] = det["objectness"]
    os.makedirs(DATASET_DIR, exist_ok=True)
    torch.save(result, path)
    return result


def load_all_detections(coco, image_ids):
    """
    YOLO11x と各OWLの条件の検出結果を読み込む．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．

    Returns:
        tuple[dict, dict]: YOLO（画像ID → {"boxes"，"scores"，"labels"}），OWLの条件 → 画像ID → {"boxes"，"scores"}．
    """
    yolo = load_ex017_detections("YOLO11x")
    owl = {"OWL-ViT_query": load_ex017_detections("OWL-ViT")}
    owl.update({k: v for k, v in owlv2_detections(coco, image_ids).items() if k != "time"})
    return yolo, owl


def budget_threshold(dets, image_ids, budget):
    """
    1枚あたりの平均ボックス数が budget になるスコアの閾値を求める．

    Args:
        dets (dict[int, dict]): 画像ID → {"scores"}．
        image_ids (list[int]): 画像ID．
        budget (int): 1枚あたりの平均ボックス数．

    Returns:
        float: 閾値（全画像のスコアの上位 budget×画像数 番目の値）．ボックスが足りない場合は最小のスコア．
    """
    scores = np.sort(np.concatenate([np.asarray(dets[i]["scores"]) for i in image_ids]))[::-1]
    return float(scores[min(budget * len(image_ids), len(scores)) - 1])


def image_record(gt, yolo_boxes, owl_boxes):
    """
    1枚の画像について，GTとボックスの重なりを判定する．

    Args:
        gt (tuple): gt_arrays の戻り値（ボックス G×4，種類，カテゴリ名）．
        yolo_boxes (np.ndarray): YOLOのボックス（Y×4）．
        owl_boxes (np.ndarray): OWLのボックス（O×4）．

    Returns:
        dict: "gt_kind"（G），"gt_yolo"・"gt_owl"（bool，G），"yolo_by_owl"（bool，Y），"yolo_tp"（bool，Y），
              "owl_by_yolo"（bool，O），"owl_gt"（bool，O：いずれかのGTと重なる），"n_owl"．
    """
    gt_boxes, kinds, _ = gt
    return {"gt_kind": np.array(kinds), "gt_yolo": covered(gt_boxes, yolo_boxes, IOU_THRESHOLD),
            "gt_owl": covered(gt_boxes, owl_boxes, IOU_THRESHOLD),
            "yolo_by_owl": covered(yolo_boxes, owl_boxes, IOU_THRESHOLD),
            "yolo_tp": covered(yolo_boxes, gt_boxes, IOU_THRESHOLD),
            "owl_by_yolo": covered(owl_boxes, yolo_boxes, IOU_THRESHOLD),
            "owl_gt": covered(owl_boxes, gt_boxes, IOU_THRESHOLD), "n_owl": len(owl_boxes)}


def ratio(num, den):
    """
    0除算を避けた割合．

    Args:
        num (float): 分子．
        den (float): 分母．

    Returns:
        float: num / den（den が0の場合は nan）．
    """
    return float(num / den) if den > 0 else float("nan")


def metrics_func(records):
    """
    画像ごとの判定から，YOLOとOWLの重なりの指標を求める．

    Args:
        records (list[dict]): image_record の戻り値のリスト．

    Returns:
        dict: 平均ボックス数，YOLOのボックスの被覆率，GTの種類ごとの4分割の件数と条件付きの検出率など．
    """
    cat = {k: np.concatenate([r[k] for r in records]) if records else np.zeros(0) for k in records[0] if k != "n_owl"}
    m = {"n_images": len(records), "mean_owl_boxes": float(np.mean([r["n_owl"] for r in records])),
         "n_yolo_boxes": int(len(cat["yolo_by_owl"])),
         "yolo_box_covered_by_owl": ratio(cat["yolo_by_owl"].sum(), len(cat["yolo_by_owl"])),
         "yolo_tp_box_covered_by_owl": ratio((cat["yolo_by_owl"] & cat["yolo_tp"]).sum(), cat["yolo_tp"].sum()),
         "owl_box_not_by_yolo": ratio((~cat["owl_by_yolo"]).sum(), len(cat["owl_by_yolo"])),
         "owl_only_box_matches_gt": ratio((~cat["owl_by_yolo"] & cat["owl_gt"]).sum(), (~cat["owl_by_yolo"]).sum())}
    for kind in KINDS:
        sel = np.ones(len(cat["gt_kind"]), bool) if kind == "all" else cat["gt_kind"] == kind
        y, o = cat["gt_yolo"][sel], cat["gt_owl"][sel]
        counts = {"both": int((y & o).sum()), "yolo_only": int((y & ~o).sum()),
                  "owl_only": int((~y & o).sum()), "neither": int((~y & ~o).sum())}
        m[kind] = {"n_gt": int(sel.sum()), **counts, "recall_yolo": ratio(y.sum(), sel.sum()),
                   "recall_owl": ratio(o.sum(), sel.sum()), "recall_union": ratio((y | o).sum(), sel.sum()),
                   "owl_given_yolo_hit": ratio(counts["both"], y.sum()),
                   "owl_given_yolo_miss": ratio(counts["owl_only"], (~y).sum())}
    return m


def bootstrap_ci(records, keys, n=N_BOOTSTRAP, seed=SEED):
    """
    画像単位のブートストラップで指標の95%信頼区間を求める．

    Args:
        records (list[dict]): image_record の戻り値のリスト．
        keys (list[tuple[str, str | None]]): (GTの種類 または None, 指標名) のリスト．
        n (int): 再標本化の回数．
        seed (int): 乱数シード．

    Returns:
        dict[str, list[float]]: "種類/指標名" → [下限, 上限]．
    """
    rng = np.random.default_rng(seed)
    values = {k: [] for k in keys}
    for _ in tqdm(range(n), desc="bootstrap", leave=False):
        m = metrics_func([records[j] for j in rng.integers(0, len(records), len(records))])
        for kind, name in keys:
            values[(kind, name)].append(m[kind][name] if kind else m[name])
    return {f"{kind}/{name}" if kind else name: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for (kind, name), v in values.items()}


def evaluate(lvis, image_ids, yolo, owl_dets, threshold, with_ci=False):
    """
    OWLのスコアの閾値を1つ決めて，全画像の指標を求める．

    Args:
        lvis (dict): load_lvis の戻り値．
        image_ids (list[int]): 画像ID．
        yolo (dict): YOLOの検出結果．
        owl_dets (dict): OWLの1つの条件の検出結果．
        threshold (float): OWLのスコアの閾値．
        with_ci (bool): ブートストラップの信頼区間も求めるか．

    Returns:
        tuple[dict, list[dict]]: 指標，画像ごとの判定．
    """
    records = []
    for i in image_ids:
        y = np.asarray(yolo[i]["boxes"]).reshape(-1, 4)[np.asarray(yolo[i]["scores"]) >= YOLO_CONF]
        o = np.asarray(owl_dets[i]["boxes"]).reshape(-1, 4)[np.asarray(owl_dets[i]["scores"]) >= threshold]
        records.append(image_record(gt_arrays(lvis, i), y, o))
    m = metrics_func(records)
    m["threshold"] = threshold
    if with_ci:
        keys = [(None, "yolo_box_covered_by_owl")] + [(k, n) for k in KINDS for n in
                                                      ("owl_given_yolo_hit", "owl_given_yolo_miss", "recall_owl")]
        m["ci95"] = bootstrap_ci(records, keys)
    return m, records


def run_operating_points(lvis, image_ids, yolo, owl):
    """
    各OWLの条件について，予算を揃えた閾値と固定の閾値で評価して保存する．

    Args:
        lvis (dict): load_lvis の戻り値．
        image_ids (list[int]): 画像ID．
        yolo (dict): YOLOの検出結果．
        owl (dict): OWLの条件 → 検出結果．

    Returns:
        list[dict]: 各動作点の指標（"condition"，"operating_point" を含む）．
    """
    rows = []
    for cond in CONDITIONS:
        points = {f"budget{b}": budget_threshold(owl[cond], image_ids, b) for b in BUDGETS}
        points[f"thr{FIXED_THRESHOLDS[cond]}"] = FIXED_THRESHOLDS[cond]
        for name, thr in points.items():
            out = os.path.join(OUTPUT_DIR, cond, name, str(SEED))
            path = os.path.join(out, "metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    rows.append(json.load(f))
                continue
            m, _ = evaluate(lvis, image_ids, yolo, owl[cond], thr, with_ci=True)
            m.update({"condition": cond, "operating_point": name})
            save_json(m, path)
            save_json({"condition": cond, "operating_point": name, "threshold": thr, "yolo_conf": YOLO_CONF,
                       "iou_threshold": IOU_THRESHOLD, "gt": "LVIS v1 val", "images": "ex017 test ∩ LVIS",
                       "n_images": len(image_ids), "seed": SEED}, os.path.join(out, "config.json"))
            rows.append(m)
    return rows


def run_curves(lvis, image_ids, yolo, owl):
    """
    OWLの閾値を掃引し，平均ボックス数と条件付きの検出率の関係を CSV に保存する．

    Args:
        lvis (dict): load_lvis の戻り値．
        image_ids (list[int]): 画像ID．
        yolo (dict): YOLOの検出結果．
        owl (dict): OWLの条件 → 検出結果．
    """
    path = os.path.join(OUTPUT_DIR, "curve.csv")
    if os.path.exists(path):
        return
    lines = ["condition,threshold,mean_owl_boxes,yolo_box_covered_by_owl,"
             + ",".join(f"{k}_{n}" for k in KINDS for n in ("owl_given_yolo_hit", "owl_given_yolo_miss", "recall_owl"))]
    for cond in CONDITIONS:
        budgets = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
        thresholds = sorted({budget_threshold(owl[cond], image_ids, b) for b in budgets}, reverse=True)
        for thr in tqdm(thresholds, desc=f"curve {cond}", leave=False):
            m, _ = evaluate(lvis, image_ids, yolo, owl[cond], thr)
            vals = [m[k][n] for k in KINDS for n in ("owl_given_yolo_hit", "owl_given_yolo_miss", "recall_owl")]
            lines.append(",".join(map(str, [cond, thr, m["mean_owl_boxes"], m["yolo_box_covered_by_owl"], *vals])))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def draw_boxes(ax, boxes, color, lw=2.0, ls="-", labels=None):
    """
    ボックスとラベルを描く．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        boxes (np.ndarray): xyxy形式のボックス（K×4）．
        color (str): 色．
        lw (float): 線の太さ．
        ls (str): 線の種類．
        labels (list[str] | None): ボックスごとのラベル．
    """
    for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=ls))
        if labels:
            ax.text(x0 + 2, y0 + 2, labels[k], color="white", fontsize=7, va="top",
                    bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "lw": 0})


def example_panels(coco, lvis, image_id, yolo, owl_dets, threshold, axes):
    """
    1枚の画像について，左に YOLO，右に OWL の検出結果を色分けして描く．
    - 左：YOLOのボックス（青）．OWLが検出できなかったものは赤の破線
    - 右：YOLOのボックスと重なるOWLのボックス（緑），YOLOが見逃したGTに重なるOWLのボックス（橙，LVISのカテゴリ名），
      その他のOWLのボックス（灰の細線）

    Args:
        coco (dict): load_coco の戻り値．
        lvis (dict): load_lvis の戻り値．
        image_id (int): 画像ID．
        yolo (dict): YOLOの検出結果．
        owl_dets (dict): OWLの検出結果．
        threshold (float): OWLのスコアの閾値．
        axes (list[matplotlib.axes.Axes]): 描画先（2個）．
    """
    image = load_image(coco, image_id)
    gt_boxes, _, names = gt_arrays(lvis, image_id)
    keep = np.asarray(yolo[image_id]["scores"]) >= YOLO_CONF
    y_boxes = np.asarray(yolo[image_id]["boxes"]).reshape(-1, 4)[keep]
    y_labels = [l for l, k in zip(yolo[image_id]["labels"], keep) if k]
    o_boxes = np.asarray(owl_dets[image_id]["boxes"]).reshape(-1, 4)[np.asarray(owl_dets[image_id]["scores"]) >= threshold]
    y_cov = covered(y_boxes, o_boxes, IOU_THRESHOLD)
    o_by_y = covered(o_boxes, y_boxes, IOU_THRESHOLD)
    missed_gt = ~covered(gt_boxes, y_boxes, IOU_THRESHOLD)
    o_new = ~o_by_y & covered(o_boxes, gt_boxes[missed_gt], IOU_THRESHOLD)
    new_labels = []
    for b in o_boxes[o_new]:
        iou = covered(gt_boxes[missed_gt], b[None], IOU_THRESHOLD)
        new_labels.append(np.array(names)[missed_gt][iou][0])
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    draw_boxes(axes[0], y_boxes[y_cov], "tab:blue", labels=[l for l, c in zip(y_labels, y_cov) if c])
    draw_boxes(axes[0], y_boxes[~y_cov], "tab:red", ls="--", labels=[l for l, c in zip(y_labels, y_cov) if not c])
    draw_boxes(axes[1], o_boxes[~o_by_y & ~o_new], "lightgray", lw=0.8)
    draw_boxes(axes[1], o_boxes[o_by_y], "tab:green")
    draw_boxes(axes[1], o_boxes[o_new], "tab:orange", labels=new_labels)
    axes[0].set_title(f"YOLO11x  id={image_id}  ({len(y_boxes)} boxes, OWL missed {int((~y_cov).sum())})", fontsize=9)
    axes[1].set_title(f"OWLv2 objectness  ({len(o_boxes)} boxes, YOLO-missed GT found {int(o_new.sum())})",
                      fontsize=9)


def draw_examples(coco, lvis, image_ids, yolo, owl_dets, threshold, records, mode, n=6):
    """
    比較画像を保存する．mode="gain" はOWLだけが検出したCOCO外のGTが多い画像，
    mode="loss" はYOLOが検出したがOWLが見逃したGTが多い画像を選ぶ．

    Args:
        coco (dict): load_coco の戻り値．
        lvis (dict): load_lvis の戻り値．
        image_ids (list[int]): 画像ID．
        yolo (dict): YOLOの検出結果．
        owl_dets (dict): OWLの検出結果．
        threshold (float): OWLのスコアの閾値．
        records (list[dict]): evaluate が返した画像ごとの判定（image_ids と同じ順序）．
        mode (str): "gain" または "loss"．
        n (int): 画像の枚数．
    """
    path = os.path.join(OUTPUT_DIR, f"fig_examples_{mode}.png")
    if os.path.exists(path):
        return
    if mode == "gain":
        score = [int((~r["gt_yolo"] & r["gt_owl"] & (r["gt_kind"] == "noncoco")).sum()) for r in records]
    else:
        score = [int((r["gt_yolo"] & ~r["gt_owl"]).sum()) for r in records]
    chosen = [image_ids[j] for j in np.argsort(score, kind="stable")[::-1][:n]]
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.2 * n))
    for row, i in enumerate(chosen):
        example_panels(coco, lvis, i, yolo, owl_dets, threshold, axes[row])
    legend = [mpatches.Patch(color=c, label=t) for c, t in [
        ("tab:blue", "YOLO box also found by OWL"), ("tab:red", "YOLO box missed by OWL"),
        ("tab:green", "OWL box overlapping a YOLO box"), ("tab:orange", "OWL box on a YOLO-missed LVIS object"),
        ("lightgray", "other OWL box")]]
    fig.legend(handles=legend, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """ex018 の全ての処理を実行する．"""
    set_seed(SEED)
    coco, lvis = load_coco(), load_lvis()
    image_ids = test_image_ids(lvis)
    yolo, owl = load_all_detections(coco, image_ids)
    rows = run_operating_points(lvis, image_ids, yolo, owl)
    run_curves(lvis, image_ids, yolo, owl)
    thr = budget_threshold(owl[EXAMPLE_CONDITION], image_ids, EXAMPLE_BUDGET)
    _, records = evaluate(lvis, image_ids, yolo, owl[EXAMPLE_CONDITION], thr)
    for mode in ("gain", "loss"):
        draw_examples(coco, lvis, image_ids, yolo, owl[EXAMPLE_CONDITION], thr, records, mode)
    for m in rows:
        print(f"{m['condition']:18s} {m['operating_point']:9s} thr={m['threshold']:.3f} boxes={m['mean_owl_boxes']:.1f} "
              f"YOLObox→OWL={m['yolo_box_covered_by_owl']:.3f} "
              + " ".join(f"{k}: P(O|Y)={m[k]['owl_given_yolo_hit']:.3f} P(O|¬Y)={m[k]['owl_given_yolo_miss']:.3f}"
                         for k in ("coco", "noncoco")))


if __name__ == "__main__":
    main()
