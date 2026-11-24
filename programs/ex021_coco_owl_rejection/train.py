"""
ex021：新しい提案手法（OWLv2 + Alpha-CLIP + WordNet）の深さ0の棄却の判定の探索（order_033）．

学習は行わず，学習済みのモデルの推論と評価のみを行う．データは MSCOCO val2017 のみ（ex017・ex019・ex020 と同じ dev 500枚，test 500枚）．
結果は outputs/ex021_coco_owl_rejection/{実験}/{手法}/{条件}/{seed}/ に保存し，metrics.json がある条件はスキップする．

探索する条件（itertools.product の全ての組）：
- OWLv2 のボックス数 B ∈ {20, 50}（dev のスコアで決めた ex019 の閾値）
- Alpha-CLIP の入力 ∈ {context1.5, crop}
- 重複・部分の除去 ∈ {none, nms, nms_contain}（model.dedup_mask）
- 棄却のスコア ∈ {joint, objectness, vlm, mass, mass_joint}（model.rejection_score）
- θ：dev のスコアの分位点（41通り）から，dev の COCO の F値が最大のもの（"f1"）と，
  dev の細かい粒度の hF（vg_subtree）が最大のもの（"hf"）の2通りを選ぶ
- 深さ0のラベル ∈ {coco（COCO の80クラスの softmax が最大の語．ex019 と同じ），mass（VG の木で COCO のクラスの部分木の
  確率の合計が最大のクラス）}．mass は手法名の末尾 "_masslabel"（order_033 の実行中に，人物のボックスを残しても
  person と命名されない問題が分かったため追加した）
命名の条件（vg_subtree の T，τ．ex020 で選んだもの）は固定する．
注意：hF は正解に対応したボックスだけを評価し，誤検出では下がらない．そのため，hF で選ぶ θ は棄却しない方向に偏る．
総合の比較には COCO の F値で選んだ θ を用いる．

評価（test）：
- COCO の指標（ex017 の metrics_func）：対応したボックスの Top-1，クラスを含めた適合率・再現率・F値
- 細かい粒度の hF（ex020 の評価．VG.pdf の式(8)）：命名は coarse（COCO のクラス）と vg_subtree
- 人物の再現率：クラス非依存（IoU 0.5）と，person と命名された割合
比較：古い提案手法（DETR + CLIP-B/32，棄却は VLM の確信度）と YOLO11x 単体を同じコードで評価する．
差は画像単位の対応のあるブートストラップで検定する．

実験条件の履歴：
- order_033：初版（E1_rejection）．
- order_037：E2_factorial を追加した．ex019 の E2 の検出器 × F の比較を，棄却のスコアを部分木の確率の合計に変えて再評価する．
  検出器 ∈ {DETR，OWLv2@10，OWLv2@20，OWLv2@50} × F ∈ {CLIP-ViT-B32，Alpha-CLIP（context1.5，crop）}
  × 棄却のスコア ∈ {joint，mass，mass_joint} × 除去 ∈ {none，nms_contain} × 深さ0のラベル ∈ {coco，mass}（全ての組）．
  DETR の検出スコアは DETR 自身の確信度である．vg_subtree の T，τ は F ごとに ex020 の条件（CLIP-B/32 は Old，Alpha-CLIP は New）．
  `python train.py e2` で E2 だけを実行する．
- order_039：E3_detector を追加した．提案手法の検出器だけを替えた比較（D ∈ {OWLv2@20，OWLv2@50，YOLO11x，DETR}）と，
  F だけを替えた比較（F ∈ {CLIP-ViT-B32，Alpha-CLIP crop，Alpha-CLIP context1.5}）を，提案の判定（nms_contain，mass_joint，
  mass ラベル）と従来の判定（none，joint，coco ラベル）で評価する．YOLO11x と DETR は自身のクラスを用いず，ボックスと
  検出スコア（物体らしさの代わり）だけを用いる．`python train.py e3` で E3 だけを実行する．
  `python train.py e3lvis` で，E3 の test の残したボックスを，同じ画像の LVIS v1 の注釈（ex018 の data．474枚）と照合し，
  COCO 外の物体を囲む割合を求める（E3_detector/lvis_test.csv）．
"""
import sys
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data import DATASET_DIR, OUTPUT_DIR, load_module
from model import DEDUP_TYPES, SCORE_TYPES, dedup_mask, rejection_score

E17_DATA, E17_MODEL, E17_TRAIN = load_module("ex017")
E19_DATA, E19_MODEL, E19_TRAIN = load_module("ex019")
E20_DATA, E20_MODEL, E20_TRAIN = load_module("ex020")
save_json, box_iou, greedy_match, safe_div = E17_TRAIN.save_json, E17_TRAIN.box_iou, E17_TRAIN.greedy_match, E17_TRAIN.safe_div

SEED = 0
SPLITS = ["dev", "test"]
BUDGETS = [20, 50]
ALPHA_INPUTS = ["context1.5", "crop"]
N_THETA = 40
CRITERIA = ["f1", "hf"]
LABEL_TYPES = ["coco", "mass"]
N_QUALITATIVE = 6
E2_DETECTORS = [("DETR", None), ("OWLv2", 10), ("OWLv2", 20), ("OWLv2", 50)]   # order_037：(検出器, OWLv2 のボックス数)
E2_EMBEDDERS = ["CLIP-ViT-B32", "AlphaCLIP-ViT-B16_context1.5", "AlphaCLIP-ViT-B16_crop"]
E2_SCORES = ["joint", "mass", "mass_joint"]
E2_DEDUPS = ["none", "nms_contain"]
E3_DETECTORS = [("OWLv2", 20), ("OWLv2", 50), ("YOLO11x", None), ("DETR", None)]   # order_039
E3_JUDGES = {"proposed": ("nms_contain", "mass_joint", "mass"), "previous": ("none", "joint", "coco")}
E2_SHORT = {"CLIP-ViT-B32": "CLIP-B32", "AlphaCLIP-ViT-B16_context1.5": "Alpha-context1.5", "AlphaCLIP-ViT-B16_crop": "Alpha-crop"}


def result_dir(experiment, method, condition, seed=SEED):
    """
    結果の保存先 outputs/ex021_coco_owl_rejection/{実験}/{手法}/{条件}/{seed}/ を返す（作成する）．

    Args:
        experiment (str): 実験名．
        method (str): 手法名．
        condition (str): 条件名．
        seed (int): seed．

    Returns:
        str: ディレクトリのパス．
    """
    d = os.path.join(OUTPUT_DIR, experiment, method, condition, str(seed))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# ボックスごとの量の前計算
# ---------------------------------------------------------------------------
def box_quantities(namer, feats, subtree_params):
    """
    領域の特徴から，深さ0のラベル，確信度，部分木の確率の合計，vg_subtree の出力を求める．

    Args:
        namer: ex020 の TreeNamer．
        feats (torch.Tensor): 領域の特徴（K×D）．
        subtree_params (dict): vg_subtree の "T" と "tau"（ex020 で選んだもの）．

    Returns:
        dict: "top"（上位5つのクラス名のリスト），"conf"（K），"mass"（K），"out_coarse"，"out_sub"（(Synset 名, 語) のリスト），
              "top_mass"，"out_coarse_mass"（部分木の確率の合計によるクラスの順位と深さ0の出力）．
    """
    if len(feats) == 0:
        return {"top": [], "conf": np.zeros(0, np.float32), "mass": np.zeros(0, np.float32), "out_coarse": [], "out_sub": [],
                "top_mass": [], "out_coarse_mass": []}
    tree, cos = namer.tree, namer.cosine(feats)
    p0 = namer.softmax(cos[:, namer.coco_idx])
    coco_nodes = [namer.word_nodes[k] for k in namer.coco_idx]
    names = [tree["words"][n] for n in coco_nodes]
    order = np.argsort(-p0, axis=1)[:, :5]
    node_pos = {n: j for j, n in enumerate(namer.nodes)}
    mass = np.asarray((namer.subtree.T @ namer.softmax(cos, 1.0).T).T)[:, [node_pos[n] for n in coco_nodes]]
    sub = namer.subtree_mass(cos, subtree_params["T"], subtree_params["tau"])
    mass_all = mass
    order_mass = np.argsort(-mass_all, axis=1)[:, :5]
    return {"top": [[names[k] for k in row] for row in order], "conf": p0.max(axis=1).astype(np.float32),
            "top_mass": [[names[k] for k in row] for row in order_mass],
            "out_coarse_mass": [(coco_nodes[row[0]], names[row[0]]) for row in order_mass],
            "mass": mass.max(axis=1).astype(np.float32), "out_coarse": [(coco_nodes[row[0]], names[row[0]]) for row in order],
            "out_sub": [(tree["synset"][n], tree["words"][n] or n.removeprefix("frame:")) if n else (None, "") for n in sub]}


def build_source(pool, coco, splits, dets, thresholds, tree, det_name, emb_name, budget, subtree_params, split):
    """
    1つの (検出器, F, ボックス数) の画像ごとの量を求める（DATASET_DIR にキャッシュする）．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): ex019 の load_detections の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．
        tree (dict): ex020 の VG の木．
        det_name (str): "OWLv2" または "DETR"．
        emb_name (str): ex019 の EMBEDDER_SPECS のキー．
        budget (int | None): OWLv2 のボックス数（DETR は None）．
        subtree_params (dict): vg_subtree の条件．
        split (str): 分割の名前．

    Returns:
        list[dict]: 画像ごとの "image_id"，"gt_boxes"，"gt_labels"，"boxes"，"det_scores" と box_quantities の値．
    """
    path = os.path.join(DATASET_DIR, f"source_{det_name}{budget or ''}_{emb_name}_{split}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = dets[(det_name, split)]
    feats = E19_TRAIN.cached_features(pool, coco, emb_name, {i: x["boxes"] for i, x in d.items()}, f"{det_name}_{split}")
    namer = E20_MODEL.TreeNamer(pool.get(emb_name), tree, [s for _, s in E19_DATA.coco80_vocab()])
    out = []
    for i in tqdm(splits[split], desc=f"source {det_name}{budget or ''} {emb_name} {split}", leave=False):
        scores = np.asarray(d[i]["scores"], np.float32)
        n = len(scores) if budget is None else int((scores >= thresholds[budget]).sum())
        gt_boxes, gt_labels = E19_DATA.gt_instances(coco, i)
        out.append({"image_id": i, "gt_boxes": gt_boxes, "gt_labels": gt_labels,
                    "boxes": np.asarray(d[i]["boxes"][:n], np.float32).reshape(-1, 4), "det_scores": scores[:n],
                    **box_quantities(namer, feats[i]["feats"][:n], subtree_params)})
    os.makedirs(DATASET_DIR, exist_ok=True)
    torch.save(out, path)
    return out


def yolo_source(coco, splits, dets, split):
    """
    YOLO11x 単体の画像ごとの量（ラベルと出力は YOLO のクラス，スコアは検出スコア）．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): ex019 の load_detections の戻り値．
        split (str): 分割の名前．

    Returns:
        list[dict]: build_source と同じ形式．
    """
    synsets = E20_DATA.class_synsets()
    out = []
    for i in splits[split]:
        d = dets[("YOLO11x", split)][i]
        gt_boxes, gt_labels = E19_DATA.gt_instances(coco, i)
        s = np.asarray(d["scores"], np.float32)
        out.append({"image_id": i, "gt_boxes": gt_boxes, "gt_labels": gt_labels, "boxes": np.asarray(d["boxes"], np.float32).reshape(-1, 4),
                    "det_scores": s, "top": [[l] for l in d["labels"]], "conf": s, "mass": s, "top_mass": [[l] for l in d["labels"]],
                    "out_coarse_mass": [(synsets[l], l) for l in d["labels"]],
                    "out_coarse": [(synsets[l], l) for l in d["labels"]], "out_sub": [(synsets[l], l) for l in d["labels"]]})
    return out


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def with_label(source, label):
    """
    深さ0のラベルの種類に合わせて，"top" と "out_coarse" を差し替えたソースを返す．

    Args:
        source (list[dict]): build_source の戻り値．
        label (str): "coco" または "mass"．

    Returns:
        list[dict]: ソース（label = "coco" ならそのまま）．
    """
    if label == "coco":
        return source
    return [r | {"top": r["top_mass"], "out_coarse": r["out_coarse_mass"]} for r in source]


def config_scores(source, score_type, dedup):
    """
    棄却のスコアを求め，重複・部分の除去で除くボックスのスコアを −∞ にする．

    Args:
        source (list[dict]): build_source の戻り値．
        score_type (str): 棄却のスコアの種類．
        dedup (str): 重複・部分の除去の種類．

    Returns:
        list[np.ndarray]: 画像ごとのスコア（K）．
    """
    out = []
    for r in source:
        s = rejection_score(score_type, r["conf"], r["det_scores"], r["mass"]).astype(np.float64)
        keep = dedup_mask(r["boxes"], [t[0] for t in r["top"]], s, dedup)
        out.append(np.where(keep, s, -np.inf))
    return out


def coco_metrics(source, scores, theta):
    """
    COCO の指標（ex017 の metrics_func）．

    Args:
        source (list[dict]): build_source の戻り値．
        scores (list[np.ndarray]): config_scores の戻り値．
        theta (float): θ．

    Returns:
        tuple[dict, np.ndarray]: 指標と画像ごとの計数（N×6）．
    """
    records = [{"gt_boxes": r["gt_boxes"], "gt_labels": r["gt_labels"], "boxes": r["boxes"], "top": r["top"],
                "scores": s, "det_scores": r["det_scores"]} for r, s in zip(source, scores)]
    return E17_TRAIN.metrics_func(records, theta)


def fine_metrics(source, scores, theta, instances, key):
    """
    細かい粒度の hF（ex020 の評価）と人物の再現率を求める．

    Args:
        source (list[dict]): build_source の戻り値．
        scores (list[np.ndarray]): config_scores の戻り値．
        theta (float): θ．
        instances (dict): ex020 の fine_instances の戻り値．
        key (str): 出力の種類（"out_coarse" または "out_sub"）．

    Returns:
        tuple[dict, np.ndarray]: 指標（"hP"，"hR"，"hF"，"naming_acc"，"finer_acc"，"person_recall"，"person_named"）と
                                 画像ごとの計数（N×3）．
    """
    synsets = E20_DATA.class_synsets()
    counts, status = [], {"exact": 0, "finer": 0, "coarse_correct": 0, "wrong": 0, "none": 0}
    n_person = n_person_hit = n_person_named = 0
    for r, s in zip(source, scores):
        keep = np.where(s >= theta)[0]
        order = list(np.argsort(-r["det_scores"][keep], kind="stable"))
        pairs = {g: keep[p] for p, g in greedy_match(box_iou(r["boxes"][keep], r["gt_boxes"]), order).items()}
        c = np.zeros(3)
        for g, cls, fines in instances[r["image_id"]]:
            k = pairs.get(g)
            if cls == "person":
                n_person += 1
                n_person_hit += k is not None
                n_person_named += k is not None and r["top"][k][0] == "person"
            if not fines:
                continue
            res = E20_TRAIN.instance_score(r[key][k][0] if k is not None else None, fines, synsets[cls])
            c += [res["inter"], res["n_pred"], res["n_gt"]]
            status[res["status"]] += 1
        counts.append(c)
    counts = np.array(counts)
    n = sum(status.values())
    m = E20_TRAIN.hierarchical_metrics(counts.sum(0))
    m.update({"naming_acc": safe_div(status["exact"] + status["finer"] + status["coarse_correct"], n),
              "finer_acc": safe_div(status["exact"] + status["finer"], n), "rate_none": safe_div(status["none"], n),
              "person_recall": safe_div(n_person_hit, n_person), "person_named": safe_div(n_person_named, n_person)})
    return m, counts


def all_metrics(source, scores, theta, instances):
    """
    COCO の指標，細かい粒度の hF（coarse と vg_subtree），人物の再現率をまとめて求める．

    Args:
        source (list[dict]): build_source の戻り値．
        scores (list[np.ndarray]): config_scores の戻り値．
        theta (float): θ．
        instances (dict): ex020 の fine_instances の戻り値．

    Returns:
        tuple[dict, dict]: 指標（COCO の指標と "sub_*"，"coarse_*" の細かい粒度の指標，"kept_per_image"）と，
                           ブートストラップ用の計数（"coco"：N×6，"sub"：N×3，"coarse"：N×3）．
    """
    m, c_coco = coco_metrics(source, scores, theta)
    sub, c_sub = fine_metrics(source, scores, theta, instances, "out_sub")
    coarse, c_coarse = fine_metrics(source, scores, theta, instances, "out_coarse")
    keys = ("top1_acc", "precision", "recall", "f1", "mean_iou")
    out = {**{k: m[k] for k in keys}, **{f"sub_{k}": v for k, v in sub.items()},
           **{f"coarse_{k}": coarse[k] for k in ("hP", "hR", "hF")},
           "kept_per_image": float(np.mean([(s >= theta).sum() for s in scores]))}
    return out, {"coco": c_coco, "sub": c_sub, "coarse": c_coarse}


def theta_grid(scores):
    """
    θ の候補（−∞ と，残るボックスのスコアの分位点）．

    Args:
        scores (list[np.ndarray]): dev の config_scores の戻り値．

    Returns:
        list[float]: θ の候補．
    """
    s = np.concatenate(scores)
    s = s[np.isfinite(s)]
    return [-np.inf] + sorted(set(np.quantile(s, np.linspace(0, 0.98, N_THETA)).tolist()))


def evaluate_config(name, config, sources, instances, experiment="E1_rejection"):
    """
    1つの条件について，dev で θ を2つの基準で選び，dev と test の結果を保存する．

    Args:
        name (str): 手法名（ディレクトリ名）．
        config (dict): 条件（"source"，"score_type"，"dedup" など）．
        sources (dict[str, list[dict]]): 分割 → build_source の戻り値．
        instances (dict[str, dict]): 分割 → ex020 の fine_instances の戻り値．
        experiment (str): 実験名（保存先のディレクトリ）．

    Returns:
        dict: 基準 → {"theta"，"dev"，"test"} と計数（"counts"：基準 → 種類 → N×K）．
    """
    base = os.path.join(OUTPUT_DIR, experiment, name)
    path = os.path.join(base, "summary.json")
    sources = {s: with_label(v, config.get("label", "coco")) for s, v in sources.items()}
    scores = {s: config_scores(sources[s], config["score_type"], config["dedup"]) for s in SPLITS}
    if os.path.exists(path):
        summary = json.load(open(path))
        summary["counts"] = {c: {k: np.load(os.path.join(result_dir(experiment, name, f"{c}_test"), f"counts_{k}.npy"))
                                 for k in ("coco", "sub", "coarse")} for c in CRITERIA}
        return summary
    curve = []
    for theta in theta_grid(scores["dev"]):
        m, _ = all_metrics(sources["dev"], scores["dev"], theta, instances["dev"])
        curve.append({"theta": theta, **m})
    curve = pd.DataFrame(curve)
    os.makedirs(base, exist_ok=True)
    curve.to_csv(os.path.join(base, "theta_curve_dev.csv"), index=False)
    summary = {"method": name, **config, "counts": {}}
    for crit, col in (("f1", "f1"), ("hf", "sub_hF")):
        theta = float(curve.loc[curve[col].idxmax(), "theta"])
        summary[crit] = {"theta": theta}
        for split in SPLITS:
            m, counts = all_metrics(sources[split], scores[split], theta, instances[split])
            d = result_dir(experiment, name, f"{crit}_{split}")
            save_json(m, os.path.join(d, "metrics.json"))
            save_json({**config, "theta": theta, "selection": f"dev {col}", "split": split, "seed": SEED},
                      os.path.join(d, "config.json"))
            for k, v in counts.items():
                np.save(os.path.join(d, f"counts_{k}.npy"), v)
            summary[crit][split] = m
            if split == "test":
                summary["counts"][crit] = counts
    save_json({k: v for k, v in summary.items() if k != "counts"}, path)
    return summary


# ---------------------------------------------------------------------------
# 実験
# ---------------------------------------------------------------------------
def run_e1(pool, coco, splits, instances):
    """
    E1_rejection：OWLv2 の全ての条件と，古い提案手法・YOLO11x 単体を評価する．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → ex020 の fine_instances の戻り値．

    Returns:
        tuple[dict, dict]: 手法名 → evaluate_config の戻り値，手法名 → (分割 → ソース)．
    """
    dets, thresholds = E19_TRAIN.load_detections(coco, splits)
    tree = E20_DATA.build_tree()
    params = {m: json.load(open(os.path.join(E20_DATA.OUTPUT_DIR, "E2_pipeline", f"{m}+vg_subtree", "summary.json")))["params"]
              for m in ("New", "Old")}
    results, sources_all = {}, {}
    for budget, alpha in itertools.product(BUDGETS, ALPHA_INPUTS):
        emb = f"AlphaCLIP-ViT-B16_{alpha}"
        sources = {s: build_source(pool, coco, splits, dets, thresholds, tree, "OWLv2", emb, budget, params["New"], s) for s in SPLITS}
        sources_all[(f"OWLv2@{budget}", alpha)] = sources
        for dedup, score_type, label in tqdm(list(itertools.product(DEDUP_TYPES, SCORE_TYPES, LABEL_TYPES)),
                                             desc=f"configs @{budget} {alpha}", leave=False):
            name = f"New_OWLv2@{budget}_{alpha}_{dedup}_{score_type}" + ("_masslabel" if label == "mass" else "")
            config = {"source": f"OWLv2@{budget}", "alpha_input": alpha, "dedup": dedup, "score_type": score_type, "label": label,
                      "embedder": emb, "budget": budget, "owl_threshold": thresholds[budget], "subtree_params": params["New"]}
            results[name] = evaluate_config(name, config, sources, instances)
    old = {s: build_source(pool, coco, splits, dets, thresholds, tree, "DETR", "CLIP-ViT-B32", None, params["Old"], s) for s in SPLITS}
    for label in LABEL_TYPES:
        name = "Old_DETR_CLIP-B32_none_vlm" + ("_masslabel" if label == "mass" else "")
        results[name] = evaluate_config(name, {"source": "DETR", "dedup": "none", "score_type": "vlm", "label": label,
                                               "embedder": "CLIP-ViT-B32", "subtree_params": params["Old"]}, old, instances)
    yolo = {s: yolo_source(coco, splits, dets, s) for s in SPLITS}
    results["YOLO11x_closed"] = evaluate_config("YOLO11x_closed", {"source": "YOLO11x", "dedup": "none", "score_type": "objectness",
                                                                   "embedder": None}, yolo, instances)
    sources_all["Old"], sources_all["YOLO11x"] = old, yolo
    return results, sources_all


def run_e2(pool, coco, splits, instances):
    """
    E2_factorial（order_037）：検出器 × F × 棄却のスコア × 除去 × 深さ0のラベルの全ての組を評価する．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → ex020 の fine_instances の戻り値．

    Returns:
        dict: 手法名 → evaluate_config の戻り値（"detector"，"embedder" などの条件を含む）．
    """
    dets, thresholds = E19_TRAIN.load_detections(coco, splits)
    tree = E20_DATA.build_tree()
    params = {m: json.load(open(os.path.join(E20_DATA.OUTPUT_DIR, "E2_pipeline", f"{m}+vg_subtree", "summary.json")))["params"]
              for m in ("New", "Old")}
    results = {}
    for (det, budget), emb in itertools.product(E2_DETECTORS, E2_EMBEDDERS):
        sub_params = params["Old" if emb.startswith("CLIP") else "New"]
        sources = {s: build_source(pool, coco, splits, dets, thresholds, tree, det, emb, budget, sub_params, s) for s in SPLITS}
        detector = f"{det}@{budget}" if budget else det
        for score_type, dedup, label in tqdm(list(itertools.product(E2_SCORES, E2_DEDUPS, LABEL_TYPES)),
                                             desc=f"E2 {detector} {emb}", leave=False):
            name = f"{detector}_{E2_SHORT[emb]}_{dedup}_{score_type}_{label}label"
            config = {"source": detector, "detector": det, "budget": budget, "embedder": emb, "alpha_input": E2_SHORT[emb],
                      "dedup": dedup, "score_type": score_type, "label": label, "subtree_params": sub_params,
                      "owl_threshold": thresholds.get(budget) if budget else None}
            results[name] = evaluate_config(name, config, sources, instances, "E2_factorial")
    return results


def summary_table_e2(results):
    """
    E2_factorial の test の結果の表を保存する（θ は dev の F値と hF の2つの基準）．

    Args:
        results (dict): run_e2 の戻り値．

    Returns:
        pd.DataFrame: 表．
    """
    rows = []
    for name, r in results.items():
        for crit in CRITERIA:
            rows.append({"method": name, "detector": r["source"], "embedder": r["embedder"], "dedup": r["dedup"],
                         "score_type": r["score_type"], "label": r["label"], "criterion": crit, "theta": r[crit]["theta"],
                         **{f"dev_{k}": r[crit]["dev"][k] for k in ("f1", "sub_hF")}, **r[crit]["test"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E2_factorial", "summary_test.csv"), index=False)
    return df


def e2_name(detector, emb, dedup, score_type, label):
    """
    E2_factorial の手法名を作る．

    Args:
        detector (str): "DETR" または "OWLv2@{B}"．
        emb (str): E2_EMBEDDERS のいずれか．
        dedup (str): 除去の種類．
        score_type (str): 棄却のスコア．
        label (str): 深さ0のラベル．

    Returns:
        str: 手法名．
    """
    return f"{detector}_{E2_SHORT[emb]}_{dedup}_{score_type}_{label}label"


def significance_e2(results):
    """
    E2_factorial の主な対の差を，画像単位の対応のあるブートストラップで検定する（θ は dev の F値で選ぶ）．
    - 同じ検出器・F で，棄却のスコア joint → mass，mass_joint（ex019 の条件：除去なし，coco ラベル）
    - 最良の判定（nms_contain，mass_joint，mass ラベル）での CLIP-B/32 と Alpha-CLIP，DETR と OWLv2

    Args:
        results (dict): run_e2 の戻り値．

    Returns:
        pd.DataFrame: 検定の表（b − a）．
    """
    pairs = []
    for detector, emb in itertools.product(["DETR", "OWLv2@10"], ["CLIP-ViT-B32", "AlphaCLIP-ViT-B16_context1.5"]):
        for score in ("mass", "mass_joint"):
            pairs.append((e2_name(detector, emb, "none", "joint", "coco"), e2_name(detector, emb, "none", score, "coco")))
    best = ("nms_contain", "mass_joint", "mass")
    for detector in ("DETR", "OWLv2@10", "OWLv2@20", "OWLv2@50"):
        for emb in ("AlphaCLIP-ViT-B16_context1.5", "AlphaCLIP-ViT-B16_crop"):
            pairs.append((e2_name(detector, "CLIP-ViT-B32", *best), e2_name(detector, emb, *best)))
    for emb in E2_EMBEDDERS:
        for detector in ("OWLv2@10", "OWLv2@20", "OWLv2@50"):
            pairs.append((e2_name("DETR", emb, *best), e2_name(detector, emb, *best)))
    rows = []
    for a, b in pairs:
        ca, cb = results[a]["counts"]["f1"], results[b]["counts"]["f1"]
        f1 = E19_TRAIN.paired_bootstrap(ca["coco"], cb["coco"])
        hf = E20_TRAIN.paired_bootstrap_hf(ca["sub"], cb["sub"])
        rows.append({"a": a, "b": b, **{f"{k}_{x}": f1[k][x] for k in ("top1_acc", "f1") for x in ("diff", "ci95", "p_value")},
                     **{f"sub_hF_{x}": hf[x] for x in ("diff", "ci95", "p_value")}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "E2_factorial", "paired_bootstrap_test.csv"), index=False)
    return out


def plot_e2(df, filename="fig_e2_factorial.png"):
    """
    E2_factorial の test の F値と人物の再現率を，検出器 × F ごとに棄却のスコアで比べる（θ は dev の F値で選ぶ）．
    (a)(b) ex019 の条件（除去なし，coco ラベル），(c)(d) 最良の判定の条件（nms_contain，mass ラベル）．

    Args:
        df (pd.DataFrame): summary_table_e2 の戻り値．
        filename (str): 保存するファイル名．
    """
    sub = df[df.criterion == "f1"]
    combos = [(d, e) for d, e in itertools.product(["DETR", "OWLv2@10", "OWLv2@20", "OWLv2@50"], E2_EMBEDDERS)]
    labels = [f"{d}\n{E2_SHORT[e]}" for d, e in combos]
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    settings = [("none", "coco", "dedup=none, label=coco (ex019 setting)"), ("nms_contain", "mass", "dedup=nms_contain, label=mass")]
    width = 0.8 / len(E2_SCORES)
    for row, (dedup, label, title) in enumerate(settings):
        for col, (metric, ylabel) in enumerate([("f1", "COCO F1 (test)"), ("sub_person_recall", "Person recall (test)")]):
            ax = axes[row, col]
            for k, score in enumerate(E2_SCORES):
                vals = [sub[(sub.detector == d) & (sub.embedder == e) & (sub.dedup == dedup) & (sub.label == label)
                            & (sub.score_type == score)][metric].iloc[0] for d, e in combos]
                bars = ax.bar(np.arange(len(combos)) + (k - 1) * width, vals, width, label=score)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=6)
            ax.set_xticks(np.arange(len(combos)), labels, fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({'abcd'[2 * row + col]}) {title}", fontsize=10)
            ax.legend(fontsize=8, title="rejection score")
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)


def train_e2():
    """
    E2_factorial（order_037）を実行し，表・検定・図を保存する．
    """
    E17_MODEL.set_seed(SEED)
    os.makedirs(os.path.join(OUTPUT_DIR, "E2_factorial"), exist_ok=True)
    coco = E19_DATA.load_coco()
    splits = E19_DATA.load_splits(coco)
    nouns = {s: E20_DATA.caption_nouns(coco, splits[s], s) for s in SPLITS}
    instances = {s: E20_TRAIN.fine_instances(coco, splits[s], nouns[s]) for s in SPLITS}
    pool = E19_TRAIN.EmbedderPool()
    results = run_e2(pool, coco, splits, instances)
    df = summary_table_e2(results)
    cols = ["method", "theta", "kept_per_image", "top1_acc", "precision", "recall", "f1", "sub_hF", "sub_person_recall"]
    print(df[df.criterion == "f1"].sort_values("dev_f1", ascending=False)[cols].head(15).round(3).to_string())
    print(significance_e2(results).round(3).to_string())
    plot_e2(df)


def run_e3(pool, coco, splits, instances):
    """
    E3_detector（order_039）：提案手法の検出器 D と分類器 F を替えた全ての組を，提案の判定と従来の判定で評価する．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → ex020 の fine_instances の戻り値．

    Returns:
        dict: 手法名 → evaluate_config の戻り値（"detector"，"embedder"，"judge" を含む）．
    """
    dets, thresholds = E19_TRAIN.load_detections(coco, splits)
    tree = E20_DATA.build_tree()
    params = {m: json.load(open(os.path.join(E20_DATA.OUTPUT_DIR, "E2_pipeline", f"{m}+vg_subtree", "summary.json")))["params"]
              for m in ("New", "Old")}
    results = {}
    for (det, budget), emb in itertools.product(E3_DETECTORS, E2_EMBEDDERS):
        sub_params = params["Old" if emb.startswith("CLIP") else "New"]
        sources = {s: build_source(pool, coco, splits, dets, thresholds, tree, det, emb, budget, sub_params, s) for s in SPLITS}
        detector = f"{det}@{budget}" if budget else det
        for judge, (dedup, score_type, label) in E3_JUDGES.items():
            name = f"{detector}_{E2_SHORT[emb]}_{judge}"
            config = {"source": detector, "detector": det, "budget": budget, "embedder": emb, "judge": judge,
                      "dedup": dedup, "score_type": score_type, "label": label, "subtree_params": sub_params,
                      "owl_threshold": thresholds.get(budget) if budget else None}
            results[name] = evaluate_config(name, config, sources, instances, "E3_detector")
    return results


def significance_e3(results):
    """
    E3_detector の主な対の差を，画像単位の対応のあるブートストラップで検定する（提案の判定，θ は dev の F値で選ぶ）．
    - 同じ F で，OWLv2@50 と YOLO11x，DETR，OWLv2@20
    - 同じ D で，CLIP-B/32 と Alpha-CLIP（crop，context1.5）

    Args:
        results (dict): run_e3 の戻り値．

    Returns:
        pd.DataFrame: 検定の表（b − a）．
    """
    pairs = [(f"{d}_{E2_SHORT[e]}_proposed", f"OWLv2@50_{E2_SHORT[e]}_proposed")
             for e in E2_EMBEDDERS for d in ("YOLO11x", "DETR", "OWLv2@20")]
    pairs += [(f"{d}_CLIP-B32_proposed", f"{d}_{E2_SHORT[e]}_proposed")
              for d in ("OWLv2@20", "OWLv2@50", "YOLO11x", "DETR") for e in E2_EMBEDDERS[1:]]
    rows = []
    for a, b in pairs:
        ca, cb = results[a]["counts"]["f1"], results[b]["counts"]["f1"]
        f1 = E19_TRAIN.paired_bootstrap(ca["coco"], cb["coco"])
        hf = E20_TRAIN.paired_bootstrap_hf(ca["sub"], cb["sub"])
        rows.append({"a": a, "b": b, **{f"{k}_{x}": f1[k][x] for k in ("top1_acc", "f1") for x in ("diff", "ci95", "p_value")},
                     **{f"sub_hF_{x}": hf[x] for x in ("diff", "ci95", "p_value")}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "E3_detector", "paired_bootstrap_test.csv"), index=False)
    return out


def kept_boxes(source, config, theta):
    """
    1つの条件で残したボックスを画像ごとに返す．

    Args:
        source (list[dict]): build_source の戻り値（test）．
        config (dict): "score_type"，"dedup"，"label" を含む条件．
        theta (float): 棄却の閾値．

    Returns:
        dict[int, np.ndarray]: 画像 ID → 残したボックス（K×4）．
    """
    src = with_label(source, config["label"])
    scores = config_scores(src, config["score_type"], config["dedup"])
    return {r["image_id"]: r["boxes"][s >= theta] for r, s in zip(src, scores)}


def lvis_counts(boxes_by_image, coco_gt, lvis, image_ids):
    """
    残したボックスを COCO の正解と LVIS の注釈で照合する（クラス非依存，IoU 0.5 以上）．

    Args:
        boxes_by_image (dict[int, np.ndarray]): 画像 ID → ボックス（K×4）．
        coco_gt (dict[int, np.ndarray]): 画像 ID → COCO の正解のボックス（G×4）．
        lvis (dict): ex018 の load_lvis の戻り値．
        image_ids (list[int]): LVIS の注釈がある画像 ID．

    Returns:
        dict: "boxes_per_image"，"noncoco_recall"（COCO 外の物体のうち囲んだ割合），"box_no_coco_gt"（COCO の正解に当たらない
              ボックスの割合），"no_coco_gt_on_noncoco"（そのうち COCO 外の物体を囲む割合）．
    """
    E18_DATA = load_module("ex018")[0]
    n_box = n_nococo = n_nococo_on_lvis = n_noncoco = n_noncoco_found = 0
    for i in image_ids:
        boxes = boxes_by_image.get(i, np.zeros((0, 4), np.float32))
        lv_boxes, kinds, _ = E18_DATA.gt_arrays(lvis, i)
        noncoco = lv_boxes[np.array([k == "noncoco" for k in kinds], dtype=bool)]
        hit_coco = (box_iou(boxes, coco_gt[i]) >= 0.5).any(axis=1) if len(boxes) and len(coco_gt[i]) else np.zeros(len(boxes), bool)
        hit_non = (box_iou(boxes, noncoco) >= 0.5).any(axis=1) if len(boxes) and len(noncoco) else np.zeros(len(boxes), bool)
        found = (box_iou(noncoco, boxes) >= 0.5).any(axis=1) if len(boxes) and len(noncoco) else np.zeros(len(noncoco), bool)
        n_box += len(boxes)
        n_nococo += int((~hit_coco).sum())
        n_nococo_on_lvis += int((~hit_coco & hit_non).sum())
        n_noncoco += len(noncoco)
        n_noncoco_found += int(found.sum())
    return {"boxes_per_image": n_box / len(image_ids), "noncoco_recall": safe_div(n_noncoco_found, n_noncoco),
            "box_no_coco_gt": safe_div(n_nococo, n_box), "no_coco_gt_on_noncoco": safe_div(n_nococo_on_lvis, n_nococo),
            "n_noncoco": n_noncoco}


def train_e3_lvis():
    """
    E3_detector の test の残したボックス（提案の判定と従来の判定）と，棄却前の全てのボックスを LVIS の注釈と照合する．
    """
    E18_DATA = load_module("ex018")[0]
    coco = E19_DATA.load_coco()
    splits = E19_DATA.load_splits(coco)
    lvis = E18_DATA.load_lvis()
    image_ids = [i for i in splits["test"] if i in set(E18_DATA.test_image_ids(lvis))]
    coco_gt = {i: E19_DATA.gt_instances(coco, i)[0] for i in image_ids}
    dets, thresholds = E19_TRAIN.load_detections(coco, splits)
    tree = E20_DATA.build_tree()
    rows = []
    for (det, budget), emb in itertools.product(E3_DETECTORS, E2_EMBEDDERS):
        detector = f"{det}@{budget}" if budget else det
        source = build_source(None, coco, splits, dets, thresholds, tree, det, emb, budget, None, "test")
        if emb == E2_EMBEDDERS[0]:
            rows.append({"method": f"{detector}_all_boxes", "detector": detector, "embedder": None, "judge": "none",
                         **lvis_counts({r["image_id"]: r["boxes"] for r in source}, coco_gt, lvis, image_ids)})
        for judge in E3_JUDGES:
            name = f"{detector}_{E2_SHORT[emb]}_{judge}"
            summary = json.load(open(os.path.join(OUTPUT_DIR, "E3_detector", name, "summary.json")))
            kept = kept_boxes(source, summary, summary["f1"]["theta"])
            rows.append({"method": name, "detector": detector, "embedder": emb, "judge": judge,
                         **lvis_counts(kept, coco_gt, lvis, image_ids)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E3_detector", "lvis_test.csv"), index=False)
    print(f"LVIS の注釈がある test の画像：{len(image_ids)}枚")
    print(df.round(3).to_string())


def train_e3():
    """
    E3_detector（order_039）を実行し，表と検定を保存する．
    """
    E17_MODEL.set_seed(SEED)
    os.makedirs(os.path.join(OUTPUT_DIR, "E3_detector"), exist_ok=True)
    coco = E19_DATA.load_coco()
    splits = E19_DATA.load_splits(coco)
    nouns = {s: E20_DATA.caption_nouns(coco, splits[s], s) for s in SPLITS}
    instances = {s: E20_TRAIN.fine_instances(coco, splits[s], nouns[s]) for s in SPLITS}
    pool = E19_TRAIN.EmbedderPool()
    results = run_e3(pool, coco, splits, instances)
    rows = [{"method": n, "detector": r["source"], "embedder": r["embedder"], "judge": r["judge"], "criterion": "f1",
             "theta": r["f1"]["theta"], "dev_f1": r["f1"]["dev"]["f1"], **r["f1"]["test"]} for n, r in results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E3_detector", "summary_test.csv"), index=False)
    cols = ["method", "kept_per_image", "top1_acc", "precision", "recall", "f1", "sub_hF", "sub_person_recall"]
    print(df[cols].round(3).to_string())
    print(significance_e3(results).round(3).to_string())


def summary_table(results):
    """
    test の結果の表（全ての条件 × θ の基準）を保存する．

    Args:
        results (dict): run_e1 の戻り値．

    Returns:
        pd.DataFrame: 表．
    """
    rows = []
    for name, r in results.items():
        for crit in CRITERIA:
            rows.append({"method": name, "source": r["source"], "alpha_input": r.get("alpha_input"), "dedup": r["dedup"],
                         "label": r.get("label", "coco"),
                         "score_type": r["score_type"], "criterion": crit, "theta": r[crit]["theta"],
                         **{f"dev_{k}": r[crit]["dev"][k] for k in ("f1", "sub_hF")}, **r[crit]["test"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E1_rejection", "summary_test.csv"), index=False)
    return df


def best_new(df, crit):
    """
    dev の基準の値が最大の新しい提案手法の条件を選ぶ（test ではなく dev で選ぶ）．

    Args:
        df (pd.DataFrame): summary_table の戻り値．
        crit (str): "f1" または "hf"．

    Returns:
        str: 手法名．
    """
    sub = df[df.method.str.startswith("New") & (df.criterion == crit)]
    return sub.loc[sub[f"dev_{'f1' if crit == 'f1' else 'sub_hF'}"].idxmax(), "method"]


def significance(results, df):
    """
    dev で選んだ最良の新しい提案手法と，現在の新しい提案手法・古い提案手法・YOLO11x 単体の差を検定する．

    Args:
        results (dict): run_e1 の戻り値．
        df (pd.DataFrame): summary_table の戻り値．

    Returns:
        pd.DataFrame: 検定の表．
    """
    current = "New_OWLv2@20_context1.5_none_joint"
    rows = []
    for crit in CRITERIA:
        new = best_new(df, crit)
        for ref in (current, "Old_DETR_CLIP-B32_none_vlm", "Old_DETR_CLIP-B32_none_vlm_masslabel", "YOLO11x_closed"):
            a, b = results[ref]["counts"][crit], results[new]["counts"][crit]
            f1 = E19_TRAIN.paired_bootstrap(a["coco"], b["coco"])
            hf = E20_TRAIN.paired_bootstrap_hf(a["sub"], b["sub"])
            rows.append({"criterion": crit, "new": new, "baseline": ref,
                         **{f"{k}_{x}": f1[k][x] for k in ("top1_acc", "f1") for x in ("diff", "ci95", "p_value")},
                         **{f"sub_hF_{x}": hf[x] for x in ("diff", "ci95", "p_value")}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "E1_rejection", "paired_bootstrap_test.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def plot_factors(df, filename="fig_rejection_factors.png"):
    """
    棄却のスコア × 重複の除去ごとの test の F値と hF（θ は dev の F値で選ぶ．ボックス数・入力・ラベルは dev の F値で最良のもの）と，
    COCO の適合率と人物の再現率を描く．

    Args:
        df (pd.DataFrame): summary_table の戻り値．
        filename (str): 保存するファイル名．
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    new = df[df.method.str.startswith("New")]
    refs = df[~df.method.str.startswith("New")]
    for ax, (crit, col, dev_col, title) in zip(axes[:2], [("f1", "f1", "dev_f1", "(a) COCO F1 (θ by dev F1)"),
                                                          ("f1", "sub_hF", "dev_f1", "(b) fine hF, vg_subtree (θ by dev F1)")]):
        width = 0.8 / len(DEDUP_TYPES)
        for k, dedup in enumerate(DEDUP_TYPES):
            sub = new[(new.criterion == crit) & (new.dedup == dedup)]
            best = sub.loc[sub.groupby("score_type")[dev_col].idxmax()].set_index("score_type").reindex(SCORE_TYPES)
            ax.bar(np.arange(len(SCORE_TYPES)) + (k - 1) * width, best[col], width, label=f"New, dedup={dedup}")
        for ref, ls in (("Old_DETR_CLIP-B32_none_vlm", "--"), ("YOLO11x_closed", ":")):
            v = refs[(refs.method == ref) & (refs.criterion == crit)][col].iloc[0]
            ax.axhline(v, ls=ls, color="k", label=ref.split("_")[0] + f" ({v:.3f})")
        ax.set_xticks(np.arange(len(SCORE_TYPES)), SCORE_TYPES)
        ax.set_ylabel(f"{col} (test)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    for crit, marker in (("f1", "o"), ("hf", "s")):
        sub = new[new.criterion == crit]
        ax.scatter(sub.precision, sub.sub_person_recall, marker=marker, s=14, alpha=0.6, label=f"New configs (θ by {crit})")
        for ref, color in (("Old_DETR_CLIP-B32_none_vlm", "red"), ("YOLO11x_closed", "gray")):
            r = refs[(refs.method == ref) & (refs.criterion == crit)].iloc[0]
            ax.scatter([r.precision], [r.sub_person_recall], marker=marker, s=80, color=color, label=f"{ref.split('_')[0]} (θ by {crit})")
    ax.set_xlabel("COCO precision (test)")
    ax.set_ylabel("Person recall, class-agnostic (test)")
    ax.set_title("(c) Precision vs person recall")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)


def draw_examples(coco, splits, results, sources_all, df, filename="fig_rejection_examples.png"):
    """
    現在の新しい提案手法，dev で選んだ最良の新しい提案手法（F値と hF の基準），古い提案手法，YOLO11x 単体の出力（vg_subtree）を並べる．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        results (dict): run_e1 の戻り値．
        sources_all (dict): run_e1 のソース．
        df (pd.DataFrame): summary_table の戻り値．
        filename (str): 保存するファイル名．
    """
    columns = [("Current New", "New_OWLv2@20_context1.5_none_joint", "f1"), ("Best New (θ by F1)", best_new(df, "f1"), "f1"),
               ("Best New (θ by hF)", best_new(df, "hf"), "hf"), ("Old", "Old_DETR_CLIP-B32_none_vlm", "hf"),
               ("YOLO11x", "YOLO11x_closed", "f1")]
    ids = sorted(splits["test"])[::len(splits["test"]) // N_QUALITATIVE][:N_QUALITATIVE]
    fig, axes = plt.subplots(len(ids), len(columns), figsize=(4.5 * len(columns), 3.8 * len(ids)))
    for c, (title, name, crit) in enumerate(columns):
        r = results[name]
        key = "YOLO11x" if name.startswith("YOLO") else ("Old" if name.startswith("Old") else (r["source"], r["alpha_input"]))
        src = {x["image_id"]: x for x in sources_all[key]["test"]}
        for row, i in enumerate(ids):
            x = with_label([src[i]], r.get("label", "coco"))[0]
            s = config_scores([x], r["score_type"], r["dedup"])[0]
            ax = axes[row, c]
            ax.imshow(E19_DATA.load_image(coco, i))
            ax.axis("off")
            for k in np.where(s >= r[crit]["theta"])[0]:
                x0, y0, x1, y1 = x["boxes"][k]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", lw=1.2))
                ax.text(x0, y0, x["out_sub"][k][1], fontsize=6, bbox={"facecolor": "cyan", "alpha": 0.7, "pad": 1})
            if row == 0:
                ax.set_title(f"{title}\n{name}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=100)
    plt.close(fig)


def train():
    """
    ex021 の全ての実験（E1_rejection，要約，検定，図）を実行する．
    """
    E17_MODEL.set_seed(SEED)
    os.makedirs(os.path.join(OUTPUT_DIR, "E1_rejection"), exist_ok=True)
    coco = E19_DATA.load_coco()
    splits = E19_DATA.load_splits(coco)
    nouns = {s: E20_DATA.caption_nouns(coco, splits[s], s) for s in SPLITS}
    instances = {s: E20_TRAIN.fine_instances(coco, splits[s], nouns[s]) for s in SPLITS}
    pool = E19_TRAIN.EmbedderPool()
    results, sources_all = run_e1(pool, coco, splits, instances)
    df = summary_table(results)
    cols = ["method", "criterion", "theta", "kept_per_image", "top1_acc", "precision", "recall", "f1", "sub_hF", "sub_person_recall"]
    print(df.sort_values("dev_f1", ascending=False)[cols].head(15).round(3).to_string())
    print(df[~df.method.str.startswith("New") | (df.method == "New_OWLv2@20_context1.5_none_joint")][cols].round(3).to_string())
    print(significance(results, df).round(3).to_string())
    plot_factors(df)
    draw_examples(coco, splits, results, sources_all, df)


if __name__ == "__main__":
    {"e2": train_e2, "e3": train_e3, "e3lvis": train_e3_lvis}.get(sys.argv[1] if len(sys.argv) > 1 else "", train)()
