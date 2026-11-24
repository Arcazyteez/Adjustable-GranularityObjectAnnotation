"""
ex019：新しい提案手法（OWLv2 + Alpha-CLIP + WordNet）と古い提案手法（DETR + CLIP + WordNet）の比較（order_029）．

学習は行わず，学習済みのモデルの推論と評価のみを行う．データは MSCOCO val2017 のみ（ex017 と同じ dev 500枚，test 500枚）．
結果は outputs/ex019_coco_owl_alphaclip_wordnet/{実験}/{手法}/{条件}/{seed}/ に保存し，metrics.json がある条件はスキップする．

実験：
- E1_naming：正解のボックスに対する深さ0（COCO 80クラス）の命名の Top-1/Top-5 正解率．検出の影響を除いて F だけを比べる．
  F ∈ {CLIP-ViT-B32（古い提案手法．QuickGELU），CLIP-ViT-B32-GELU（ex017 の設定），CLIP-ViT-B16，
  Alpha-CLIP-ViT-B16 × 入力 {crop, context1.5, context2, full}}．Alpha-CLIP の入力は dev の Top-1 で選ぶ．
- E2_pipeline：検出から命名までの評価（ex017 の E3 と同じ指標：対応したボックスの Top-1/Top-5，クラスを含めた適合率・再現率・F値，IoU）．
  D ∈ {DETR（閾値0.5），OWLv2 objectness}，F ∈ {CLIP-ViT-B32，CLIP-ViT-B16，Alpha-CLIP（E1 で選んだ入力）}．
  棄却のスコアは VLM の確信度（softmax の最大値）と，検出スコアとの積（joint）．
  OWLv2 の1枚あたりの平均ボックス数（dev のスコアで閾値を決める）と θ は dev の F値で選び，test で報告する．
  古い提案手法と新しい提案手法の差は，画像単位の対応のあるブートストラップで検定する．
- E3_speed：1枚あたりの検出・特徴計算・WordNet の細分化の時間と，ボックス数．
- E4_depth：test で深さ0が正しいインスタンスについて，WordNet の深さごとの確信度．比較画像（古い提案手法と新しい提案手法）．

実験条件の履歴：
- order_029：D ∈ {DETR, OWLv2}．OWLv2 の予算は {5, 10, 20, 30, 50} 個/枚から dev の F値で選ぶ（10個/枚が選ばれた）．
- order_030：D に YOLO11x（conf 0.25）を追加し，YOLO11x 単体（Close Vocabulary モデル）とも比較する．
  OWLv2 の予算を20個/枚と50個/枚に固定した条件（手法名の末尾 "@20"，"@50"）も評価する．
- order_032：ハイブリッド手法（YOLO11x の検出に F と WordNet を組み合わせる YOLO11x + F）を比較から外した．
  VLM を用いる構成の D は DETR と OWLv2 のみである．YOLO11x は単体（YOLO11x-closed）と人物の再現率の比較だけに用いる．
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data import (ALPHA_INPUTS, DATASET_DIR, OUTPUT_DIR, coco80_vocab, gt_instances, load_coco, load_ex017,
                  load_image, load_splits, make_alpha_input, make_crop)
from model import AlphaCLIPEmbedder, OWLv2Objectness, clip_embedder, granular_annotation

E17_DATA, E17_MODEL, E17_TRAIN = load_ex017()
set_seed = E17_MODEL.set_seed
save_json, result_dir_e17 = E17_TRAIN.save_json, E17_TRAIN.result_dir
metrics_func, metrics_with_ci, safe_div = E17_TRAIN.metrics_func, E17_TRAIN.metrics_with_ci, E17_TRAIN.safe_div

SEED = 0
SPLITS = ["dev", "test"]
MODE = "softmax"                                    # 論文の確信度（候補語彙内のソフトマックス）
OWL_BUDGETS = [5, 10, 20, 30, 50]                   # OWLv2 の1枚あたりの平均ボックス数の候補（dev のスコアで閾値を決める）
SCORE_TYPES = ["vlm", "joint"]                      # 棄却のスコア：VLM の確信度，検出スコアとの積
N_BOOTSTRAP = 1000
N_SPEED_IMAGES = 100
N_QUALITATIVE = 6
OLD_METHOD = ("DETR", "CLIP-ViT-B32")
DETECTORS = ["DETR", "OWLv2"]                       # VLM を用いる構成の D（order_030 で加えた YOLO11x は order_032 で外した）
FIXED_OWL_BUDGETS = [20, 50]                        # order_030：予算を固定した OWLv2 の条件
YOLO_CLOSED = "YOLO11x-closed"                      # YOLO11x 自身のクラスと検出スコア（Close Vocabulary モデル）
PERSON_BUDGETS = [10, 20, 50]
EMBEDDER_SPECS = {                                  # 名前 → (読み込み関数のキー, 入力)
    "CLIP-ViT-B32": ("clip", "ViT-B-32-quickgelu"),
    "CLIP-ViT-B32-GELU": ("clip", "ViT-B-32"),
    "CLIP-ViT-B16": ("clip", "ViT-B-16-quickgelu"),
    **{f"AlphaCLIP-ViT-B16_{m}": ("alpha", m) for m in ALPHA_INPUTS},
}


def result_dir(experiment, method, condition, seed=SEED):
    """
    結果の保存先 outputs/ex019_coco_owl_alphaclip_wordnet/{実験}/{手法}/{条件}/{seed}/ を返す（作成する）．

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
# モデルと特徴のキャッシュ
# ---------------------------------------------------------------------------
class EmbedderPool:
    """埋め込みモデルを必要になった時に1つずつ読み込む（VRAM 12GB に収めるため，同時に保持するのは1つ）．"""

    def __init__(self):
        """読み込み済みのモデルを持たない状態で初期化する．"""
        self.name, self.model = None, None

    def get(self, name):
        """
        Args:
            name (str): EMBEDDER_SPECS のキー（Alpha-CLIP は入力によらず同じモデル）．

        Returns:
            OpenCLIPEmbedder | AlphaCLIPEmbedder: 埋め込みモデル．
        """
        key = "AlphaCLIP" if name.startswith("AlphaCLIP") else name
        if self.name != key:
            self.model = None
            torch.cuda.empty_cache()
            set_seed(SEED)
            kind, arg = EMBEDDER_SPECS[name]
            self.model = AlphaCLIPEmbedder() if kind == "alpha" else clip_embedder(name, arg)
            self.name = key
        return self.model


def region_features(embedder, name, image, boxes):
    """
    ボックスの領域の特徴を求める（CLIP は切り抜き画像，Alpha-CLIP は窓とαマスク）．

    Args:
        embedder: EmbedderPool.get の戻り値．
        name (str): EMBEDDER_SPECS のキー．
        image (PIL.Image.Image): 画像．
        boxes (np.ndarray): xyxy形式のボックス（K×4）．

    Returns:
        torch.Tensor: L2正規化した特徴（K×D）．
    """
    if len(boxes) == 0:
        return torch.zeros(0, 512)
    kind, arg = EMBEDDER_SPECS[name]
    if kind == "clip":
        return embedder.image_features([make_crop(image, b) for b in boxes])
    inputs = [make_alpha_input(image, b, arg) for b in boxes]
    return embedder.image_features([x[0] for x in inputs], [x[1] for x in inputs])


def timed(fn):
    """
    関数を実行し，GPUの同期を含めた所要時間を返す．

    Args:
        fn (callable): 引数なしの関数．

    Returns:
        tuple[object, float]: 戻り値と秒数．
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def cached_features(pool, coco, name, boxes_by_image, tag):
    """
    ボックスごとの特徴をキャッシュから読み込み，無ければ計算して保存する．

    Args:
        pool (EmbedderPool): 埋め込みモデル．
        coco (dict): load_coco の戻り値．
        name (str): EMBEDDER_SPECS のキー．
        boxes_by_image (dict[int, np.ndarray]): 画像ID → ボックス（K×4）．
        tag (str): キャッシュのファイル名の識別子（例："gt_dev"，"OWLv2_test"）．

    Returns:
        dict[int, dict]: 画像ID → {"feats"（K×D），"time"（秒）}．
    """
    path = os.path.join(DATASET_DIR, f"feat_{tag}_{name}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    embedder = pool.get(name)
    first = next(iter(boxes_by_image))
    region_features(embedder, name, load_image(coco, first), boxes_by_image[first])   # ウォームアップ
    out = {}
    for i, boxes in tqdm(boxes_by_image.items(), desc=f"feat {tag} {name}", leave=False):
        image = load_image(coco, i)
        feats, sec = timed(lambda: region_features(embedder, name, image, boxes))
        out[i] = {"feats": feats, "time": sec}
    os.makedirs(DATASET_DIR, exist_ok=True)
    torch.save(out, path)
    return out


def owl_detections(coco, image_ids, tag):
    """
    OWLv2 objectness の検出結果をキャッシュから読み込み，無ければ検出して保存する．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．
        tag (str): 分割の名前．

    Returns:
        dict[int, dict]: 画像ID → {"boxes"（K×4），"scores"（K），"labels"，"time"}．スコアの降順．
    """
    path = os.path.join(DATASET_DIR, f"detections_OWLv2_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    set_seed(SEED)
    detector = OWLv2Objectness()
    timed(lambda: detector.detect(load_image(coco, image_ids[0])))
    out = {}
    for i in tqdm(image_ids, desc=f"OWLv2 {tag}", leave=False):
        image = load_image(coco, i)
        det, sec = timed(lambda: detector.detect(image))
        out[i] = det | {"time": sec}
    os.makedirs(DATASET_DIR, exist_ok=True)
    torch.save(out, path)
    del detector
    torch.cuda.empty_cache()
    return out


def budget_thresholds(dev_dets):
    """
    dev の OWLv2 のスコアから，1枚あたりの平均ボックス数が各予算になる閾値を求める（test にも同じ閾値を用いる）．

    Args:
        dev_dets (dict[int, dict]): dev の owl_detections の戻り値．

    Returns:
        dict[int, float]: 予算 → 閾値．
    """
    scores = np.sort(np.concatenate([d["scores"] for d in dev_dets.values()]))[::-1]
    return {b: float(scores[min(b * len(dev_dets), len(scores)) - 1]) for b in OWL_BUDGETS}


def load_detections(coco, splits):
    """
    DETR（ex017 のキャッシュ）と OWLv2 の検出結果を読み込み，OWLv2 は最大の予算の閾値以上のボックスに絞る．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．

    Returns:
        tuple[dict, dict]: (検出器, 分割) → 画像ID → 検出結果（DETR，YOLO11x，OWLv2），予算 → 閾値．
    """
    dets = {(d, s): E17_TRAIN.cached_detections(coco, d, splits[s], s) for d, s in itertools.product(["DETR", "YOLO11x"], SPLITS)}
    owl = {s: owl_detections(coco, splits[s], s) for s in SPLITS}
    thresholds = budget_thresholds(owl["dev"])
    floor = min(thresholds.values())
    for s in SPLITS:
        dets[("OWLv2", s)] = {i: d | {"boxes": d["boxes"][d["scores"] >= floor], "scores": d["scores"][d["scores"] >= floor]}
                              for i, d in owl[s].items()}
    return dets, thresholds


# ---------------------------------------------------------------------------
# E1：正解のボックスに対する命名
# ---------------------------------------------------------------------------
def run_e1(pool, coco, splits):
    """
    E1：正解のボックスの深さ0の命名の Top-1/Top-5 正解率と，1ボックスあたりの特徴計算の時間．Alpha-CLIP の入力を dev で選ぶ．

    Args:
        pool (EmbedderPool): 埋め込みモデル．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．

    Returns:
        str: 選んだ Alpha-CLIP の入力を含む F の名前（例："AlphaCLIP-ViT-B16_context1.5"）．
    """
    names = coco["class_names"]
    rows = []
    for name, split in itertools.product(EMBEDDER_SPECS, SPLITS):
        gt = {i: gt_instances(coco, i) for i in splits[split]}
        d = result_dir("E1_naming", name, split)
        path = os.path.join(d, "metrics.json")
        if os.path.exists(path):
            rows.append(json.load(open(path)))
            continue
        feats = cached_features(pool, coco, name, {i: g[0] for i, g in gt.items()}, f"gt_{split}")
        embedder = pool.get(name)
        correct1, correct5 = [], []
        for i, (_, labels) in gt.items():
            if not labels:
                continue
            order = np.argsort(-embedder.scores(feats[i]["feats"], names, MODE), axis=1)[:, :5]
            correct1 += [names[o[0]] == l for o, l in zip(order, labels)]
            correct5 += [l in [names[k] for k in o] for o, l in zip(order, labels)]
        n_boxes = sum(len(g[1]) for g in gt.values())
        m = {"embedder": name, "split": split, "n_gt": len(correct1), "top1_acc": float(np.mean(correct1)),
             "top5_acc": float(np.mean(correct5)),
             "sec_per_box": float(sum(v["time"] for v in feats.values()) / max(n_boxes, 1))}
        save_json(m, path)
        save_json({"embedder": name, "input": EMBEDDER_SPECS[name][1], "mode": MODE, "vocabulary": "coco80",
                   "boxes": "COCO ground truth (iscrowd=0)", "split": split, "seed": SEED}, os.path.join(d, "config.json"))
        np.save(os.path.join(d, "correct_top1.npy"), np.array(correct1))
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E1_naming", "summary.csv"), index=False)
    alpha_dev = df[(df.split == "dev") & df.embedder.str.startswith("AlphaCLIP")]
    best = alpha_dev.loc[alpha_dev.top1_acc.idxmax(), "embedder"]
    print("E1", df.pivot(index="embedder", columns="split", values="top1_acc").round(3).to_dict(), "→ 選んだ入力:", best)
    return best


def mcnemar_e1(name_a, name_b, split="test"):
    """
    E1 の2つの F の Top-1 の正誤について，正確な McNemar 検定（二項検定）を行う．

    Args:
        name_a (str): F の名前．
        name_b (str): F の名前．
        split (str): 分割．

    Returns:
        dict: "a_only"，"b_only"（片方のみ正解の件数），"p_value"．
    """
    from scipy.stats import binomtest
    a = np.load(os.path.join(result_dir("E1_naming", name_a, split), "correct_top1.npy"))
    b = np.load(os.path.join(result_dir("E1_naming", name_b, split), "correct_top1.npy"))
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a": name_a, "b": name_b, "split": split, "a_only": a_only, "b_only": b_only, "p_value": float(p)}


# ---------------------------------------------------------------------------
# E2：検出から命名まで
# ---------------------------------------------------------------------------
def make_records(coco, image_ids, dets, feats, embedder, n_keep_fn, score_type):
    """
    評価用の画像ごとの予測と正解をまとめる（ex017 の build_records と同じ形式）．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．
        dets (dict[int, dict]): 検出結果（スコアの降順）．
        feats (dict[int, dict]): cached_features の戻り値（dets と同じボックスの順序）．
        embedder: 埋め込みモデル（scores を持つ）．
        n_keep_fn (callable): 検出結果 → 用いる上位のボックス数．
        score_type (str): "vlm"（VLM の確信度）または "joint"（検出スコアとの積）．

    Returns:
        list[dict]: "image_id"，"gt_boxes"，"gt_labels"，"boxes"，"top"，"scores"，"det_scores"．
    """
    names = coco["class_names"]
    records = []
    for i in image_ids:
        n = n_keep_fn(dets[i])
        gt_boxes, gt_labels = gt_instances(coco, i)
        f = feats[i]["feats"][:n]
        s = embedder.scores(f, names, MODE) if n else np.zeros((0, len(names)), np.float32)
        order = np.argsort(-s, axis=1)[:, :5]
        conf = s.max(axis=1) if n else np.zeros(0, np.float32)
        det_scores = np.asarray(dets[i]["scores"][:n], np.float32)
        records.append({"image_id": i, "gt_boxes": gt_boxes, "gt_labels": gt_labels,
                        "boxes": np.asarray(dets[i]["boxes"][:n], np.float32).reshape(-1, 4),
                        "top": [[names[k] for k in row] for row in order],
                        "scores": (conf * det_scores if score_type == "joint" else conf).astype(np.float32),
                        "det_scores": det_scores})
    return records


def select_on_dev(records_by_budget, out_dir):
    """
    dev の F値が最大となる (予算, θ) を選ぶ．θ の候補は −∞ と dev のスコアの分位点（ex017 の select_theta）．

    Args:
        records_by_budget (dict[int | None, list[dict]]): 予算（DETR は None）→ dev の records．
        out_dir (str): 予算ごとの theta_curve.csv の保存先．

    Returns:
        dict: "budget_all"（θ なしで F値が最大の予算），"budget"，"theta"（θ ありで F値が最大の組）．
    """
    rows = []
    for budget, records in records_by_budget.items():
        sub = os.path.join(out_dir, f"budget{budget}" if budget else "all")
        theta = E17_TRAIN.select_theta(records, sub)
        rows.append({"budget": budget, "theta": theta, "f1_theta": metrics_func(records, theta)[0]["f1"],
                     "f1_all": metrics_func(records)[0]["f1"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "selection.csv"), index=False)
    best = df.loc[df.f1_theta.idxmax()]
    budget_all = df.loc[df.f1_all.idxmax(), "budget"]
    to_int = lambda b: None if b is None or pd.isna(b) else int(b)   # noqa: E731
    return {"budget_all": to_int(budget_all), "budget": to_int(best["budget"]), "theta": float(best["theta"])}


def paired_bootstrap(counts_a, counts_b, n=N_BOOTSTRAP, seed=SEED):
    """
    画像単位の対応のあるブートストラップで，2手法の Top-1 正解率と F値の差（b − a）の95%信頼区間と両側 p 値を求める．

    Args:
        counts_a (np.ndarray): 手法 a の metrics_func の画像ごとの計数（N×6）．
        counts_b (np.ndarray): 手法 b の計数（N×6，同じ画像の順序）．
        n (int): 再標本化の回数．
        seed (int): 乱数シード．

    Returns:
        dict: 指標名 → {"diff"，"ci95"，"p_value"}．
    """
    def stats(s):
        """合計した計数から Top-1 正解率と F値を求める．"""
        p, r = safe_div(s[3], s[4]), safe_div(s[3], s[5])
        return {"top1_acc": safe_div(s[1], s[0]), "f1": safe_div(2 * p * r, p + r) if s[3] else 0.0}
    rng = np.random.default_rng(seed)
    base = {k: stats(counts_b.sum(0))[k] - stats(counts_a.sum(0))[k] for k in ("top1_acc", "f1")}
    diffs = {k: [] for k in base}
    for _ in range(n):
        idx = rng.integers(0, len(counts_a), len(counts_a))
        sa, sb = stats(counts_a[idx].sum(0)), stats(counts_b[idx].sum(0))
        for k in diffs:
            diffs[k].append(sb[k] - sa[k])
    out = {}
    for k, v in diffs.items():
        v = np.array(v)
        p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        out[k] = {"diff": float(base[k]), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  "p_value": float(max(p, 1 / n))}
    return out


def evaluate_method(pool, coco, splits, dets, thresholds, det_name, emb_name, score_type, fixed_budget=None):
    """
    1つの (D, F, 棄却のスコア) について，dev で予算と θ を選び，dev と test の結果を保存する．

    Args:
        pool (EmbedderPool): 埋め込みモデル．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): load_detections の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．
        det_name (str): "DETR"，"YOLO11x"，"OWLv2" のいずれか．
        emb_name (str): EMBEDDER_SPECS のキー．
        score_type (str): "vlm" または "joint"．
        fixed_budget (int | None): OWLv2 の予算を固定する場合の個数（None なら dev で選ぶ）．

    Returns:
        dict: 選んだ設定と test の指標（"all"：θ なし，"theta"：θ あり）．
    """
    method = f"{det_name}+{emb_name}+{score_type}" + (f"@{fixed_budget}" if fixed_budget else "")
    out_dir = os.path.join(OUTPUT_DIR, "E2_pipeline", method)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        return json.load(open(summary_path))
    feats = {s: cached_features(pool, coco, emb_name, {i: d["boxes"] for i, d in dets[(det_name, s)].items()},
                                f"{det_name}_{s}") for s in SPLITS}
    embedder = pool.get(emb_name)

    def records_for(split, budget):
        """分割と予算の records を作る（DETR，YOLO11x は全てのボックス）．"""
        keep = (lambda d: len(d["boxes"])) if budget is None else (lambda d: int((d["scores"] >= thresholds[budget]).sum()))
        return make_records(coco, splits[split], dets[(det_name, split)], feats[split], embedder, keep, score_type)

    budgets = [None] if det_name != "OWLv2" else ([fixed_budget] if fixed_budget else OWL_BUDGETS)
    chosen = select_on_dev({b: records_for("dev", b) for b in budgets}, out_dir)
    summary = {"method": method, "detector": det_name, "embedder": emb_name, "score_type": score_type,
               "budget_selection": "fixed" if fixed_budget else ("dev F1" if det_name == "OWLv2" else None), **chosen}
    for split, (cond, budget, theta) in itertools.product(SPLITS, [("all", chosen["budget_all"], -np.inf),
                                                                   ("theta", chosen["budget"], chosen["theta"])]):
        records = records_for(split, budget)
        m = metrics_with_ci(records, theta)
        m["boxes_per_image"] = float(np.mean([len(r["boxes"]) for r in records]))
        m["kept_per_image"] = float(np.mean([(r["scores"] >= theta).sum() for r in records]))
        d = result_dir("E2_pipeline", method, f"{cond}_{split}")
        save_json(m, os.path.join(d, "metrics.json"))
        save_json({"detector": det_name, "embedder": emb_name, "score_type": score_type, "mode": MODE,
                   "budget": budget, "owl_threshold": thresholds.get(budget) if budget else None, "theta": theta,
                   "selection": "dev F1", "split": split, "vocabulary": "coco80", "hyponyms": False, "seed": SEED},
                  os.path.join(d, "config.json"))
        np.save(os.path.join(d, "counts.npy"), metrics_func(records, theta)[1])
        summary[f"{cond}_{split}"] = {k: m[k] for k in ("top1_acc", "top5_acc", "precision", "recall", "f1", "mean_iou",
                                                        "boxes_per_image", "kept_per_image", "top1_acc_ci95", "f1_ci95")}
    save_json(summary, summary_path)
    print("E2", method, {k: round(v, 3) for k, v in summary["theta_test"].items() if isinstance(v, float)})
    return summary


def evaluate_closed(coco, splits, dets, det_name="YOLO11x"):
    """
    Close Vocabulary モデル（検出器自身のクラスと検出スコア）を評価する．θ は検出スコアに対して dev の F値で選ぶ．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): load_detections の戻り値．
        det_name (str): 検出器の名前．

    Returns:
        dict: evaluate_method と同じ形式の結果．
    """
    method = f"{det_name}-closed"
    out_dir = os.path.join(OUTPUT_DIR, "E2_pipeline", method)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        return json.load(open(summary_path))
    records = {s: E17_TRAIN.build_records(coco, dets[(det_name, s)], splits[s], E17_TRAIN.closed_labels_fn(dets[(det_name, s)]))
               for s in SPLITS}
    theta = E17_TRAIN.select_theta(records["dev"], os.path.join(out_dir, "all"))
    summary = {"method": method, "detector": det_name, "embedder": "closed", "score_type": "det",
               "budget_selection": None, "budget_all": None, "budget": None, "theta": theta}
    for split, (cond, t) in itertools.product(SPLITS, [("all", -np.inf), ("theta", theta)]):
        m = metrics_with_ci(records[split], t)
        m["boxes_per_image"] = float(np.mean([len(r["boxes"]) for r in records[split]]))
        m["kept_per_image"] = float(np.mean([(r["scores"] >= t).sum() for r in records[split]]))
        d = result_dir("E2_pipeline", method, f"{cond}_{split}")
        save_json(m, os.path.join(d, "metrics.json"))
        save_json({"detector": det_name, "embedder": "closed (detector labels)", "conf": 0.25, "theta": t,
                   "selection": "dev F1", "split": split, "seed": SEED}, os.path.join(d, "config.json"))
        np.save(os.path.join(d, "counts.npy"), metrics_func(records[split], t)[1])
        summary[f"{cond}_{split}"] = {k: m[k] for k in ("top1_acc", "top5_acc", "precision", "recall", "f1", "mean_iou",
                                                        "boxes_per_image", "kept_per_image", "top1_acc_ci95", "f1_ci95")}
    save_json(summary, summary_path)
    print("E2", method, {k: round(v, 3) for k, v in summary["theta_test"].items() if isinstance(v, float)})
    return summary


def person_recall(coco, splits, dets, thresholds):
    """
    test の正解を人物と人物以外に分け，検出器ごとのクラス非依存の再現率（IoU 0.5）を CSV に保存する．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): load_detections の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．

    Returns:
        pandas.DataFrame: 検出器ごとの "person"，"other"，"all" の再現率と1枚あたりのボックス数．
    """
    getters = {"DETR": lambda d: d["boxes"], "YOLO11x": lambda d: d["boxes"],
               **{f"OWLv2@{b}": (lambda d, t=thresholds[b]: d["boxes"][d["scores"] >= t]) for b in PERSON_BUDGETS}}
    rows = []
    for name, get in getters.items():
        det_key = name.split("@")[0]
        hit, n, n_boxes = {"person": 0, "other": 0}, {"person": 0, "other": 0}, 0
        for i in splits["test"]:
            gt_boxes, labels = gt_instances(coco, i)
            boxes = np.asarray(get(dets[(det_key, "test")][i]), np.float32).reshape(-1, 4)
            n_boxes += len(boxes)
            if not labels:
                continue
            found = E17_TRAIN.box_iou(boxes, gt_boxes).max(axis=0) >= 0.5 if len(boxes) else np.zeros(len(labels), bool)
            for f, label in zip(found, labels):
                key = "person" if label == "person" else "other"
                hit[key] += int(f)
                n[key] += 1
        rows.append({"detector": name, "boxes_per_image": n_boxes / len(splits["test"]),
                     "recall_person": hit["person"] / n["person"], "recall_other": hit["other"] / n["other"],
                     "recall_all": (hit["person"] + hit["other"]) / (n["person"] + n["other"]),
                     "n_person": n["person"], "n_other": n["other"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E2_pipeline", "recall_person_test.csv"), index=False)
    return df


def run_e2(pool, coco, splits, dets, thresholds, alpha_name):
    """
    E2：D × F × 棄却のスコアの全ての組を評価し，古い提案手法と新しい提案手法の差を検定する．

    Args:
        pool (EmbedderPool): 埋め込みモデル．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): load_detections の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．
        alpha_name (str): E1 で選んだ Alpha-CLIP の F の名前．

    Returns:
        pandas.DataFrame: 全ての組の test の結果．
    """
    rows = []
    configs = [(d, e, t, None) for e, d, t in itertools.product(["CLIP-ViT-B32", "CLIP-ViT-B16", alpha_name], DETECTORS, SCORE_TYPES)]
    configs += [("OWLv2", e, t, b) for e, t, b in itertools.product(["CLIP-ViT-B32", "CLIP-ViT-B16", alpha_name], SCORE_TYPES,
                                                                    FIXED_OWL_BUDGETS)]
    summaries = [evaluate_method(pool, coco, splits, dets, thresholds, *c) for c in configs]
    summaries.append(evaluate_closed(coco, splits, dets))
    for s in summaries:
        for cond in ("all", "theta"):
            rows.append({"method": s["method"], "detector": s["detector"], "embedder": s["embedder"],
                         "score_type": s["score_type"], "budget_selection": s.get("budget_selection"), "rejection": cond,
                         "budget": s["budget_all"] if cond == "all" else s["budget"],
                         "theta": -np.inf if cond == "all" else s["theta"], **s[f"{cond}_test"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "E2_pipeline", "summary_test.csv"), index=False)
    tests = []
    old = f"{OLD_METHOD[0]}+{OLD_METHOD[1]}+vlm"
    for score_type, cond, suffix in itertools.product(SCORE_TYPES, ("all", "theta"), ["", "@20", "@50"]):
        new = f"OWLv2+{alpha_name}+{score_type}{suffix}"
        for ref in (old, YOLO_CLOSED, f"DETR+{alpha_name}+{score_type}",
                    f"OWLv2+CLIP-ViT-B32+{score_type}{suffix}"):
            a = np.load(os.path.join(result_dir("E2_pipeline", ref, f"{cond}_test"), "counts.npy"))
            b = np.load(os.path.join(result_dir("E2_pipeline", new, f"{cond}_test"), "counts.npy"))
            tests.append({"baseline": ref, "new": new, "rejection": cond, **{f"{k}_{x}": v[x] for k, v in
                          paired_bootstrap(a, b).items() for x in ("diff", "ci95", "p_value")}})
    pd.DataFrame(tests).to_csv(os.path.join(OUTPUT_DIR, "E2_pipeline", "paired_bootstrap_test.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# E3：速度，E4：深さごとの確信度と比較画像
# ---------------------------------------------------------------------------
def chosen_setting(method):
    """
    E2 で選んだ設定（θ あり）を読み込む．

    Args:
        method (str): "D+F+score_type"．

    Returns:
        dict: E2 の summary.json．
    """
    return json.load(open(os.path.join(OUTPUT_DIR, "E2_pipeline", method, "summary.json")))


def kept_boxes(det, setting, thresholds):
    """
    選んだ予算の閾値で絞った検出結果を返す．

    Args:
        det (dict): 1枚の検出結果．
        setting (dict): chosen_setting の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．

    Returns:
        tuple[np.ndarray, np.ndarray]: ボックス（K×4）と検出スコア（K）．
    """
    if setting["budget"] is None:
        return np.asarray(det["boxes"]).reshape(-1, 4), np.asarray(det["scores"])
    keep = det["scores"] >= thresholds[setting["budget"]]
    return det["boxes"][keep], det["scores"][keep]


def annotate(embedder, emb_name, image, boxes, det_scores, setting, vocab):
    """
    1枚の画像のボックスに粒度可変アノテーションを行う（深さ0の確信度が θ 未満のボックスは棄却する）．

    Args:
        embedder: 埋め込みモデル．
        emb_name (str): EMBEDDER_SPECS のキー．
        image (PIL.Image.Image): 画像．
        boxes (np.ndarray): ボックス（K×4）．
        det_scores (np.ndarray): 検出スコア（K）．
        setting (dict): chosen_setting の戻り値．
        vocab (list[tuple[str, str]]): 深さ0の語彙．

    Returns:
        list[dict]: 残したボックスの "box"，"det_score"，"chain"．
    """
    feats = region_features(embedder, emb_name, image, boxes)
    out = []
    for k in range(len(boxes)):
        chain = granular_annotation(embedder, feats[k:k + 1], vocab, -np.inf, MODE)
        if not chain:
            continue
        s0 = chain[0]["score"] * (det_scores[k] if setting["score_type"] == "joint" else 1.0)
        if s0 >= setting["theta"]:
            out.append({"box": [float(v) for v in boxes[k]], "det_score": float(det_scores[k]), "chain": chain})
    return out


def closed_annotations(det, theta):
    """
    Close Vocabulary モデルの出力をアノテーションの形式（深さ0のみ）に変換する．

    Args:
        det (dict): 1枚の検出結果（"boxes"，"scores"，"labels"）．
        theta (float): 検出スコアの閾値．

    Returns:
        list[dict]: "box"，"det_score"，"chain"（深さ0の1要素）．
    """
    return [{"box": [float(v) for v in b], "det_score": float(s),
             "chain": [{"depth": 0, "word": l, "synset": None, "score": float(s), "n_candidates": 80, "top": [[l, float(s)]]}]}
            for b, s, l in zip(det["boxes"], det["scores"], det["labels"]) if s >= theta]


def run_e3_e4(pool, coco, splits, dets, thresholds, methods, figure_methods=None, filename="fig_comparison.png"):
    """
    E3：test の先頭 N_SPEED_IMAGES 枚で，検出・特徴計算・WordNet の細分化を含む1枚あたりの時間を測る．
    E4：同じ画像のアノテーションから，深さ0が正しいインスタンスの深さごとの確信度を集計し，比較画像を保存する．

    Args:
        pool (EmbedderPool): 埋め込みモデル．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): load_detections の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．
        methods (dict[str, str]): 表示名 → "D+F+score_type"．
        figure_methods (list[str] | None): 比較画像に並べる表示名（None なら全て）．
        filename (str): 比較画像のファイル名．
    """
    vocab = coco80_vocab()
    ids = splits["test"][:N_SPEED_IMAGES]
    for label, method in methods.items():
        d = result_dir("E3_speed", method, "test100")
        if os.path.exists(os.path.join(d, "metrics.json")):
            continue
        setting = chosen_setting(method)
        closed = setting["embedder"] == "closed"
        embedder = None if closed else pool.get(setting["embedder"])
        annotations, times = {}, []
        if not closed:
            annotate(embedder, setting["embedder"], load_image(coco, ids[0]), np.zeros((1, 4)) + [0, 0, 50, 50],
                     np.ones(1), setting, vocab)   # ウォームアップ
        for i in tqdm(ids, desc=f"E3 {label}", leave=False):
            image = load_image(coco, i)
            det = dets[(setting["detector"], "test")][i]
            boxes, scores = kept_boxes(det, setting, thresholds)
            if closed:
                ann, sec = closed_annotations(det, setting["theta"]), 0.0
            else:
                ann, sec = timed(lambda: annotate(embedder, setting["embedder"], image, boxes, scores, setting, vocab))
            annotations[i] = ann
            times.append({"detect": float(dets[(setting["detector"], "test")][i]["time"]), "annotate": sec,
                          "n_boxes": len(boxes), "n_kept": len(ann),
                          "max_depth": max([len(a["chain"]) - 1 for a in ann], default=-1)})
        t = pd.DataFrame(times)
        m = {"label": label, "method": method, "n_images": len(ids), "detect_sec_per_image": float(t.detect.mean()),
             "annotate_sec_per_image": float(t.annotate.mean()),
             "total_sec_per_image": float((t.detect + t.annotate).mean()), "boxes_per_image": float(t.n_boxes.mean()),
             "kept_per_image": float(t.n_kept.mean()), "annotate_sec_per_kept_box": float(t.annotate.sum() / max(t.n_kept.sum(), 1)),
             "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0}
        save_json(m, os.path.join(d, "metrics.json"))
        save_json({"method": method, "setting": setting, "images": "test[:100]", "seed": SEED}, os.path.join(d, "config.json"))
        save_json({str(k): v for k, v in annotations.items()}, os.path.join(d, "annotations.json"))
        depth_table(coco, annotations, method)
    draw_comparison(coco, ids, {k: methods[k] for k in (figure_methods or methods)}, filename)


def depth_table(coco, annotations, method):
    """
    深さ0のラベルが正解（IoU 0.5 以上で対応したGTと一致）のインスタンスについて，深さごとの件数と平均確信度を保存する．

    Args:
        coco (dict): load_coco の戻り値．
        annotations (dict[int, list[dict]]): 画像ID → annotate の戻り値．
        method (str): 手法名．
    """
    rows = []
    for i, ann in annotations.items():
        gt_boxes, gt_labels = gt_instances(coco, i)
        if not ann or not gt_labels:
            continue
        iou = E17_TRAIN.box_iou(np.array([a["box"] for a in ann], np.float32), gt_boxes)
        pairs = E17_TRAIN.greedy_match(iou, list(np.argsort([-a["det_score"] for a in ann], kind="stable")))
        for p, g in pairs.items():
            if ann[p]["chain"][0]["word"] != gt_labels[g]:
                continue
            rows += [{"depth": s["depth"], "score": s["score"], "n_candidates": s["n_candidates"]} for s in ann[p]["chain"]]
    df = pd.DataFrame(rows).groupby("depth").agg(n=("score", "size"), mean_score=("score", "mean"),
                                                 mean_candidates=("n_candidates", "mean")).reset_index()
    df.to_csv(os.path.join(result_dir("E4_depth", method, "test100"), "depth_table.csv"), index=False)


def draw_comparison(coco, ids, methods, filename="fig_comparison.png"):
    """
    同じ画像に対する古い提案手法と新しい提案手法のアノテーション（深さ0〜2の語）を並べた比較画像を保存する．

    Args:
        coco (dict): load_coco の戻り値．
        ids (list[int]): 候補の画像ID（E3 と同じ）．
        methods (dict[str, str]): 表示名 → 手法名．
        filename (str): 保存するファイル名．
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    path = os.path.join(OUTPUT_DIR, "E4_depth", filename)
    if os.path.exists(path):
        return
    anns = {label: json.load(open(os.path.join(result_dir("E3_speed", m, "test100"), "annotations.json")))
            for label, m in methods.items()}
    chosen = [i for i in ids if 3 <= len(gt_instances(coco, i)[1]) <= 8][:N_QUALITATIVE]
    fig, axes = plt.subplots(len(chosen), len(methods), figsize=(7 * len(methods), 5 * len(chosen)))
    for row, i in enumerate(chosen):
        image = load_image(coco, i)
        for col, (label, ann) in enumerate(anns.items()):
            ax = axes[row, col]
            ax.imshow(image)
            ax.axis("off")
            items = sorted(ann[str(i)], key=lambda a: -a["chain"][0]["score"])[:8]
            for k, a in enumerate(items):
                x0, y0, x1, y1 = a["box"]
                color = plt.cm.tab10(k % 10)
                ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2))
                text = " > ".join(f"{s['word']}({s['score']:.2f})" for s in a["chain"][:3])
                ax.text(x0 + 2, y0 + 2, text, fontsize=6.5, color="white", va="top",
                        bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "lw": 0})
            ax.set_title(f"{label}  id={i}  ({len(ann[str(i)])} boxes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def train():
    """ex019 の全ての実験を実行する．"""
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coco = load_coco()
    splits = load_splits(coco)
    pool = EmbedderPool()
    alpha_name = run_e1(pool, coco, splits)
    e1_tests = [mcnemar_e1(a, b) for a, b in [("CLIP-ViT-B32", alpha_name), ("CLIP-ViT-B16", alpha_name),
                                              ("CLIP-ViT-B32-GELU", "CLIP-ViT-B32")]]
    pd.DataFrame(e1_tests).to_csv(os.path.join(OUTPUT_DIR, "E1_naming", "mcnemar_test.csv"), index=False)
    dets, thresholds = load_detections(coco, splits)
    save_json({str(k): v for k, v in thresholds.items()}, os.path.join(OUTPUT_DIR, "owl_budget_thresholds.json"))
    run_e2(pool, coco, splits, dets, thresholds, alpha_name)
    person_recall(coco, splits, dets, thresholds)

    def best(prefix, suffix=""):
        """棄却のスコアを dev の F値で選んだ手法名を返す．"""
        return max((f"{prefix}+{s}{suffix}" for s in SCORE_TYPES), key=lambda m: chosen_setting(m)["theta_dev"]["f1"])
    # order_029 の2手法（図 fig_comparison.png）
    methods = {"Old: DETR + CLIP-ViT-B/32": f"{OLD_METHOD[0]}+{OLD_METHOD[1]}+vlm",
               "New: OWLv2 + Alpha-CLIP": best(f"OWLv2+{alpha_name}")}
    run_e3_e4(pool, coco, splits, dets, thresholds, methods)
    # order_030：YOLO11x と予算を固定した新しい提案手法を追加（図 fig_comparison_order030.png）
    methods.update({"YOLO11x (closed)": YOLO_CLOSED,
                    "New: OWLv2@20 + Alpha-CLIP": best(f"OWLv2+{alpha_name}", "@20"),
                    "New: OWLv2@50 + Alpha-CLIP": best(f"OWLv2+{alpha_name}", "@50")})
    run_e3_e4(pool, coco, splits, dets, thresholds, methods,
              ["Old: DETR + CLIP-ViT-B/32", "YOLO11x (closed)", "New: OWLv2@20 + Alpha-CLIP", "New: OWLv2@50 + Alpha-CLIP"],
              "fig_comparison_order030.png")


if __name__ == "__main__":
    train()
