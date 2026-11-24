"""
ex009の実行スクリプト：COCOでの提案手法・ハイブリッド手法・YOLO11x単体の比較（order_011の要求1，3）．

学習は行わない．各手法の深さ0の決め方は以下のとおりである．
- yolo:     YOLO11xの語をそのまま出力する（比較対象，粒度変更なし）
- proposed: DETRのボックスのクロップに対し，SigLIPでCOCOの80語から argmax の語を選ぶ（確信度はシグモイドまたはソフトマックス）
- hybrid:   YOLO11xの語を深さ0とする（確信度はYOLOの検出スコア）
proposed と hybrid は，深さ0の語のSynsetから下位語の集合が空になるまでSigLIPで下位語をたどる．

評価:
- COCO（dev500，test500）: 深さ0のTop-1正解率，正命名率 r（深さ0が正解の数 / GT総数），N_h（深さ1以降に細分化できた正解数），
  深さ0が誤った枝と正しい枝を確信度で区別するAUROC
- LVIS（lvis_dev，lvis_test）: COCOのSynsetからLVISのSynsetまでの経路を正解とする階層的適合率 hP，再現率 hR，F値 hF，
  深さ1の正解率とAUROC，累積確信度 c_d = Π p_j による停止規則（τ はlvis_devで hF が最大となる値）

出力: outputs/ex009_coco_siglip_wordnet_annotation/{手法}/{確信度}/{分割}_th{閾値}/{seed}/
    config.json，instances.json，metrics.json，stopping_curve.csv
    および直下の table_conditions.csv，table_stopping.csv
"""
import functools
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet as wn
from scipy.stats import rankdata
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image, make_crop  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.vlm import vlm_logits  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402
from data import SPLITS, build_split_instances, load_context  # noqa: E402
from model import NORMALIZATIONS, hierarchical_chain, load_detr, load_model, load_yolo, normalize  # noqa: E402

# ===== 実験条件（order_011） =====
EXPERIMENT = "ex009_coco_siglip_wordnet_annotation"
SEEDS = [42]
CROP_MODE = "pad"
IOU_THRESHOLD = 0.5
# 手法 → 検出スコアの閾値（評価インスタンス数が近い組：DETR 0.5 と YOLO 0.25，DETR 0.97 と YOLO 0.7）
METHODS = {"yolo": [0.25, 0.7], "proposed": [0.5, 0.97], "hybrid": [0.25, 0.7]}
DETECTOR = {"yolo": "yolo", "proposed": "detr", "hybrid": "yolo"}
TAUS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEPTH_LIMITS = [0, 1, 2, 3]
COCO_WORDS = list(COCO_TO_SYNSET)


def target_dir(method, normalization, split, threshold, seed):
    """
    1条件の保存先を返す．

    Args:
        method (str): 手法名．
        normalization (str): 確信度の正規化関数（yolo は "none"）．
        split (str): 分割名．
        threshold (float): 検出スコアの閾値．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, normalization, f"{split}_th{threshold}", str(seed))


def normalizations(method):
    """
    手法で比較する確信度の正規化関数を返す．

    Args:
        method (str): 手法名．

    Returns:
        list[str]: 正規化関数のリスト（yolo はVLMを用いないため ["none"]）．
    """
    return ["none"] if method == "yolo" else NORMALIZATIONS


class FeatureCache:
    """クロップのSigLIPの画像特徴と，(クロップ，深さ0のSynset) ごとの下位語の展開結果を保持する．"""

    def __init__(self, vlm, coco):
        """
        Args:
            vlm (SigLIPModel): VLM．
            coco (COCO): instancesのCOCOオブジェクト．
        """
        self.vlm, self.coco = vlm, coco
        self.features, self.chains = {}, {}

    @staticmethod
    def key(ins):
        """
        クロップを識別するキーを返す．

        Args:
            ins (dict): "image_id"，"box" を持つインスタンス．

        Returns:
            tuple: (画像ID, 丸めたボックス)．
        """
        return ins["image_id"], tuple(round(v, 2) for v in ins["box"])

    def prepare(self, instances, batch_size=256):
        """
        未計算のクロップの画像特徴をまとめて計算する．

        Args:
            instances (list[dict]): インスタンス．
            batch_size (int): 一度にVLMへ入力するクロップ数．

        Returns:
            None
        """
        todo = sorted({self.key(i): i for i in instances if self.key(i) not in self.features}.items(),
                      key=lambda kv: kv[0][0])
        for start in tqdm(range(0, len(todo), batch_size), desc="SigLIP features", leave=False):
            chunk = todo[start:start + batch_size]
            crops = [make_crop(load_image(self.coco, k[0]), ins["box"], CROP_MODE) for k, ins in chunk]
            for (k, _), feat in zip(chunk, self.vlm.image_features(crops)):
                self.features[k] = feat

    def root_logits(self, instances):
        """
        全インスタンスのCOCOの80語に対するロジットを計算する．

        Args:
            instances (list[dict]): インスタンス（長さN）．

        Returns:
            torch.Tensor: ロジット（N×80）．
        """
        feats = torch.stack([self.features[self.key(i)] for i in instances])
        return vlm_logits(self.vlm, feats, COCO_WORDS)

    def chain(self, ins, word):
        """
        深さ0の語から下位語を最後まで展開した結果を返す（キャッシュする）．

        Args:
            ins (dict): インスタンス．
            word (str): 深さ0の語（COCOのカテゴリ名）．

        Returns:
            list[dict]: hierarchical_chain の戻り値．
        """
        k = (self.key(ins), word)
        if k not in self.chains:
            self.chains[k] = hierarchical_chain(self.vlm, self.features[self.key(ins)], word, COCO_TO_SYNSET[word])
        return self.chains[k]


def annotate(cache, method, normalization, instances):
    """
    全インスタンスの深さ0を決め，proposed と hybrid では正誤によらず下位語を最後まで展開する．

    Args:
        cache (FeatureCache | None): 画像特徴のキャッシュ（yolo では None）．
        method (str): 手法名．
        normalization (str): 確信度の正規化関数．
        instances (list[dict]): インスタンス．

    Returns:
        list[dict]: インスタンスに "root_word"，"root_conf"，"top1_correct"，"chain" を追記したもの．
    """
    if method == "proposed":
        cache.prepare(instances)
        probs = normalize(cache.root_logits(instances), normalization).cpu().numpy() if instances else []
    elif method == "hybrid":
        cache.prepare(instances)
    records = []
    for i, ins in enumerate(tqdm(instances, desc=f"{method} {normalization}", leave=False)):
        if method == "proposed":
            best = int(np.argmax(probs[i]))
            word, conf = COCO_WORDS[best], float(probs[i][best])
        else:
            word, conf = ins["yolo_word"], float(ins["det_score"])
        if method == "yolo":
            chain = [{"depth": 0, "word": word, "synset": COCO_TO_SYNSET[word], "conf": conf, "n_candidates": None}]
        else:
            chain = [{"depth": s["depth"], "word": s["word"], "synset": s["synset"], "conf": s[f"conf_{normalization}"],
                      "n_candidates": s["n_candidates"]} for s in cache.chain(ins, word)]
            chain[0]["conf"] = conf
        records.append(ins | {"root_word": word, "root_conf": conf, "top1_correct": word == ins["gt_name"], "chain": chain})
    return records


# ===================== 評価指標 =====================

def output_depth(chain, rule, value):
    """
    停止規則に従って出力する深さを決める．

    Args:
        chain (list[dict]): 下位語の展開結果．
        rule (str): "tau"（累積確信度が value 以上の最も深い段階）または "depth"（深さの上限 value）．
        value (float): τ または深さの上限．

    Returns:
        int: 出力する深さ（-1は棄却）．
    """
    if rule == "depth":
        return min(int(value), len(chain) - 1)
    above = np.nonzero(np.cumprod([s["conf"] for s in chain]) >= value)[0]
    if len(above) == 0 or above[0] != 0:
        return -1
    return int(above[-1])  # 累積確信度は単調非増加であるため，閾値以上の深さは先頭から連続する


@functools.lru_cache(maxsize=None)
def is_strict_descendant(node_name, ancestor_name):
    """
    node が ancestor の真の下位語であるかを判定する．

    Args:
        node_name (str): Synset名．
        ancestor_name (str): Synset名．

    Returns:
        bool: 真の下位語であるか．
    """
    node, ancestor = wn.synset(node_name), wn.synset(ancestor_name)
    return node != ancestor and any(ancestor in path for path in node.hypernym_paths())


def lvis_metrics(records, rule, value):
    """
    LVISの経路を正解とする階層的適合率・再現率・F値（マイクロ平均）を計算する．
    LVISのSynsetより細かい出力は正誤を判定できないため除き，棄却したインスタンスは再現率の分母のみに含める．

    Args:
        records (list[dict]): "gt_path"，"lvis_synset" を持つ記録．
        rule (str): 停止規則．
        value (float): 停止規則のパラメータ．

    Returns:
        dict: hP，hR，hF，exact（LVISのSynsetまで出力した割合），coverage（棄却されなかった割合）．
    """
    inter = pred = gt = 0
    exact, covered = [], []
    for r in records:
        depth = output_depth(r["chain"], rule, value)
        out = [s["synset"] for s in r["chain"][:depth + 1]]
        out = [s for s in out if not is_strict_descendant(s, r["lvis_synset"])]
        inter += len(set(out) & set(r["gt_path"]))
        pred += len(out)
        gt += len(r["gt_path"])
        exact.append(r["lvis_synset"] in out)
        covered.append(depth >= 0)
    h_p = inter / pred if pred else float("nan")
    h_r = inter / gt if gt else float("nan")
    h_f = 2 * h_p * h_r / (h_p + h_r) if pred and h_p + h_r > 0 else 0.0
    return {"hP": float(h_p), "hR": float(h_r), "hF": float(h_f), "exact": float(np.mean(exact)),
            "coverage": float(np.mean(covered))}


def roc_auc(labels, scores):
    """
    正例を負例より高いスコアとする確率（AUROC）を順位統計量から計算する．

    Args:
        labels (np.ndarray): 正例をTrueとするラベル（長さN）．
        scores (np.ndarray): スコア（長さN）．

    Returns:
        float: AUROC（正例と負例の一方が無い場合は NaN）．
    """
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def coco_metrics(records, n_gt):
    """
    COCOのGTで判定できる指標を計算する．

    Args:
        records (list[dict]): 記録．
        n_gt (int): GTインスタンスの総数．

    Returns:
        dict: n，top1，r，n_h，wrong_branch_rate，auroc_root / last / cum（深さ0の正誤を各確信度で区別するAUROC）．
    """
    correct = np.array([r["top1_correct"] for r in records])
    root = np.array([r["root_conf"] for r in records])
    last = np.array([r["chain"][-1]["conf"] for r in records])
    cum = np.array([np.prod([s["conf"] for s in r["chain"]]) for r in records])
    return {"n": len(records), "n_gt": n_gt, "top1": float(correct.mean()), "r": float(correct.sum() / n_gt),
            "n_h": int(sum(c and len(r["chain"]) > 1 for c, r in zip(correct, records))),
            "wrong_branch_rate": float(1 - correct.mean()),
            "auroc_root": roc_auc(correct, root), "auroc_last": roc_auc(correct, last), "auroc_cum": roc_auc(correct, cum)}


def depth1_metrics(records):
    """
    深さ0が正解のLVISのインスタンスについて，深さ1の正解率と，深さ1の確信度による正誤のAUROCを計算する．

    Args:
        records (list[dict]): LVISの分割の記録．

    Returns:
        dict: depth1_n，depth1_correct，depth1_auroc．
    """
    good = [r for r in records if r["top1_correct"] and len(r["chain"]) > 1]
    if not good:
        return {"depth1_n": 0, "depth1_correct": float("nan"), "depth1_auroc": float("nan")}
    labels = np.array([r["chain"][1]["synset"] in r["gt_path"] for r in good])
    return {"depth1_n": len(good), "depth1_correct": float(labels.mean()),
            "depth1_auroc": roc_auc(labels, np.array([r["chain"][1]["conf"] for r in good]))}


def stopping_curve(records, is_lvis):
    """
    全ての停止規則（τ，深さの上限）についてLVISの指標を計算する．

    Args:
        records (list[dict]): 記録．
        is_lvis (bool): LVISの分割か（COCOの分割では None を返す）．

    Returns:
        pd.DataFrame | None: 行が停止規則，列が rule，value と各指標．
    """
    if not is_lvis:
        return None
    rows = [{"rule": rule, "value": v} | lvis_metrics(records, rule, v)
            for rule, values in [("tau", TAUS), ("depth", DEPTH_LIMITS)] for v in values]
    return pd.DataFrame(rows)


# ===================== 実行 =====================

def run_condition(cache, ctx, method, normalization, split, threshold, seed):
    """
    1条件（手法 × 確信度 × 分割 × 閾値）を評価して保存する（完了済みならスキップする）．

    Args:
        cache (FeatureCache): 画像特徴のキャッシュ．
        ctx (dict): load_context の戻り値．
        method (str): 手法名．
        normalization (str): 確信度の正規化関数．
        split (str): 分割名．
        threshold (float): 検出スコアの閾値．
        seed (int): シード．

    Returns:
        None
    """
    out_dir = target_dir(method, normalization, split, threshold, seed)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        return
    start = time.time()
    instances, n_gt = build_split_instances(ctx, split, DETECTOR[method], threshold, IOU_THRESHOLD)
    records = annotate(cache, method, normalization, instances)
    is_lvis = split.startswith("lvis")
    metrics = coco_metrics(records, n_gt) | (depth1_metrics(records) | lvis_metrics(records, "tau", 0.0) if is_lvis else {})
    os.makedirs(out_dir, exist_ok=True)
    curve = stopping_curve(records, is_lvis)
    if curve is not None:
        curve.to_csv(os.path.join(out_dir, "stopping_curve.csv"), index=False)
    config = {"method": method, "normalization": normalization, "split": split, "threshold": threshold,
              "detector": DETECTOR[method], "vlm": "SigLIP-ViT-B16" if method != "yolo" else None,
              "n_images": len(ctx["splits"][split]), "crop_mode": CROP_MODE, "iou_threshold": IOU_THRESHOLD,
              "seed": seed, "elapsed_sec": time.time() - start}
    for name, obj in [("instances.json", records), ("metrics.json", metrics), ("config.json", config)]:
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, indent=None if name == "instances.json" else 4)
    print(method, normalization, split, threshold, {k: metrics.get(k) for k in ["n", "top1", "n_h", "hF"]})


def conditions():
    """
    実行する（手法，確信度，分割，閾値）の組を返す．

    Returns:
        list[tuple[str, str, str, float]]: 条件のリスト．
    """
    return [(m, norm, split, t) for m in METHODS for norm, split, t in
            itertools.product(normalizations(m), SPLITS, METHODS[m])]


def write_tables(seed):
    """
    集計表を outputs/ex009_coco_siglip_wordnet_annotation/ 直下に保存する．
    table_stopping.csv は，lvis_dev で hF が最大の τ を選び，lvis_test の停止なし・深さ1まで・τ の結果を並べる．

    Args:
        seed (int): シード．

    Returns:
        None
    """
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    rows = []
    for method, norm, split, threshold in conditions():
        with open(os.path.join(target_dir(method, norm, split, threshold, seed), "metrics.json")) as f:
            rows.append({"method": method, "normalization": norm, "split": split, "threshold": threshold} | json.load(f))
    pd.DataFrame(rows).to_csv(os.path.join(root, "table_conditions.csv"), index=False)
    stop_rows = []
    for method in METHODS:
        for norm, threshold in itertools.product(normalizations(method), METHODS[method]):
            curves = {s: pd.read_csv(os.path.join(target_dir(method, norm, s, threshold, seed), "stopping_curve.csv"))
                      for s in ["lvis_dev", "lvis_test"]}
            dev = curves["lvis_dev"][curves["lvis_dev"]["rule"] == "tau"]
            tau = float(dev.loc[dev["hF"].idxmax(), "value"])
            test = curves["lvis_test"]
            for label, rule, value in [("none", "tau", 0.0), ("depth1", "depth", 1), ("tau", "tau", tau)]:
                row = test[(test["rule"] == rule) & np.isclose(test["value"], value)].iloc[0]
                stop_rows.append({"method": method, "normalization": norm, "threshold": threshold, "stopping": label,
                                  "tau": value if rule == "tau" else None} | row.drop(["rule", "value"]).to_dict())
    pd.DataFrame(stop_rows).to_csv(os.path.join(root, "table_stopping.csv"), index=False)


def main():
    """全条件（手法 × 確信度 × 分割 × 閾値 × seed）を評価し，集計表を作成する（完了済みの条件はスキップ）．"""
    for seed in SEEDS:
        ctx = load_context(seed, lambda: load_detr(seed), lambda: load_yolo(seed))
        os.makedirs(os.path.join(OUTPUT_DIR, EXPERIMENT), exist_ok=True)
        cache = FeatureCache(load_model(seed), ctx["coco"])
        for method, norm, split, threshold in conditions():
            run_condition(cache, ctx, method, norm, split, threshold, seed)
        write_tables(seed)


if __name__ == "__main__":
    main()
