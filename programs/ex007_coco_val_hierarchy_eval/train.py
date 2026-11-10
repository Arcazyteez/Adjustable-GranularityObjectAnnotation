"""
ex007の実行スクリプト：粒度可変アノテーションの評価の改善（order_008）．

学習は行わない．提案手法（DETR + CLIP-ViT-B/32）とハイブリッド構成（YOLO11x + CLIP-ViT-B/32）について，
深さ0が不正解のインスタンスも除外せずにWordNetの下位語を最後まで展開し，以下を評価する．

- A1: 評価数 n が近い組（提案手法 θ=0.5 とYOLO11xの閾値0.25，θ=0.97 と0.7）での比較と深さ1の推定単語の一致率
- A5: 深さ0の誤りが下位語へ伝播するインスタンスの割合と，その確信度
- B1: 累積確信度 c_d = Π_{j≤d} p_j が τ を下回った段階で細分化を止める規則（c_0 < τ なら棄却）
- A3: τ はLVISの検証用分割（lvis_dev）で決定し，テスト用分割（test500，lvis_test）で報告する
- A4: LVIS v1のカテゴリのSynsetを細かい粒度の正解とし，階層的適合率・再現率（hP，hR，hF）で深さ1以降を評価する

出力: outputs/ex007_coco_val_hierarchy_eval/{手法}/{分割}_{閾値}/{seed}/
    config.json, instances.json, metrics.json, stopping_curve.csv
    および outputs/ex007_coco_val_hierarchy_eval/ 直下の集計表（table_*.csv）
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
from common.coco import category_names, load_coco, load_image, make_crop  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET, EXTRA_TO_SYNSET  # noqa: E402
from data import (attach_lvis, build_instances, detect_detr, detect_yolo, load_lvis,  # noqa: E402
                  make_splits)
from model import hierarchical_annotation_from_feature, load_detr, load_model, load_yolo, scores_from_features  # noqa: E402

# ===== 実験条件（order_008） =====
EXPERIMENT = "ex007_coco_val_hierarchy_eval"
SEEDS = [42]
VLM_NAME = "CLIP-ViT-B32"
CROP_MODE = "pad"
IOU_THRESHOLD = 0.5
# 手法 → 検出スコアの閾値（提案手法はDETRの θ，ハイブリッド構成はYOLO11xの閾値）
CONDITIONS = {
    "proposed": [0.5, 0.7, 0.9, 0.97],
    "hybrid": [0.25, 0.5, 0.7],
    "hybrid_switch": [0.25, 0.5, 0.7],       # COCO外の22語への切替（γ=0）
    "hybrid_switch_c0.5": [0.25, 0.5, 0.7],  # COCO外の22語への切替（γ=0.5）
}
SPLITS = ["dev500", "test500", "lvis_dev", "lvis_test"]
SWITCH_SPLITS = ["dev500", "test500"]  # 切替の構成はCOCOの80クラスでの誤切替の確認のみに用いる
# 評価数 n が近い（提案手法の条件，ハイブリッド構成の条件）の組
PAIRS = [(("proposed", 0.5), ("hybrid", 0.25)), (("proposed", 0.97), ("hybrid", 0.7))]
TAUS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEPTH_LIMITS = [0, 1, 2, 3]  # 比較：深さの上限で細分化を止める規則
WORD_TO_SYNSET = COCO_TO_SYNSET | EXTRA_TO_SYNSET


def target_dir(method, split, threshold, seed):
    """
    1条件の結果の保存先を返す．

    Args:
        method (str): 手法名．
        split (str): 分割名．
        threshold (float): 検出スコアの閾値．
        seed (int): シード．

    Returns:
        str: 保存先ディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, f"{split}_th{threshold}", str(seed))


# ===================== 深さ0の推定と下位語の展開 =====================

def decide_root(method, ins, scores, vocab, extra_words):
    """
    深さ0の単語，確信度，その決定に用いたモデルを返す（ハイブリッド構成の切替は論文の式(9)）．

    Args:
        method (str): 手法名．
        ins (dict): インスタンス（ハイブリッド構成では "yolo_word"，"det_score" を持つ）．
        scores (np.ndarray | None): 候補語彙 vocab に対するCLIPのスコア（長さV）．
        vocab (list[str]): 候補語彙（提案手法は80語，切替の構成は102語）．
        extra_words (set[str]): COCO外の語の集合．

    Returns:
        tuple[str, float, str]: 深さ0の単語，確信度，"clip" または "yolo"．
    """
    if method == "proposed":
        best = int(np.argmax(scores))
        return vocab[best], float(scores[best]), "clip"
    if method == "hybrid":
        return ins["yolo_word"], float(ins["det_score"]), "yolo"
    gamma = float(method.split("_c")[1]) if "_c" in method else 0.0
    best = int(np.argmax(scores))
    if vocab[best] in extra_words and scores[best] >= gamma:
        return vocab[best], float(scores[best]), "clip"
    return ins["yolo_word"], float(ins["det_score"]), "yolo"


class FeatureCache:
    """クロップの画像特徴と，(クロップ，深さ0の単語) ごとの下位語の展開結果を保持する．"""

    def __init__(self, vlm, coco):
        """
        Args:
            vlm (CLIPModel): CLIP．
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
        for start in tqdm(range(0, len(todo), batch_size), desc="image features", leave=False):
            chunk = todo[start:start + batch_size]
            crops = [make_crop(load_image(self.coco, k[0]), ins["box"], CROP_MODE) for k, ins in chunk]
            for (k, _), feat in zip(chunk, self.vlm.image_features(crops)):
                self.features[k] = feat

    def scores(self, ins, vocab):
        """
        クロップの候補語彙上のスコアを返す．

        Args:
            ins (dict): インスタンス．
            vocab (list[str]): 候補語彙．

        Returns:
            np.ndarray: スコア（長さV）．
        """
        return scores_from_features(self.vlm, self.features[self.key(ins)], vocab).cpu().numpy()

    def chain(self, ins, word, conf):
        """
        深さ0の単語から下位語を最後まで展開した結果を返す（深さ0の確信度のみ差し替える）．

        Args:
            ins (dict): インスタンス．
            word (str): 深さ0の単語．
            conf (float): 深さ0の確信度．

        Returns:
            list[dict]: 各深さの {"depth", "word", "synset", "conf", "n_candidates"}．
        """
        k = (self.key(ins), word)
        if k not in self.chains:
            self.chains[k] = hierarchical_annotation_from_feature(
                self.vlm, self.features[self.key(ins)], word, WORD_TO_SYNSET[word], 1.0)
        chain = [dict(step) for step in self.chains[k]]
        chain[0]["conf"] = conf
        chain[0]["n_candidates"] = None
        return chain


def annotate(cache, method, instances, vocab80, vocab102):
    """
    全インスタンスの深さ0を推定し，正誤によらず下位語を最後まで展開する．

    Args:
        cache (FeatureCache): 画像特徴のキャッシュ．
        method (str): 手法名．
        instances (list[dict]): インスタンス．
        vocab80 (list[str]): COCOの80クラス名．
        vocab102 (list[str]): COCOの80クラス名とCOCO外の22語．

    Returns:
        list[dict]: インスタンスに "root_word"，"root_conf"，"root_source"，"top1_correct"，"chain" を追記したもの．
    """
    cache.prepare(instances)
    vocab = vocab80 if method == "proposed" else vocab102
    extra = set(EXTRA_TO_SYNSET)
    records = []
    for ins in tqdm(instances, desc=method, leave=False):
        scores = cache.scores(ins, vocab) if method != "hybrid" else None
        word, conf, source = decide_root(method, ins, scores, vocab, extra)
        chain = cache.chain(ins, word, conf)
        chain[0]["n_candidates"] = len(vocab) if source == "clip" else 80
        records.append(ins | {"root_word": word, "root_conf": conf, "root_source": source,
                              "top1_correct": word == ins["gt_name"], "chain": chain})
    return records


# ===================== 評価指標 =====================

def cumulative_conf(chain):
    """
    各深さの累積確信度 c_d = Π_{j≤d} p_j を計算する．

    Args:
        chain (list[dict]): 下位語の展開結果．

    Returns:
        np.ndarray: 累積確信度（長さ len(chain)）．
    """
    return np.cumprod([step["conf"] for step in chain])


def output_depth(chain, rule, value):
    """
    停止規則に従って出力する深さを決める．

    Args:
        chain (list[dict]): 下位語の展開結果．
        rule (str): "tau"（累積確信度が value 未満になる前の深さ）または "depth"（深さの上限 value）．
        value (float): τ または深さの上限．

    Returns:
        int: 出力する深さ（-1は棄却）．
    """
    if rule == "depth":
        return min(int(value), len(chain) - 1)
    above = np.nonzero(cumulative_conf(chain) >= value)[0]
    if len(above) == 0 or above[0] != 0:
        return -1
    # 累積確信度は単調非増加であるため，閾値以上の深さは先頭から連続する
    return int(above[-1])


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


def hierarchical_counts(record, depth):
    """
    LVISの経路を正解として，出力した経路と正解の経路の重なりを数える．
    LVISのSynsetより細かい出力（正誤を判定できない語）は出力の経路から除く．

    Args:
        record (dict): "chain"，"gt_path"，"lvis_synset" を持つ記録．
        depth (int): 出力する深さ（-1は棄却）．

    Returns:
        tuple[int, int, int]: (|P∩G|, |P|, |G|)．
    """
    gt = set(record["gt_path"])
    predicted = [s["synset"] for s in record["chain"][:depth + 1]]
    predicted = [s for s in predicted if not is_strict_descendant(s, record["lvis_synset"])]
    return len(set(predicted) & gt), len(predicted), len(gt)


def coco_level_metrics(records, n_gt, rule, value):
    """
    COCOの80クラスで判定できる，停止規則を適用した後の指標を計算する．

    Args:
        records (list[dict]): 記録．
        n_gt (int): GTインスタンスの総数．
        rule (str): 停止規則．
        value (float): 停止規則のパラメータ．

    Returns:
        dict: coverage（棄却されなかった割合），precision（出力のうち深さ0が正解の割合），
            r（深さ0が正解の出力数 / GT総数），wrong_branch_outputs（誤った枝の出力数），mean_depth_correct．
    """
    depths = np.array([output_depth(r["chain"], rule, value) for r in records])
    correct = np.array([r["top1_correct"] for r in records])
    out = depths >= 0
    return {"coverage": float(out.mean()), "precision": float(correct[out].mean()) if out.any() else float("nan"),
            "r": float((correct & out).sum() / n_gt), "n_correct_outputs": int((correct & out).sum()),
            "wrong_branch_outputs": int((~correct & out).sum()),
            "mean_depth_correct": float(depths[correct & out].mean()) if (correct & out).any() else float("nan")}


def lvis_level_metrics(records, rule, value):
    """
    LVISの細かい粒度の正解に対する階層的適合率・再現率・F値（マイクロ平均）を計算する．

    Args:
        records (list[dict]): "gt_path" を持つ記録．
        rule (str): 停止規則．
        value (float): 停止規則のパラメータ．

    Returns:
        dict: hP，hR，hF，exact（LVISのSynsetまで到達した割合），coverage．
    """
    counts, exact, covered = [], [], []
    for r in records:
        depth = output_depth(r["chain"], rule, value)
        counts.append(hierarchical_counts(r, depth) if depth >= 0 else (0, 0, len(r["gt_path"])))
        exact.append(r["lvis_synset"] in [s["synset"] for s in r["chain"][:depth + 1]])
        covered.append(depth >= 0)
    inter, pred, gt = np.array(counts).sum(axis=0)
    h_p = inter / pred if pred else float("nan")
    h_r = inter / gt
    h_f = 2 * h_p * h_r / (h_p + h_r) if pred and (h_p + h_r) else 0.0
    return {"hP": float(h_p), "hR": float(h_r), "hF": float(h_f), "exact": float(np.mean(exact)),
            "coverage": float(np.mean(covered))}


def roc_auc_score(labels, scores):
    """
    正例（正しい枝）を負例（誤った枝）より高いスコアとする確率（AUROC）を，順位統計量から計算する．

    Args:
        labels (np.ndarray): 正例をTrueとするラベル（長さN）．
        scores (np.ndarray): スコア（長さN）．

    Returns:
        float: AUROC（同順位は0.5として数える）．
    """
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(scores)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def error_propagation_metrics(records):
    """
    深さ0の誤りの伝播に関する指標と，正しい枝・誤った枝を区別する確信度の性能（AUROC）を計算する．

    Args:
        records (list[dict]): 記録．

    Returns:
        dict: 誤った枝の割合，平均深さ，最深段の確信度の平均，累積確信度の平均，各確信度のAUROC．
    """
    correct = np.array([r["top1_correct"] for r in records])
    last = np.array([r["chain"][-1]["conf"] for r in records])
    cumulative = np.array([cumulative_conf(r["chain"])[-1] for r in records])
    root = np.array([r["root_conf"] for r in records])
    depth = np.array([len(r["chain"]) - 1 for r in records])
    both = 0 < correct.sum() < len(correct)
    return {
        "wrong_branch_rate": float(1 - correct.mean()),
        "mean_depth_correct": float(depth[correct].mean()), "mean_depth_wrong": float(depth[~correct].mean()),
        "mean_last_conf_correct": float(last[correct].mean()), "mean_last_conf_wrong": float(last[~correct].mean()),
        "mean_cum_conf_correct": float(cumulative[correct].mean()), "mean_cum_conf_wrong": float(cumulative[~correct].mean()),
        "auroc_last_conf": float(roc_auc_score(correct, last)) if both else float("nan"),
        "auroc_root_conf": float(roc_auc_score(correct, root)) if both else float("nan"),
        "auroc_cum_conf": float(roc_auc_score(correct, cumulative)) if both else float("nan"),
    }


def stopping_curve(records, n_gt, is_lvis):
    """
    全ての停止規則（τ，深さの上限）について指標を計算する．

    Args:
        records (list[dict]): 記録．
        n_gt (int): GTインスタンスの総数（LVISの分割では使わない）．
        is_lvis (bool): LVISの分割か．

    Returns:
        pd.DataFrame: 行が停止規則，列が rule，value と各指標．
    """
    rows = []
    for rule, values in [("tau", TAUS), ("depth", DEPTH_LIMITS)]:
        for value in values:
            metrics = coco_level_metrics(records, n_gt, rule, value)
            if is_lvis:
                metrics |= {f"lvis_{k}": v for k, v in lvis_level_metrics(records, rule, value).items()}
            rows.append({"rule": rule, "value": value} | metrics)
    return pd.DataFrame(rows)


def summarize(records, n_gt, is_lvis):
    """
    1条件の評価指標をまとめる．

    Args:
        records (list[dict]): 記録．
        n_gt (int): GTインスタンスの総数．
        is_lvis (bool): LVISの分割か．

    Returns:
        dict: 評価指標．
    """
    correct = [r["top1_correct"] for r in records]
    metrics = {"n": len(records), "n_gt": n_gt, "top1": float(np.mean(correct)), "r": float(sum(correct) / n_gt),
               "n_h": int(sum(correct)), "n_switched": int(sum(r["root_source"] == "clip" for r in records)
                                                            if records and "yolo_word" in records[0] else 0)}
    metrics |= error_propagation_metrics(records)
    if is_lvis:
        good = [r for r in records if r["top1_correct"] and len(r["chain"]) > 1]
        metrics["lvis_depth1_correct"] = float(np.mean([r["chain"][1]["synset"] in r["gt_path"] for r in good]))
        metrics["lvis_n_depth1"] = len(good)
    return metrics


def depth1_agreement(records_a, records_b):
    """
    両手法の深さ0が正解となった共通のGTインスタンスにおける，深さ1の推定単語の一致率を計算する．

    Args:
        records_a (list[dict]): 一方の手法の記録．
        records_b (list[dict]): 他方の手法の記録．

    Returns:
        dict: n_common（深さ1を持つ共通インスタンス数）と agreement．
    """
    def index(records):
        return {(r["image_id"], r["gt_index"]): r for r in records if r["top1_correct"] and len(r["chain"]) > 1}
    a, b = index(records_a), index(records_b)
    common = sorted(set(a) & set(b))
    agree = [a[k]["chain"][1]["word"] == b[k]["chain"][1]["word"] for k in common]
    return {"n_common": len(common), "agreement": float(np.mean(agree)) if agree else float("nan")}


# ===================== 実行 =====================

def run_condition(cache, method, split, threshold, image_ids, detections, lvis, vocab80, vocab102, seed):
    """
    1条件（手法 × 分割 × 閾値）を評価して保存する（完了済みならキャッシュを読み込む）．

    Args:
        cache (FeatureCache): 画像特徴のキャッシュ．
        method (str): 手法名．
        split (str): 分割名．
        threshold (float): 検出スコアの閾値．
        image_ids (list[int]): 分割の画像ID．
        detections (dict): 検出結果のキャッシュ．
        lvis (dict): LVISのアノテーション．
        vocab80 (list[str]): COCOの80クラス名．
        vocab102 (list[str]): 80クラスとCOCO外の22語．
        seed (int): シード．

    Returns:
        list[dict]: 記録．
    """
    out_dir = target_dir(method, split, threshold, seed)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        with open(os.path.join(out_dir, "instances.json")) as f:
            return json.load(f)
    start = time.time()
    instances, n_gt = build_instances(cache.coco, image_ids, detections, threshold, IOU_THRESHOLD)
    is_lvis = split.startswith("lvis")
    if is_lvis:
        instances = [ins for ins in instances if attach_lvis(ins, lvis.get(ins["image_id"], []), IOU_THRESHOLD)]
    records = annotate(cache, method, instances, vocab80, vocab102)
    metrics = summarize(records, n_gt, is_lvis)
    os.makedirs(out_dir, exist_ok=True)
    stopping_curve(records, n_gt, is_lvis).to_csv(os.path.join(out_dir, "stopping_curve.csv"), index=False)
    with open(os.path.join(out_dir, "instances.json"), "w") as f:
        json.dump(records, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"method": method, "split": split, "threshold": threshold, "n_images": len(image_ids),
                   "vlm": VLM_NAME, "crop_mode": CROP_MODE, "iou_threshold": IOU_THRESHOLD, "seed": seed,
                   "elapsed_sec": time.time() - start}, f, indent=4)
    print(method, split, threshold, {k: metrics[k] for k in ["n", "top1", "r", "n_h", "auroc_cum_conf"]})
    return records


def select_tau(curves):
    """
    検証用分割（lvis_dev）で階層的F値 hF が最大となる τ を選ぶ．

    Args:
        curves (pd.DataFrame): stopping_curve の戻り値（lvis_dev）．

    Returns:
        float: 選ばれた τ．
    """
    tau_rows = curves[curves["rule"] == "tau"]
    return float(tau_rows.loc[tau_rows["lvis_hF"].idxmax(), "value"])


def write_tables(results, seed):
    """
    集計表を outputs/ex007_coco_val_hierarchy_eval/ 直下に保存する．

    Args:
        results (dict): (手法, 分割, 閾値) → 記録．
        seed (int): シード．

    Returns:
        None
    """
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    rows = []
    for (method, split, threshold) in results:
        with open(os.path.join(target_dir(method, split, threshold, seed), "metrics.json")) as f:
            rows.append({"method": method, "split": split, "threshold": threshold} | json.load(f))
    pd.DataFrame(rows).to_csv(os.path.join(root, "table_conditions.csv"), index=False)

    pair_rows = []
    for split, ((m_a, t_a), (m_b, t_b)) in itertools.product(["dev500", "test500"], PAIRS):
        pair_rows.append({"split": split, "proposed": f"{m_a}_{t_a}", "hybrid": f"{m_b}_{t_b}"}
                         | depth1_agreement(results[(m_a, split, t_a)], results[(m_b, split, t_b)]))
    pd.DataFrame(pair_rows).to_csv(os.path.join(root, "table_pairs_agreement.csv"), index=False)

    stop_rows = []
    for method in ["proposed", "hybrid"]:
        for threshold in CONDITIONS[method]:
            curves = {s: pd.read_csv(os.path.join(target_dir(method, s, threshold, seed), "stopping_curve.csv"))
                      for s in SPLITS}
            tau = select_tau(curves["lvis_dev"])
            for split, value in itertools.product(["test500", "lvis_test"], [0.0, tau]):
                row = curves[split][(curves[split]["rule"] == "tau") & np.isclose(curves[split]["value"], value)].iloc[0]
                stop_rows.append({"method": method, "threshold": threshold, "split": split, "tau": value,
                                  "selected_tau": tau} | row.drop(["rule", "value"]).to_dict())
    pd.DataFrame(stop_rows).to_csv(os.path.join(root, "table_stopping.csv"), index=False)


def main():
    """全条件（手法 × 閾値 × 分割 × seed）の評価を実行し，集計表を作成する（完了済みの条件はスキップ）．"""
    coco = load_coco("instances")
    lvis = load_lvis()
    vocab80 = category_names(coco)
    vocab102 = vocab80 + list(EXTRA_TO_SYNSET)
    all_ids = sorted(coco.getImgIds())
    for seed in SEEDS:
        splits = make_splits(coco, set(lvis), seed)
        os.makedirs(os.path.join(OUTPUT_DIR, EXPERIMENT), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, EXPERIMENT, f"splits_seed{seed}.json"), "w") as f:
            json.dump(splits, f)
        detections = {"detr": detect_detr(coco, all_ids, lambda: load_detr(seed)),
                      "yolo": detect_yolo(coco, all_ids, lambda: load_yolo(seed))}
        torch.cuda.empty_cache()
        cache = FeatureCache(load_model(VLM_NAME, seed), coco)
        results = {}
        for method, split in itertools.product(CONDITIONS, SPLITS):
            if method.startswith("hybrid_switch") and split not in SWITCH_SPLITS:
                continue
            det = detections["detr" if method == "proposed" else "yolo"]
            for threshold in CONDITIONS[method]:
                results[(method, split, threshold)] = run_condition(
                    cache, method, split, threshold, splits[split], det, lvis, vocab80, vocab102, seed)
        write_tables(results, seed)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
