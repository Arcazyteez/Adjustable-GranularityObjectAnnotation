"""
ex008の実行スクリプト：SigLIPとシグモイド，YOLOとVLMのスコアの比較，WordNetの同じ深さの語彙による深さ0（order_009）．

学習は行わない．ex007と同じ分割・検出結果・評価指標を用い，以下を比較する．

1. VLMと正規化関数（SCORERS）
   - CLIP-ViT-B32_softmax（現在の構成，ex007と同じ）
   - SigLIP-ViT-B16_softmax（モデルのみ変更）
   - SigLIP-ViT-B16_sigmoid（モデルと正規化関数を変更）
2. 深さ0の決め方（METHODS）
   - proposed:           DETR + VLMでCOCOの80語から選ぶ（現在の提案手法）
   - proposed_wn{k}:     DETR + VLMでWordNetの同じ深さの語彙から選ぶ
   - hybrid:             YOLO11xの語をそのまま用いる
   - hybrid_switch_c0.5: VLMの102語の最大スコアがCOCO外の語かつ0.5以上ならVLMの語（現在のハイブリッド構成，論文の式(9)）
   - hybrid_cmp:         VLMの80語の最大スコアがYOLOの検出スコアより高ければVLMの語
   - hybrid_cmp_wn{k}:   VLMのWordNetの同じ深さの語彙の最大スコアがYOLOの検出スコアより高ければVLMの語
3. 評価
   - dev500，test500，lvis_dev，lvis_test: ex007と同じ指標（深さ0のTop-1，誤りの伝播とAUROC，LVISの hP/hR/hF，停止規則）
   - lvis_dev_noncoco，lvis_test_noncoco: COCO外の物体（LVIS）に対し，深さ0の語がLVISのSynsetの上位語（自身を含む）か
   - ex004の実画像（COCO外の物体）の定性評価

出力: outputs/ex008_coco_val_siglip_wordnet_root/{VLM}_{正規化関数}/{手法}/{分割}_th{閾値}/{seed}/
    config.json, instances.json, metrics.json, (stopping_curve.csv)
    および outputs/ex008_coco_val_siglip_wordnet_root/ 直下の集計表（table_*.csv）
"""
import itertools
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_coco, load_image, make_crop  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402
from data import (ancestors, build_noncoco_instances, build_wordnet_root_vocab, coco_extra_vocab,  # noqa: E402
                  coco_vocab, import_experiment_module, is_non_coco_synset, load_ex007)
from model import (VLM_NORMALIZATIONS, hierarchical_chain, load_detr, load_model, load_yolo,  # noqa: E402
                   logits_from_features, normalize)

ex007_data, _, ex007 = load_ex007()

# ===== 実験条件（order_009） =====
EXPERIMENT = "ex008_coco_val_siglip_wordnet_root"
SEEDS = [42]
CROP_MODE = "pad"
IOU_THRESHOLD = 0.5
VLMS = ["CLIP-ViT-B32", "SigLIP-ViT-B16"]
VOCAB_KS = [1, 2, 3]   # WordNetの同じ深さの語彙を作る際の祖先までの段数 k
MAIN_K = 2             # 主要な結果に用いる k
MIN_COUNT = 1          # SemCorの出現頻度の下限
SWITCH_GAMMA = 0.5     # hybrid_switch_c0.5 の γ
# 手法 → 検出スコアの閾値（proposed* はDETRの θ，hybrid* はYOLO11xの閾値）
METHODS = ({"proposed": [0.5, 0.97]} | {f"proposed_wn{k}": [0.5, 0.97] for k in VOCAB_KS}
           | {"hybrid": [0.25, 0.7], "hybrid_switch_c0.5": [0.25, 0.7], "hybrid_cmp": [0.25, 0.7]}
           | {f"hybrid_cmp_wn{k}": [0.25, 0.7] for k in VOCAB_KS})
COCO_SPLITS = ["dev500", "test500"]
LVIS_SPLITS = ["lvis_dev", "lvis_test"]
NONCOCO_SPLITS = {"lvis_dev_noncoco": "lvis_dev", "lvis_test_noncoco": "lvis_test"}  # 分割名 → 画像の分割
# LVISの細かい粒度の評価は k = MAIN_K と比較の基準となる手法のみ行う
LVIS_METHODS = ["proposed", f"proposed_wn{MAIN_K}", "hybrid", "hybrid_cmp", f"hybrid_cmp_wn{MAIN_K}"]
QUALITATIVE_METHODS = ["hybrid_switch_c0.5", "hybrid_cmp", f"hybrid_cmp_wn{MAIN_K}"]
QUALITATIVE_CONF = 0.25
# シグモイドの累積確信度は小さくなるため，ex007より小さい τ も探索する
TAUS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
ex007.TAUS = TAUS
COCO_SYNSET_NAMES = set(COCO_TO_SYNSET.values())


def scorers():
    """
    比較するVLMと正規化関数の組を返す．

    Returns:
        list[tuple[str, str]]: (VLMの名前, 正規化関数) のリスト．
    """
    return [(vlm, norm) for vlm in VLMS for norm in VLM_NORMALIZATIONS[vlm]]


def target_dir(scorer, method, split, threshold, seed):
    """
    1条件の結果の保存先を返す．

    Args:
        scorer (str): "{VLM}_{正規化関数}"．
        method (str): 手法名．
        split (str): 分割名．
        threshold (float): 検出スコアの閾値．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, scorer, method, f"{split}_th{threshold}", str(seed))


def method_vocab(method):
    """
    手法の深さ0の候補語彙を返す．

    Args:
        method (str): 手法名．

    Returns:
        list[dict] | None: {"word"，"synset"，"coco"} のリスト（hybrid は None）．
    """
    if method == "hybrid":
        return None
    if method.startswith("hybrid_switch"):
        return coco_extra_vocab()
    if "_wn" in method:
        return build_wordnet_root_vocab(int(method.split("_wn")[1]), MIN_COUNT)
    return coco_vocab()


def decide_root(method, ins, probs, vocab):
    """
    深さ0の単語，Synset，確信度と，その決定に用いたモデルを返す．

    Args:
        method (str): 手法名．
        ins (dict): インスタンス（hybrid* では "yolo_word"，"det_score" を持つ）．
        probs (np.ndarray | None): 候補語彙に対するVLMの確信度（長さV）．
        vocab (list[dict] | None): 候補語彙．

    Returns:
        dict: root_word，root_synset，root_conf，root_source（"vlm" または "yolo"），switched（YOLOと異なる語か），
            vlm_word，vlm_conf（VLMの最大確信度），vlm_conf_yolo_word（YOLOの語に対するVLMの確信度）．
    """
    best = int(np.argmax(probs)) if probs is not None else None
    info = {"vlm_word": vocab[best]["word"] if best is not None else None,
            "vlm_conf": float(probs[best]) if best is not None else None, "vlm_conf_yolo_word": None}
    vlm_root = {"root_word": info["vlm_word"], "root_synset": vocab[best]["synset"] if best is not None else None,
                "root_conf": info["vlm_conf"], "root_source": "vlm"}
    if method.startswith("proposed"):
        return vlm_root | info | {"switched": False}
    yolo_root = {"root_word": ins["yolo_word"], "root_synset": COCO_TO_SYNSET[ins["yolo_word"]],
                 "root_conf": float(ins["det_score"]), "root_source": "yolo"}
    if best is not None:
        words = [v["word"] for v in vocab]
        info["vlm_conf_yolo_word"] = float(probs[words.index(ins["yolo_word"])])
    if method == "hybrid":
        use_vlm = False
    elif method.startswith("hybrid_switch"):
        use_vlm = vocab[best]["coco"] is None and probs[best] >= SWITCH_GAMMA
    else:  # hybrid_cmp*: VLMの最大確信度がYOLOの検出スコアより高ければVLMの語
        use_vlm = probs[best] > ins["det_score"]
    root = vlm_root if use_vlm else yolo_root
    return root | info | {"switched": bool(use_vlm and root["root_synset"] != yolo_root["root_synset"])}


class FeatureCache:
    """1つのVLMについて，クロップの画像特徴と，(クロップ，深さ0のSynset) ごとの下位語の展開結果を保持する．"""

    def __init__(self, vlm, coco):
        """
        Args:
            vlm (CLIPModel | SigLIPModel): VLM．
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
        for start in tqdm(range(0, len(todo), batch_size), desc=f"{self.vlm.name} features", leave=False):
            chunk = todo[start:start + batch_size]
            crops = [make_crop(load_image(self.coco, k[0]), ins["box"], CROP_MODE) for k, ins in chunk]
            for (k, _), feat in zip(chunk, self.vlm.image_features(crops)):
                self.features[k] = feat

    def probs(self, instances, vocab, normalization):
        """
        全インスタンスの深さ0の候補語彙上の確信度を計算する．

        Args:
            instances (list[dict]): インスタンス（長さN）．
            vocab (list[dict]): 候補語彙（長さV）．
            normalization (str): 正規化関数．

        Returns:
            np.ndarray: 確信度（N×V）．
        """
        if not instances:
            return np.zeros((0, len(vocab)))
        feats = torch.stack([self.features[self.key(i)] for i in instances])
        logits = logits_from_features(self.vlm, feats, [v["word"] for v in vocab])
        return normalize(logits, normalization).cpu().numpy()

    def chain(self, ins, root, normalization):
        """
        深さ0から下位語を最後まで展開し，指定した正規化関数の確信度を "conf" とした結果を返す．

        Args:
            ins (dict): インスタンス．
            root (dict): decide_root の戻り値．
            normalization (str): 正規化関数．

        Returns:
            list[dict]: 各深さの {"depth", "word", "synset", "conf", "n_candidates"}．
        """
        k = (self.key(ins), root["root_synset"])
        if k not in self.chains:
            self.chains[k] = hierarchical_chain(self.vlm, self.features[self.key(ins)], root["root_word"],
                                                root["root_synset"])
        chain = [{"depth": s["depth"], "word": s["word"], "synset": s["synset"], "conf": s[f"conf_{normalization}"],
                  "n_candidates": s["n_candidates"]} for s in self.chains[k]]
        chain[0]["word"], chain[0]["conf"] = root["root_word"], root["root_conf"]
        return chain


def annotate(cache, method, normalization, instances):
    """
    全インスタンスの深さ0を決定し，正誤によらず下位語を最後まで展開する．

    Args:
        cache (FeatureCache): 画像特徴のキャッシュ．
        method (str): 手法名．
        normalization (str): 正規化関数．
        instances (list[dict]): インスタンス．

    Returns:
        list[dict]: インスタンスに decide_root の戻り値，"top1_correct"，"chain"（COCO外の物体では "root_on_path"）を追記したもの．
    """
    cache.prepare(instances)
    vocab = method_vocab(method)
    probs = cache.probs(instances, vocab, normalization) if vocab else [None] * len(instances)
    records = []
    for ins, p in zip(tqdm(instances, desc=method, leave=False), probs):
        root = decide_root(method, ins, p, vocab)
        record = ins | root | {"chain": cache.chain(ins, root, normalization)}
        if ins["gt_name"] is not None:
            record["top1_correct"] = root["root_synset"] == COCO_TO_SYNSET[ins["gt_name"]]
        else:
            record["root_on_path"] = (root["root_synset"] == ins["lvis_synset"]
                                      or root["root_synset"] in ancestors(ins["lvis_synset"]))
        records.append(record)
    return records


# ===================== 評価指標 =====================

def switch_metrics(records):
    """
    深さ0の切替（YOLOと異なる語の採用）と，COCO外の語の出力に関する指標を計算する．

    Args:
        records (list[dict]): COCOのGTと対応付けた記録．

    Returns:
        dict: n_switched，n_switch_fix（YOLOが不正解でVLMが正解），n_switch_break（YOLOが正解でVLMが不正解），
            n_root_non_coco（深さ0がCOCO外の語の数），mean_root_conf_correct / wrong．
    """
    switched = [r for r in records if r["switched"]]
    yolo_correct = [r.get("yolo_word") == r["gt_name"] for r in switched]
    correct = np.array([r["top1_correct"] for r in records])
    root_conf = np.array([r["root_conf"] for r in records])
    return {"n_switched": len(switched),
            "n_switch_fix": int(sum(r["top1_correct"] and not y for r, y in zip(switched, yolo_correct))),
            "n_switch_break": int(sum(y and not r["top1_correct"] for r, y in zip(switched, yolo_correct))),
            "n_root_non_coco": int(sum(r["root_synset"] not in COCO_SYNSET_NAMES for r in records)),
            "mean_root_conf_correct": float(root_conf[correct].mean()) if correct.any() else float("nan"),
            "mean_root_conf_wrong": float(root_conf[~correct].mean()) if (~correct).any() else float("nan")}


def noncoco_metrics(records):
    """
    COCO外の物体に対する深さ0と下位語の展開の指標を計算する．

    Args:
        records (list[dict]): COCO外の物体の記録．

    Returns:
        dict: n，root_on_path（深さ0がLVISのSynsetの上位語または自身である割合），root_is_coco（深さ0がCOCOの語の割合），
            reach_lvis（展開した経路がLVISのSynsetに到達した割合），n_switched，n_categories（LVISのカテゴリ数）．
    """
    if not records:
        return {"n": 0}
    on_path = np.array([r["root_on_path"] for r in records])
    return {"n": len(records), "root_on_path": float(on_path.mean()), "n_root_on_path": int(on_path.sum()),
            "root_is_coco": float(np.mean([r["root_synset"] in COCO_SYNSET_NAMES for r in records])),
            "reach_lvis": float(np.mean([r["lvis_synset"] in [s["synset"] for s in r["chain"]] for r in records])),
            "n_switched": int(sum(r["switched"] for r in records)),
            "n_categories": len({r["lvis_synset"] for r in records})}


def save_condition(out_dir, records, metrics, config, curve=None):
    """
    1条件の結果を保存する．

    Args:
        out_dir (str): 保存先．
        records (list[dict]): 記録．
        metrics (dict): 指標．
        config (dict): 実験条件．
        curve (pd.DataFrame | None): 停止規則の曲線．

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    if curve is not None:
        curve.to_csv(os.path.join(out_dir, "stopping_curve.csv"), index=False)
    for name, obj in [("instances.json", records), ("metrics.json", metrics), ("config.json", config)]:
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, indent=None if name == "instances.json" else 4)


def run_condition(cache, scorer, method, split, threshold, ctx, seed):
    """
    1条件（VLMと正規化関数 × 手法 × 分割 × 閾値）を評価して保存する（完了済みならスキップする）．

    Args:
        cache (FeatureCache): 画像特徴のキャッシュ．
        scorer (tuple[str, str]): (VLMの名前, 正規化関数)．
        method (str): 手法名．
        split (str): 分割名．
        threshold (float): 検出スコアの閾値．
        ctx (dict): "coco"，"splits"，"detections"，"lvis" を持つ辞書．
        seed (int): シード．

    Returns:
        None
    """
    name = "_".join(scorer)
    out_dir = target_dir(name, method, split, threshold, seed)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        return
    start = time.time()
    det = ctx["detections"]["detr" if method.startswith("proposed") else "yolo"]
    curve = None
    if split in NONCOCO_SPLITS:
        image_ids = ctx["splits"][NONCOCO_SPLITS[split]]
        instances = build_noncoco_instances(ctx["coco"], image_ids, det, threshold, ctx["lvis"], IOU_THRESHOLD)
        records = annotate(cache, method, scorer[1], instances)
        metrics = noncoco_metrics(records)
    else:
        image_ids = ctx["splits"][split]
        instances, n_gt = ex007_data.build_instances(ctx["coco"], image_ids, det, threshold, IOU_THRESHOLD)
        is_lvis = split in LVIS_SPLITS
        if is_lvis:
            instances = [i for i in instances
                         if ex007_data.attach_lvis(i, ctx["lvis"].get(i["image_id"], []), IOU_THRESHOLD)]
        records = annotate(cache, method, scorer[1], instances)
        metrics = ex007.summarize(records, n_gt, is_lvis) | switch_metrics(records)
        curve = ex007.stopping_curve(records, n_gt, is_lvis)
    config = {"vlm": scorer[0], "normalization": scorer[1], "method": method, "split": split, "threshold": threshold,
              "n_images": len(image_ids), "crop_mode": CROP_MODE, "iou_threshold": IOU_THRESHOLD, "seed": seed,
              "vocab_size": len(method_vocab(method) or []), "min_count": MIN_COUNT, "switch_gamma": SWITCH_GAMMA,
              "elapsed_sec": time.time() - start}
    save_condition(out_dir, records, metrics, config, curve)
    print(name, method, split, threshold, {k: metrics.get(k) for k in ["n", "top1", "n_switched", "root_on_path"]})


def condition_list():
    """
    実行する（手法，分割，閾値）の組を返す．

    Returns:
        list[tuple[str, str, float]]: 条件のリスト．
    """
    conditions = []
    for method, thresholds in METHODS.items():
        splits = COCO_SPLITS + list(NONCOCO_SPLITS) + (LVIS_SPLITS if method in LVIS_METHODS else [])
        conditions += [(method, split, t) for split, t in itertools.product(splits, thresholds)]
    return conditions


# ===================== 定性評価 =====================

def draw_regions(ax, image, regions, title):
    """
    画像に検出ボックスと各段階の推定単語・確信度を描画する．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        image (PIL.Image.Image): 入力画像．
        regions (list[dict]): 領域ごとの {"box", "root_source", "chain"}．
        title (str): タイトル．

    Returns:
        None
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region["box"]
        color = plt.cm.tab10.colors[i % 10]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
        head = "VLM: " if region["root_source"] == "vlm" else "YOLO: "
        lines = [(head if s["depth"] == 0 else "> ") + f"{s['word']} ({s['conf']:.2f})" for s in region["chain"]]
        ax.text(x0, y0, "\n".join(lines), fontsize=6, color="white", va="top",
                bbox=dict(facecolor=color, alpha=0.75, pad=1))


def run_qualitative(vlms, seed):
    """
    ex004の実画像（COCO外の物体）で，YOLO11x（閾値0.25）を用いるハイブリッド構成の定性評価を行う．

    Args:
        vlms (dict[str, CLIPModel | SigLIPModel]): VLMの名前 → VLM．
        seed (int): シード．

    Returns:
        None
    """
    todo = [(s, m) for s, m in itertools.product(scorers(), QUALITATIVE_METHODS)
            if not os.path.exists(os.path.join(target_dir("_".join(s), m, "open_vocab", QUALITATIVE_CONF, seed),
                                               "annotations.json"))]
    if not todo:
        return
    samples = import_experiment_module("ex006_coco_val500_yolo_clip_hybrid", "data").load_qualitative_samples()["open_vocab"]
    detector = load_yolo(seed)
    detections = {s["name"]: detector.detect(s["image"]) for s in samples}
    del detector
    torch.cuda.empty_cache()
    for (vlm_name, norm), method in todo:
        vlm, vocab = vlms[vlm_name], method_vocab(method)
        results = {}
        for s in samples:
            det = detections[s["name"]]
            if not det["boxes"]:
                results[s["name"]] = []
                continue
            feats = vlm.image_features([make_crop(s["image"], b, CROP_MODE) for b in det["boxes"]])
            probs = normalize(logits_from_features(vlm, feats, [v["word"] for v in vocab]), norm).cpu().numpy()
            regions = []
            for box, score, top5, feat, p in zip(det["boxes"], det["scores"], det["top5"], feats, probs):
                ins = {"yolo_word": top5[0], "det_score": score}
                root = decide_root(method, ins, p, vocab)
                chain = [{"depth": c["depth"], "word": c["word"], "synset": c["synset"],
                          "conf": c[f"conf_{norm}"], "n_candidates": c["n_candidates"]}
                         for c in hierarchical_chain(vlm, feat, root["root_word"], root["root_synset"])]
                chain[0]["conf"] = root["root_conf"]
                regions.append({"box": box, "det_score": score} | root | {"chain": chain})
            results[s["name"]] = regions
        out_dir = target_dir(f"{vlm_name}_{norm}", method, "open_vocab", QUALITATIVE_CONF, seed)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "annotations.json"), "w") as f:
            json.dump({s["name"]: {"meta": s["meta"], "regions": results[s["name"]]} for s in samples}, f, indent=1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax in axes.flat:
            ax.axis("off")
        for ax, s in zip(axes.flat, samples):
            draw_regions(ax, s["image"], results[s["name"]], f"{s['name']} ({len(results[s['name']])} boxes)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grid.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        for name, regions in results.items():
            print(vlm_name, norm, method, name,
                  [" > ".join(f"{c['word']}({c['conf']:.2f})" for c in r["chain"][:3]) for r in regions[:3]])


# ===================== 集計 =====================

def read_metrics(scorer, method, split, threshold, seed):
    """
    1条件の指標を読み込む．

    Args:
        scorer (str): "{VLM}_{正規化関数}"．
        method (str): 手法名．
        split (str): 分割名．
        threshold (float): 閾値．
        seed (int): シード．

    Returns:
        dict: metrics.json の内容．
    """
    with open(os.path.join(target_dir(scorer, method, split, threshold, seed), "metrics.json")) as f:
        return json.load(f)


def vocab_table(ctx):
    """
    WordNetの同じ深さの語彙の大きさと，COCO外の語・LVISのCOCO外のカテゴリをどれだけ含むかを集計する．

    Args:
        ctx (dict): "lvis"，"splits" を持つ辞書．

    Returns:
        pd.DataFrame: k ごとの語彙の統計．
    """
    extra = {v["synset"] for v in coco_extra_vocab() if v["coco"] is None}
    lvis_ids = set(ctx["splits"]["lvis_test"])
    noncoco = {a["synset"] for i, anns in ctx["lvis"].items() if i in lvis_ids for a in anns}
    noncoco = {s for s in noncoco if is_non_coco_synset(s)}
    rows = []
    for k in VOCAB_KS:
        vocab = build_wordnet_root_vocab(k, MIN_COUNT)
        synsets = {v["synset"] for v in vocab}
        covered = [s for s in noncoco if s in synsets or ancestors(s) & synsets]
        rows.append({"k": k, "vocab_size": len(vocab), "n_non_coco_words": sum(v["coco"] is None for v in vocab),
                     "extra22_in_vocab": sum(s in synsets for s in extra),
                     "lvis_test_noncoco_categories": len(noncoco), "lvis_categories_with_vocab_ancestor": len(covered)})
    return pd.DataFrame(rows)


def depth1_auroc(records):
    """
    深さ0が正解のLVISのインスタンスについて，深さ1の確信度で深さ1の正誤（LVISの経路上か）を区別できるかを評価する．

    Args:
        records (list[dict]): "gt_path" を持つLVISの分割の記録．

    Returns:
        dict: n，depth1_correct（深さ1の正解率），mean_conf_correct / wrong，auroc（深さ1の確信度），
            auroc_single_excluded（候補語彙が1語の段階を除いたAUROC）．
    """
    good = [r for r in records if r["top1_correct"] and len(r["chain"]) > 1]
    labels = np.array([r["chain"][1]["synset"] in r["gt_path"] for r in good])
    conf = np.array([r["chain"][1]["conf"] for r in good])
    multi = np.array([r["chain"][1]["n_candidates"] > 1 for r in good])
    def auroc(mask):
        ok = mask & np.isfinite(conf)
        return ex007.roc_auc_score(labels[ok], conf[ok]) if 0 < labels[ok].sum() < ok.sum() else float("nan")
    return {"n": len(good), "depth1_correct": float(labels.mean()), "mean_conf_correct": float(conf[labels].mean()),
            "mean_conf_wrong": float(conf[~labels].mean()), "auroc": auroc(np.ones(len(good), dtype=bool)),
            "auroc_single_excluded": auroc(multi)}


def write_tables(ctx, seed):
    """
    集計表を outputs/ex008_coco_val_siglip_wordnet_root/ 直下に保存する．

    Args:
        ctx (dict): 実行時の共通データ．
        seed (int): シード．

    Returns:
        None
    """
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    rows, noncoco_rows, stop_rows = [], [], []
    for (vlm, norm), (method, split, threshold) in itertools.product(scorers(), condition_list()):
        name = f"{vlm}_{norm}"
        row = {"scorer": name, "method": method, "split": split, "threshold": threshold}
        (noncoco_rows if split in NONCOCO_SPLITS else rows).append(row | read_metrics(name, method, split, threshold, seed))
    pd.DataFrame(rows).to_csv(os.path.join(root, "table_conditions.csv"), index=False)
    pd.DataFrame(noncoco_rows).to_csv(os.path.join(root, "table_noncoco.csv"), index=False)
    for (vlm, norm), method in itertools.product(scorers(), LVIS_METHODS):
        name = f"{vlm}_{norm}"
        for threshold in METHODS[method]:
            curves = {s: pd.read_csv(os.path.join(target_dir(name, method, s, threshold, seed), "stopping_curve.csv"))
                      for s in LVIS_SPLITS + ["test500"]}
            tau = ex007.select_tau(curves["lvis_dev"])
            for split, value in itertools.product(["test500", "lvis_test"], [0.0, tau]):
                curve = curves[split]
                row = curve[(curve["rule"] == "tau") & np.isclose(curve["value"], value)].iloc[0]
                stop_rows.append({"scorer": name, "method": method, "threshold": threshold, "split": split,
                                  "tau": value, "selected_tau": tau} | row.drop(["rule", "value"]).to_dict())
    pd.DataFrame(stop_rows).to_csv(os.path.join(root, "table_stopping.csv"), index=False)
    vocab_table(ctx).to_csv(os.path.join(root, "table_vocab.csv"), index=False)
    depth1_rows = []
    for (vlm, norm), method in itertools.product(scorers(), LVIS_METHODS):
        for threshold, split in itertools.product(METHODS[method], LVIS_SPLITS):
            with open(os.path.join(target_dir(f"{vlm}_{norm}", method, split, threshold, seed), "instances.json")) as f:
                depth1_rows.append({"scorer": f"{vlm}_{norm}", "method": method, "threshold": threshold, "split": split}
                                   | depth1_auroc(json.load(f)))
    pd.DataFrame(depth1_rows).to_csv(os.path.join(root, "table_depth1_auroc.csv"), index=False)


def main():
    """全条件（VLMと正規化関数 × 手法 × 分割 × 閾値 × seed）の評価と定性評価を実行し，集計表を作成する．"""
    coco = load_coco("instances")
    lvis = ex007_data.load_lvis()
    all_ids = sorted(coco.getImgIds())
    for k in VOCAB_KS:
        print(f"WordNet root vocab k={k}: {len(build_wordnet_root_vocab(k, MIN_COUNT))} words")
    for seed in SEEDS:
        ctx = {"coco": coco, "lvis": lvis, "splits": ex007_data.make_splits(coco, set(lvis), seed),
               "detections": {"detr": ex007_data.detect_detr(coco, all_ids, lambda: load_detr(seed)),
                              "yolo": ex007_data.detect_yolo(coco, all_ids, lambda: load_yolo(seed))}}
        os.makedirs(os.path.join(OUTPUT_DIR, EXPERIMENT), exist_ok=True)
        vlms = {}
        for vlm_name in VLMS:
            vlms[vlm_name] = load_model(vlm_name, seed)
            cache = FeatureCache(vlms[vlm_name], coco)
            for norm, (method, split, threshold) in itertools.product(VLM_NORMALIZATIONS[vlm_name], condition_list()):
                run_condition(cache, (vlm_name, norm), method, split, threshold, ctx, seed)
            del cache
            torch.cuda.empty_cache()
        run_qualitative(vlms, seed)
        write_tables(ctx, seed)


if __name__ == "__main__":
    main()
