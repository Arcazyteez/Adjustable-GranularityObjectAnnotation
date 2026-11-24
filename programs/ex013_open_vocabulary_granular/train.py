"""
ex013の実行スクリプト：再学習なしでCOCO外の語彙の物体を検出し，粒度可変のアノテーションを行う（order_015）．

学習は行わない．前の構成に改善を1つずつ加えて，LVISのテスト用の画像で評価する．
- ex012:      YOLO11xのボックスに，WordNetの上位語6語から下位語を最後までたどる（進捗報告2026Bのアルゴリズム，θ = -1）
- A_root:     YOLO11xのボックスの深さ0をYOLO11xのクラスとし，深さ1以降をFとWordNetで細分化する
- A_add:      さらに，YOLO11xのボックスと重ならないOWLv2の物体らしさの上位 K_ADD 件の領域を加え，上位語6語から命名する
- B_prune:    さらに，WordNetの下位語を絞り込む
- BD_stop:    さらに，候補の中でのソフトマックスが τ 以上の間だけ細分化する（τ は検証用で選ぶ）
- BDC_look:   さらに，子と孫の類似度の最大値で進む枝を選ぶ（2段の先読み）
- BDE_strong: BD_stop に，SigLIP SO400M，上位語を含むプロンプト，1.5倍に広げた切り抜きを加える
- BDCE_all:   全てを加える

評価: 出力ボックスとLVISのGTをIoU 0.5以上で1対1に対応付け，出力した語とGTのSynsetのそれぞれの上位語（自身を含み，
深さ D_MIN 以上で，GENERIC を除く）の集合から，階層的適合率 hP（対応した出力のみ），再現率 hR（全てのGT），F値 hF を求める．
LVISの注釈は全ての物体を網羅していないため，GTと対応しない出力は hP に含めず，加えた領域の数と適合率を別に示す．

出力: outputs/ex013_open_vocabulary_granular/{構成}/{分割}/{seed}/metrics.json，config.json，(tau_curve.csv)
    qualitative/{構成}.json と grid_{構成}.png，および直下の table_results.csv
"""
import functools
import json
import os
import sys
import time

# 決定論的なCUDAの行列計算に必要な設定（torch の読み込み前に設定する）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet as wn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.metrics import box_iou  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402
from data import SPLITS, ancestors, load_context, owlv2_proposals  # noqa: E402
from model import WORDNET_TOPS, Embedder, OWLv2Objectness, context_crop, descend  # noqa: E402

# ===== 実験条件（order_015） =====
EXPERIMENT = "ex013_open_vocabulary_granular"
SEEDS = [42]
YOLO_SCORE = 0.25
K_ADD = 10
IOU_THRESHOLD = 0.5
D_MIN = 3
GENERIC = {"whole.n.02"}
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BASE = {"root_yolo": False, "add": 0, "pruned": False, "lookahead": False, "embedder": "SigLIP-B16", "crop": 1.0,
        "mode": "plain", "stop": False}
CONFIGS = {
    "ex012": BASE,
    "A_root": BASE | {"root_yolo": True},
    "A_add": BASE | {"root_yolo": True, "add": K_ADD},
    "B_prune": BASE | {"root_yolo": True, "add": K_ADD, "pruned": True},
    "BD_stop": BASE | {"root_yolo": True, "add": K_ADD, "pruned": True, "stop": True},
    "BDC_look": BASE | {"root_yolo": True, "add": K_ADD, "pruned": True, "stop": True, "lookahead": True},
    "BDE_strong": BASE | {"root_yolo": True, "add": K_ADD, "pruned": True, "stop": True, "embedder": "SigLIP-SO400M",
                          "crop": 1.5, "mode": "hier"},
    "BDCE_all": BASE | {"root_yolo": True, "add": K_ADD, "pruned": True, "stop": True, "lookahead": True,
                        "embedder": "SigLIP-SO400M", "crop": 1.5, "mode": "hier"},
}
QUALITATIVE_CONFIGS = ["ex012", "BDCE_all"]
N_QUALITATIVE = 6


def target_dir(config, split, seed):
    """
    1構成の保存先を返す．

    Args:
        config (str): 構成名．
        split (str): 分割名．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, config, split, str(seed))


# ===================== 出力ボックス =====================

def candidate_boxes(ctx, proposals, image_id):
    """
    1枚の画像のYOLO11xのボックスと，YOLO11xと重ならないOWLv2の領域（物体らしさの上位 K_ADD 件）を返す．

    Args:
        ctx (dict): load_context の戻り値．
        proposals (dict): owlv2_proposals の戻り値．
        image_id (str): 画像ID．

    Returns:
        list[dict]: {"box"，"score"，"source"（"yolo" または "owlv2"），("label")}（YOLO11xが先）．
    """
    d = ctx["yolo"][image_id]
    out = [{"box": b, "score": s, "source": "yolo", "label": l}
           for b, s, l in zip(d["boxes"], d["scores"], d["labels"]) if s >= YOLO_SCORE]
    base = np.asarray([b["box"] for b in out], dtype=np.float32).reshape(-1, 4)
    cand = proposals[image_id]
    boxes = np.asarray(cand["boxes"], dtype=np.float32).reshape(-1, 4)
    overlap = box_iou(boxes, base).max(axis=1) if len(base) and len(boxes) else np.zeros(len(boxes))
    extra = [k for k in range(len(boxes)) if overlap[k] < IOU_THRESHOLD][:K_ADD]
    return out + [{"box": cand["boxes"][k], "score": cand["scores"][k], "source": "owlv2"} for k in extra]


def prepare(ctx, proposals, split):
    """
    分割の画像ごとに，出力ボックスの候補，GT，両者のIoUを前計算する．

    Args:
        ctx (dict): load_context の戻り値．
        proposals (dict): owlv2_proposals の戻り値．
        split (str): 分割名．

    Returns:
        list[dict]: 画像ごとの {"image_id"，"boxes"，"gts"，"iou"（ボックス×GT）}．
    """
    images = []
    for i in ctx["splits"][split]:
        boxes = candidate_boxes(ctx, proposals, str(i))
        gts = ctx["lvis"][i]
        b = np.asarray([x["box"] for x in boxes], dtype=np.float32).reshape(-1, 4)
        g = np.asarray([x["box"] for x in gts], dtype=np.float32).reshape(-1, 4)
        iou = box_iou(b, g) if len(b) and len(g) else np.zeros((len(b), len(g)))
        images.append({"image_id": str(i), "boxes": boxes, "gts": gts, "iou": iou})
    return images


# ===================== 特徴と下位語の展開 =====================

def features(embedder, coco, images, scale):
    """
    全ての出力ボックスの候補の切り抜き画像の特徴ベクトルを計算する．

    Args:
        embedder (Embedder): F．
        coco (COCO): COCOオブジェクト．
        images (list[dict]): prepare の戻り値．
        scale (float): 切り抜きの拡大率．

    Returns:
        list[torch.Tensor]: 画像ごとの特徴ベクトル（ボックス数×D）．
    """
    out = []
    for img in tqdm(images, desc=f"{embedder.name} x{scale}", leave=False):
        image = load_image(coco, int(img["image_id"]))
        crops = [context_crop(image, b["box"], scale) for b in img["boxes"]]
        out.append(embedder.image_features(crops) if crops else torch.zeros(0))
    return out


def chains(embedder, images, feats, cfg):
    """
    全ての出力ボックスの候補について，下位語を葉まで展開する．

    Args:
        embedder (Embedder): F．
        images (list[dict]): prepare の戻り値．
        feats (list[torch.Tensor]): features の戻り値．
        cfg (dict): 構成．

    Returns:
        list[list[list[dict]]]: 画像ごと・ボックスごとの descend の戻り値．
    """
    out = []
    for img, feat in zip(tqdm(images, desc="descend", leave=False), feats):
        per = []
        for b, f in zip(img["boxes"], feat):
            root = COCO_TO_SYNSET[b["label"]] if cfg["root_yolo"] and b["source"] == "yolo" else None
            per.append(descend(embedder, f, WORDNET_TOPS, cfg["pruned"], cfg["lookahead"], cfg["mode"], root))
        out.append(per)
    return out


def output_synset(chain, stop, tau):
    """
    停止の基準に従って出力するSynsetを返す．

    Args:
        chain (list[dict]): descend の戻り値．
        stop (bool): 停止を行うか（False は葉まで出力する）．
        tau (float): ソフトマックスの閾値．

    Returns:
        tuple[str, int]: 出力するSynset名と深さ．
    """
    if not stop:
        return chain[-1]["synset"], len(chain) - 1
    depth = 0
    for i in range(1, len(chain)):
        step = chain[i]
        # 候補が1つの段階はソフトマックスが常に1となるため，親より類似度が高い場合のみ進む
        if step["soft"] < tau or (step["n_candidates"] == 1 and step["cos"] < chain[i - 1]["cos"]):
            break
        depth = i
    return chain[depth]["synset"], depth


# ===================== 評価指標 =====================

@functools.lru_cache(maxsize=None)
def evaluated_set(synset_name):
    """
    評価に用いる，Synset自身と上位語のうち深さ D_MIN 以上で GENERIC を除いたものの集合を返す．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    return frozenset(s for s in ancestors(synset_name) if wn.synset(s).min_depth() >= D_MIN and s not in GENERIC)


def evaluate(images, all_chains, cfg, tau, keep=None):
    """
    1つの構成と τ で出力を決め，LVISのGTに対する指標を計算する．

    Args:
        images (list[dict]): prepare の戻り値．
        all_chains (list): chains の戻り値．
        cfg (dict): 構成．
        tau (float): ソフトマックスの閾値．
        keep (set[str] | None): 出力を記録する画像ID．

    Returns:
        dict: hP，hR，hF，{coco, noncoco}_{n, detected, named, fine, hR}，mean_depth，added_per_image，added_precision，
            （"outputs"：keep の画像の出力）．
    """
    inter = pred = gt_total = 0
    stats = {k: {"n": 0, "detected": 0, "named": 0, "fine": 0, "inter": 0, "size": 0} for k in ["coco", "noncoco"]}
    n_added = n_added_ok = 0
    depths, kept = [], {}
    for img, ch in zip(images, all_chains):
        outputs = []
        n_owl = 0
        for j, (b, c) in enumerate(zip(img["boxes"], ch)):
            if b["source"] == "owlv2":
                n_owl += 1
                if n_owl > cfg["add"]:
                    continue
            synset, depth = output_synset(c, cfg["stop"], tau)
            outputs.append((j, synset, depth))
        if keep is not None and img["image_id"] in keep:
            kept[img["image_id"]] = [{"box": img["boxes"][j]["box"], "source": img["boxes"][j]["source"],
                                      "words": [s["word"] for s in ch[j][:d + 1]]} for j, _, d in outputs]
        for g in img["gts"]:
            size = len(evaluated_set(g["synset"]))
            gt_total += size
            stats[g["kind"]]["n"] += 1
            stats[g["kind"]]["size"] += size
        rows = [j for j, _, _ in outputs]
        matched = {}
        if rows and len(img["gts"]):
            iou = img["iou"][rows]
            r, c = linear_sum_assignment(-iou)
            matched = {int(a): int(b) for a, b in zip(r, c) if iou[a, b] >= IOU_THRESHOLD}
        for k, (j, synset, depth) in enumerate(outputs):
            added = img["boxes"][j]["source"] == "owlv2"
            n_added += added
            if k not in matched:
                continue
            g = img["gts"][matched[k]]
            t = g["synset"]
            if t in ancestors(synset):
                out_set = evaluated_set(t)  # GTより細かい語は，GTまでの経路が全て正しいとみなす（GTより下は判定できない）
            else:
                out_set = evaluated_set(synset)
            hit = len(out_set & evaluated_set(t))
            inter += hit
            pred += len(out_set)
            named = bool(out_set) and (synset in ancestors(t) or t in ancestors(synset))
            st = stats[g["kind"]]
            st["detected"] += 1
            st["named"] += named
            st["fine"] += t in ancestors(synset)
            st["inter"] += hit
            n_added_ok += added and named
            depths.append(depth)
    h_p = inter / pred if pred else float("nan")
    h_r = inter / gt_total if gt_total else float("nan")
    out = {"hP": h_p, "hR": h_r, "hF": 2 * h_p * h_r / (h_p + h_r) if pred and h_p + h_r else 0.0,
           "mean_depth": float(np.mean(depths)) if depths else float("nan"),
           "added_per_image": n_added / len(images), "added_precision": n_added_ok / n_added if n_added else float("nan")}
    for kind, st in stats.items():
        n = max(st["n"], 1)
        out |= {f"{kind}_n": st["n"], f"{kind}_detected": st["detected"] / n, f"{kind}_named": st["named"] / n,
                f"{kind}_fine": st["fine"] / n, f"{kind}_hR": st["inter"] / max(st["size"], 1)}
    if keep is not None:
        out["outputs"] = kept
    return out


# ===================== 定性評価の図 =====================

def draw(coco, outputs, path, title):
    """
    出力ボックスと単語の列を画像に描画して保存する．

    Args:
        coco (COCO): COCOオブジェクト．
        outputs (dict): 画像ID → 出力のリスト．
        path (str): 保存先．
        title (str): 図の見出し．

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (image_id, items) in zip(axes.flat, sorted(outputs.items())):
        ax.imshow(load_image(coco, int(image_id)))
        ax.axis("off")
        ax.set_title(f"{title}: image {image_id}", fontsize=9)
        for k, it in enumerate(items):
            x0, y0, x1, y1 = it["box"]
            color = "tab:blue" if it["source"] == "yolo" else "tab:red"
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
            ax.text(x0, y0, " > ".join(it["words"]), fontsize=6, color="white", va="top",
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===================== 実行 =====================

def run_config(name, cfg, ctx, data, cache, seed, keep):
    """
    1構成の τ を検証用で選び，テスト用で評価して保存する（完了済みならスキップする）．

    Args:
        name (str): 構成名．
        cfg (dict): 構成．
        ctx (dict): load_context の戻り値．
        data (dict): 分割名 → prepare の戻り値．
        cache (dict): 特徴と展開結果のキャッシュ．
        seed (int): シード．
        keep (set[str]): 定性評価の画像ID．

    Returns:
        dict: テスト用の指標と τ．
    """
    out_dir = target_dir(name, "test", seed)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        with open(os.path.join(out_dir, "metrics.json")) as f:
            return json.load(f)
    start = time.time()
    if cache.get("embedder_name") != cfg["embedder"]:
        cache.clear()
        torch.cuda.empty_cache()
        cache["embedder"], cache["embedder_name"] = Embedder(cfg["embedder"], seed), cfg["embedder"]
    emb = cache["embedder"]
    ch = {}
    for split in SPLITS:
        fkey = ("feat", cfg["crop"], split)
        if fkey not in cache:
            cache[fkey] = features(emb, ctx["coco"], data[split], cfg["crop"])
        ckey = ("chain", cfg["crop"], split, cfg["root_yolo"], cfg["pruned"], cfg["lookahead"], cfg["mode"])
        if ckey not in cache:
            cache[ckey] = chains(emb, data[split], cache[fkey], cfg)
        ch[split] = cache[ckey]
    tau, curve = 0.0, None
    if cfg["stop"]:
        curve = pd.DataFrame([{"tau": t} | evaluate(data["dev"], ch["dev"], cfg, t) for t in TAUS])
        tau = float(curve.loc[curve["hF"].idxmax(), "tau"])
    test = evaluate(data["test"], ch["test"], cfg, tau, keep)
    outputs = test.pop("outputs")
    dev = evaluate(data["dev"], ch["dev"], cfg, tau)
    for split, metrics in [("dev", dev), ("test", test)]:
        d = target_dir(name, split, seed)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics | {"tau": tau}, f, indent=4)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg | {"tau": tau, "taus": TAUS, "k_add": K_ADD, "yolo_score": YOLO_SCORE, "d_min": D_MIN,
                             "seed": seed, "elapsed_sec": time.time() - start}, f, indent=4)
    if curve is not None:
        curve.to_csv(os.path.join(target_dir(name, "dev", seed), "tau_curve.csv"), index=False)
    if name in QUALITATIVE_CONFIGS:
        q_dir = os.path.join(OUTPUT_DIR, EXPERIMENT, "qualitative")
        os.makedirs(q_dir, exist_ok=True)
        with open(os.path.join(q_dir, f"{name}.json"), "w") as f:
            json.dump(outputs, f, indent=1)
        draw(ctx["coco"], outputs, os.path.join(q_dir, f"grid_{name}.png"), name)
    print(name, "tau", tau, {k: round(v, 4) for k, v in test.items() if k in
                             ["hP", "hR", "hF", "noncoco_detected", "noncoco_named", "noncoco_fine", "coco_named",
                              "added_precision", "mean_depth"]}, flush=True)
    return test | {"tau": tau}


def qualitative_ids(data, seed):
    """
    定性評価の画像として，テスト用の画像のうちCOCO外の物体が3個以上ある画像から N_QUALITATIVE 枚を選ぶ．

    Args:
        data (dict): 分割名 → prepare の戻り値．
        seed (int): シード．

    Returns:
        set[str]: 画像ID．
    """
    import random

    cands = sorted(img["image_id"] for img in data["test"] if sum(g["kind"] == "noncoco" for g in img["gts"]) >= 3
                   and len(img["gts"]) <= 12)
    return set(random.Random(seed).sample(cands, N_QUALITATIVE))


def main():
    """全構成を実行し，集計表を作成する（完了済みの構成はスキップ）．"""
    for seed in SEEDS:
        ctx = load_context(seed)
        data = {}
        for split in SPLITS:
            proposals = owlv2_proposals(ctx, split, OWLv2Objectness)
            data[split] = prepare(ctx, proposals, split)
        torch.cuda.empty_cache()
        keep = qualitative_ids(data, seed)
        cache, rows = {}, []
        for name, cfg in CONFIGS.items():
            rows.append({"config": name} | cfg | run_config(name, cfg, ctx, data, cache, seed, keep))
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, EXPERIMENT, "table_results.csv"), index=False)


if __name__ == "__main__":
    main()
