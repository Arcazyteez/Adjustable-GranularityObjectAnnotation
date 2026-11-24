"""
ex011の実行スクリプト：COCO外の物体の命名の改善（order_012）．

基準はex010の hybrid+owlv2（YOLO11x（閾値0.25）のボックスに，YOLO11xと重ならないOWLv2の候補を加える）である．
YOLO11xのボックスは全構成で共通とし，YOLO11xの語から下位語をたどってシグモイドの累積確信度が0.05未満になる前の深さまで出力する（ex009の最良の設定）．
加えた候補（1画像あたり最大 K_POOL 件）について，以下を比較する．

- 提案2（棄却）: OWLv2の物体らしさ ≥ o，出力した語のシグモイドの確信度 ≥ γ，他の出力ボックスに面積の80%以上が含まれない
- 提案1（命名）:
    flat:     語彙から深さ0の語を選び，下位語をたどる（ex010と同じ）．深さ1以降は各段階のソフトマックス ≥ τ_s の間だけ出力
    topdown:  キャプションで頻出するカテゴリの語（animal，vehicle 等）とCOCOの80語から出発点を選び，キャプションで絞り込んだ
              WordNetの木をたどる．各段階のソフトマックス ≥ τ_s の間だけ出力する（粗い語で止める）
- 提案3・4: 語彙（wn，wn_cap），VLM（B16，SO400M），クロップの拡大率（1.0，1.5），プロンプト（plain，hier）

評価: 出力ボックスとGT（COCOの物体とCOCO外の物体）をIoU 0.5以上で1対1に対応付け，上位語を含めたSynsetの集合
（min_depth ≥ D_MIN）による階層的適合率 hP，再現率 hR，F値 hF を求める．対応しない出力ボックスは hP の分母のみ，
対応しないGTは hR の分母のみに含める．閾値（K，o，γ，containment，τ_s）は，lvis_dev の DEV_IMAGES 枚において，
1画像あたりの加えた候補の数が予算 B 以下となる組のうち，COCO外の物体の hR が最大となる組を選ぶ（B ∈ BUDGETS）．
参考として，全体の hF が最大となる組も選ぶ．

出力: outputs/ex011_open_world_naming/{構成名}/{分割}/{seed}/metrics_{B{予算} または hF}.json，config.json，grid.csv（lvis_dev），
    records_B2.json（lvis_test）
    および直下の table_results.csv
"""
import functools
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet as wn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.metrics import box_iou  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402
from data import SPLITS, ground_truth, import_experiment_module, load_context, owlv2_proposals, vocabulary  # noqa: E402
from model import context_crop, flat_chain, hyponym_chain, load_model, top_down_chain  # noqa: E402

# ===== 実験条件（order_012） =====
EXPERIMENT = "ex011_open_world_naming"
SEEDS = [42]
IOU_THRESHOLD = 0.5
YOLO_CONF = 0.25
YOLO_TAU = 0.05          # YOLO11xのボックスの停止規則（シグモイドの累積確信度）
K_POOL = 20              # 加える候補の最大数
DEV_IMAGES = 1000        # 閾値の選択に用いる lvis_dev の画像数
D_MIN = 4                # 評価に用いるSynsetの最小の深さ（entity，physical_entity，object，whole 等を除く）
CONTAIN_RATIO = 0.8
GRID = {"k": [1, 2, 5, 10, 20], "o": [0.0, 0.1, 0.2, 0.3], "gamma": [0.0, 0.003, 0.01, 0.03, 0.1],
        "contain": [False, True], "tau_s": [0.0, 0.5, 0.7, 0.9, 1.01]}
# 1画像あたりに加える候補の数の上限（予算）．全体の hF で閾値を選ぶと，LVISの注釈が網羅的でないため
# 加えた候補がほぼ全て棄却される設定が選ばれる．そこで，予算ごとにCOCO外の物体の hR が最大となる閾値を選んで比較する
BUDGETS = [1, 2, 5]
# 構成名 → (命名，語彙，VLM，クロップの拡大率，プロンプト，棄却を行うか，停止を行うか)
CONFIGS = {
    "ex010_baseline":        ("flat", "wn", "B16", 1.0, "plain", False, False),   # 物体らしさの上位 K 件をそのまま加える
    "p2_reject":             ("flat", "wn", "B16", 1.0, "plain", True, False),
    "p1_flat_stop":          ("flat", "wn", "B16", 1.0, "plain", True, True),
    "p1_topdown":            ("topdown", None, "B16", 1.0, "plain", True, True),
    "p3_flat_wncap":         ("flat", "wn_cap", "B16", 1.0, "plain", True, True),
    "p4_topdown_crop1.5":    ("topdown", None, "B16", 1.5, "plain", True, True),
    "p4_topdown_hier":       ("topdown", None, "B16", 1.0, "hier", True, True),
    "p4_topdown_so400m":     ("topdown", None, "SO400M", 1.0, "plain", True, True),
    "p4_flat_wncap_so400m":  ("flat", "wn_cap", "SO400M", 1.0, "plain", True, True),
    "p34_topdown_all":       ("topdown", None, "SO400M", 1.5, "hier", True, True),
    "p34_flat_wncap_all":    ("flat", "wn_cap", "SO400M", 1.5, "hier", True, True),
}


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


# ===================== 出力ボックスの候補 =====================

def image_ids_for(ctx, split, seed):
    """
    分割の評価に用いる画像IDを返す（lvis_dev は DEV_IMAGES 枚をサンプリングする）．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．
        seed (int): シード．

    Returns:
        list[str]: 画像ID．
    """
    ids = [str(i) for i in ctx["splits"][split]]
    return sorted(random.Random(seed).sample(ids, DEV_IMAGES)) if split == "lvis_dev" else ids


def candidate_boxes(ctx, proposals, image_id):
    """
    1枚の画像のYOLO11xのボックスと，YOLO11xと重ならないOWLv2の候補（物体らしさの上位 K_POOL 件）を返す．

    Args:
        ctx (dict): load_context の戻り値．
        proposals (dict): owlv2_proposals の戻り値．
        image_id (str): 画像ID．

    Returns:
        list[dict]: {"box"，"score"，"source"（"yolo" または "owlv2"），("yolo_word")}（YOLO11xが先）．
    """
    det = ctx["detections"]["yolo"][image_id]
    out = [{"box": b, "score": s, "source": "yolo", "yolo_word": t[0]}
           for b, s, t in zip(det["boxes"], det["scores"], det["top5"]) if s > YOLO_CONF]
    cand = proposals[image_id]
    base = np.asarray([b["box"] for b in out], dtype=np.float32).reshape(-1, 4)
    boxes = np.asarray(cand["boxes"], dtype=np.float32).reshape(-1, 4)
    overlap = box_iou(boxes, base).max(axis=1) if len(base) and len(boxes) else np.zeros(len(boxes))
    extra = [i for i in range(len(boxes)) if overlap[i] < IOU_THRESHOLD][:K_POOL]
    return out + [{"box": cand["boxes"][i], "score": cand["scores"][i], "source": "owlv2"} for i in extra]


def containment(boxes):
    """
    各ボックスについて，自身より大きい他のボックスに含まれる面積の割合の最大値を計算する．

    Args:
        boxes (np.ndarray): xyxy形式のボックス（N×4）．

    Returns:
        np.ndarray: 割合（N）．
    """
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ratio = inter / np.maximum(area[:, None], 1e-9)
    ratio[~(area[None, :] > area[:, None])] = 0  # 自身より大きいボックスのみを対象とする
    np.fill_diagonal(ratio, 0)
    return ratio.max(axis=1)


def prepare_images(ctx, proposals, gts, image_ids):
    """
    画像ごとに出力ボックスの候補，GTとのIoU，包含の割合を前計算する．

    Args:
        ctx (dict): load_context の戻り値．
        proposals (dict): owlv2_proposals の戻り値．
        gts (dict): 画像ID → GT．
        image_ids (list[str]): 画像ID．

    Returns:
        list[dict]: 画像ごとの {"image_id"，"boxes"，"gts"，"iou"（出力×GT），"contain"}．
    """
    images = []
    for image_id in image_ids:
        boxes = candidate_boxes(ctx, proposals, image_id)
        b = np.asarray([x["box"] for x in boxes], dtype=np.float32).reshape(-1, 4)
        g = np.asarray([x["box"] for x in gts[image_id]], dtype=np.float32).reshape(-1, 4)
        iou = box_iou(b, g) if len(b) and len(g) else np.zeros((len(b), len(g)))
        images.append({"image_id": image_id, "boxes": boxes, "gts": gts[image_id], "iou": iou, "contain": containment(b)})
    return images


# ===================== 画像特徴と命名 =====================

def compute_features(vlm, coco, images, scale):
    """
    全出力ボックスの候補のクロップの画像特徴を計算する．

    Args:
        vlm (SigLIPModel): VLM．
        coco (COCO): instancesのCOCOオブジェクト．
        images (list[dict]): prepare_images の戻り値．
        scale (float): クロップの拡大率．

    Returns:
        list[torch.Tensor]: 画像ごとの画像特徴（ボックス数×D）．
    """
    feats = []
    for img in tqdm(images, desc=f"{vlm.name} x{scale} features", leave=False):
        image = load_image(coco, int(img["image_id"]))
        crops = [context_crop(image, b["box"], scale) for b in img["boxes"]]
        if not crops:
            feats.append(torch.zeros(0))
            continue
        # SO400M は計算量が大きいため，画像エンコーダを半精度で計算する（特徴は単精度に戻して正規化する）
        with torch.autocast("cuda", dtype=torch.float16, enabled=vlm.name == "SigLIP-SO400M"):
            f = vlm.image_features(crops)
        f = f.float()
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return feats


def name_boxes(vlm, images, feats, naming, vocab_name, mode):
    """
    全出力ボックスの候補を命名し，各段階の語と確信度を返す．YOLO11xのボックスはYOLO11xの語から下位語をたどる．

    Args:
        vlm (SigLIPModel): VLM．
        images (list[dict]): prepare_images の戻り値．
        feats (list[torch.Tensor]): compute_features の戻り値．
        naming (str): "flat" または "topdown"．
        vocab_name (str | None): flat の語彙名．
        mode (str): プロンプトの種類．

    Returns:
        list[list[list[dict]]]: 画像ごと・ボックスごとの各段階の {"synset"，"word"，"soft"，"sig"}．
    """
    vocab = vocabulary(vocab_name) if naming == "flat" else None
    chains = []
    for img, feat in zip(tqdm(images, desc=f"naming {naming} {vocab_name} {mode}", leave=False), feats):
        per_image = []
        for b, f in zip(img["boxes"], feat):
            if b["source"] == "yolo":
                root = {"synset": COCO_TO_SYNSET[b["yolo_word"]], "word": b["yolo_word"], "soft": 1.0, "sig": b["score"]}
                per_image.append([root] + hyponym_chain(vlm, f, root["synset"], root["word"], mode))
            elif naming == "flat":
                per_image.append(flat_chain(vlm, f, vocab, mode))
            else:
                per_image.append(top_down_chain(vlm, f, mode))
        chains.append(per_image)
    return chains


def output_node(box, chain, naming, tau_s, stop):
    """
    停止規則に従って出力するSynsetと，その確信度（シグモイド）を返す．

    Args:
        box (dict): 出力ボックスの候補．
        chain (list[dict]): name_boxes の各段階．
        naming (str): 命名の方法．
        tau_s (float): 各段階のソフトマックスの閾値（加えた候補のみ）．
        stop (bool): 停止を行うか（False は最後まで出力）．

    Returns:
        tuple[str | None, float]: Synset名（出力しない場合 None）と確信度．
    """
    if box["source"] == "yolo":
        cum = np.cumprod([c["sig"] for c in chain])
        depth = int(np.nonzero(cum >= YOLO_TAU)[0][-1]) if cum[0] >= YOLO_TAU else 0
        return chain[depth]["synset"], chain[depth]["sig"]
    if not stop:
        return chain[-1]["synset"], chain[-1]["sig"]
    depth = 0  # 深さ0（flat は語彙から選んだ語，topdown は出発点の語）は停止の対象としない
    for i in range(1, len(chain)):
        if chain[i]["soft"] < tau_s:
            break
        depth = i
    return chain[depth]["synset"], chain[depth]["sig"]


# ===================== 評価指標 =====================

@functools.lru_cache(maxsize=None)
def ancestors(synset_name):
    """
    Synset自身とその全ての上位語のうち，min_depth ≥ D_MIN のものの集合を返す．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    s = wn.synset(synset_name)
    return frozenset(x.name() for x in [s] + list(s.closure(lambda y: y.hypernyms())) if x.min_depth() >= D_MIN)


def target_synset(gt):
    """
    GTのSynset名を返す．

    Args:
        gt (dict): GT．

    Returns:
        str: COCOの物体はCOCOのカテゴリのSynset，COCO外の物体はLVISのSynset．
    """
    return COCO_TO_SYNSET[gt["name"]] if gt["kind"] == "coco" else gt["synset"]


def evaluate(images, chains, naming, params, stop, keep_records=False):
    """
    1組の閾値で，棄却・停止・対応付けを行い，階層的な指標を計算する．

    Args:
        images (list[dict]): prepare_images の戻り値．
        chains (list): name_boxes の戻り値．
        naming (str): 命名の方法．
        params (dict): k，o，gamma，contain，tau_s．
        stop (bool): 加えた候補で停止を行うか．
        keep_records (bool): 対応付けの記録を返すか．

    Returns:
        dict: hP，hR，hF，COCO外の物体の detected / named / reach / hR，COCOの物体の hR / named，extra_per_image，
            extra_precision（加えた候補のうちGTと対応し正しい上位語を出力した割合）と（"records"）．
    """
    inter = n_pred = n_gt = 0
    stats = {"noncoco": [0, 0, 0, 0, 0], "coco": [0, 0, 0, 0, 0]}  # [n, detected, named, inter, |G|]
    n_extra = n_extra_ok = 0
    records = []
    for img, chain in zip(images, chains):
        outputs = []
        n_owl = 0
        for j, (b, c) in enumerate(zip(img["boxes"], chain)):
            if b["source"] != "yolo":
                n_owl += 1
                if n_owl > params["k"] or b["score"] < params["o"] or (params["contain"] and img["contain"][j] >= CONTAIN_RATIO):
                    continue
            synset, conf = output_node(b, c, naming, params["tau_s"], stop)
            if synset is None or not ancestors(synset) or (b["source"] != "yolo" and conf < params["gamma"]):
                continue
            outputs.append((j, synset))
        rows = [j for j, _ in outputs]
        iou = img["iou"][rows] if rows else np.zeros((0, len(img["gts"])))
        matched = {}
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            matched = {int(a): int(b) for a, b in zip(r, c) if iou[a, b] >= IOU_THRESHOLD}
        for g in img["gts"]:
            n_gt += len(ancestors(target_synset(g)))
        for g in img["gts"]:
            stats[g["kind"]][0] += 1
            stats[g["kind"]][4] += len(ancestors(target_synset(g)))
        for i, (j, synset) in enumerate(outputs):
            pred = ancestors(synset)
            is_extra = img["boxes"][j]["source"] != "yolo"
            n_extra += is_extra
            if i not in matched:
                n_pred += len(pred)
                continue
            g = img["gts"][matched[i]]
            t = target_synset(g)
            gt_set = ancestors(t)
            # GTより細かい出力は正誤を判定できないため，GTのSynsetの真の下位語は除く
            pred = frozenset(p for p in pred if p in gt_set or t not in ancestors(p))
            hit = len(pred & gt_set)
            inter += hit
            n_pred += len(pred)
            named = synset in gt_set or t in ancestors(synset)
            s = stats[g["kind"]]
            s[1] += 1
            s[2] += named
            s[3] += hit
            n_extra_ok += is_extra and named
            if keep_records:
                records.append({"image_id": img["image_id"], "box": img["boxes"][j]["box"], "source": img["boxes"][j]["source"],
                                "output": synset, "gt_kind": g["kind"], "gt": t, "named": bool(named)})
    h_p = inter / n_pred if n_pred else float("nan")
    h_r = inter / n_gt if n_gt else float("nan")
    out = {"hP": h_p, "hR": h_r, "hF": 2 * h_p * h_r / (h_p + h_r) if n_pred and h_p + h_r else 0.0,
           "extra_per_image": n_extra / len(images), "extra_precision": n_extra_ok / n_extra if n_extra else float("nan")}
    for kind, (n, det, named, hit, size) in stats.items():
        if n == 0:
            out |= {f"{kind}_n": 0, f"{kind}_detected": float("nan"), f"{kind}_named": float("nan"), f"{kind}_hR": float("nan")}
            continue
        out |= {f"{kind}_n": n, f"{kind}_detected": det / n, f"{kind}_named": named / n, f"{kind}_hR": hit / size}
    if keep_records:
        out["records"] = records
    return out


def param_grid(reject, stop):
    """
    構成で探索する閾値の組を返す．

    Args:
        reject (bool): 棄却を行うか（False は K のみを変え，o=γ=0，containment なし）．
        stop (bool): 停止を行うか（False は τ_s=0）．

    Returns:
        list[dict]: 閾値の組のリスト．
    """
    keys = ["k", "o", "gamma", "contain", "tau_s"]
    grid = {"k": GRID["k"], "o": GRID["o"] if reject else [0.0],
            "gamma": GRID["gamma"] if reject else [0.0], "contain": GRID["contain"] if reject else [False],
            "tau_s": GRID["tau_s"] if stop else [0.0]}
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


def dev_grid(images, chains, naming, reject, stop):
    """
    lvis_dev で全ての閾値の組の指標を計算する．

    Args:
        images (list[dict]): lvis_dev の prepare_images の戻り値．
        chains (list): name_boxes の戻り値．
        naming (str): 命名の方法．
        reject (bool): 棄却を行うか．
        stop (bool): 停止を行うか．

    Returns:
        pd.DataFrame: 行が閾値の組，列が閾値と指標．
    """
    rows = [p | evaluate(images, chains, naming, p, stop) for p in tqdm(param_grid(reject, stop), desc="grid", leave=False)]
    return pd.DataFrame(rows)


def choose(table, budget):
    """
    予算 budget 以下の組のうちCOCO外の物体の hR が最大の組を選ぶ（budget が None の場合は全体の hF が最大の組）．

    Args:
        table (pd.DataFrame): dev_grid の戻り値．
        budget (float | None): 1画像あたりの加えた候補の数の上限．

    Returns:
        dict: 閾値の組．
    """
    if budget is None:
        best = table.loc[table["hF"].idxmax()]
    else:
        feasible = table[table["extra_per_image"] <= budget]
        best = feasible.sort_values(["noncoco_hR", "hP"], ascending=False).iloc[0]
    return {k: (bool(best[k]) if k == "contain" else (int(best[k]) if k == "k" else float(best[k])))
            for k in ["k", "o", "gamma", "contain", "tau_s"]}


def get_chains(cache, ctx, data, vlm_name, scale, naming, vocab_name, mode, seed):
    """
    分割ごとの命名結果を返す（画像特徴と命名結果はキャッシュし，VLMは必要なときのみ読み込み直す）．

    Args:
        cache (dict): キャッシュ．
        ctx (dict): load_context の戻り値．
        data (dict): 分割名 → prepare_images の戻り値．
        vlm_name (str): VLMの名前．
        scale (float): クロップの拡大率．
        naming (str): 命名の方法．
        vocab_name (str | None): 語彙名．
        mode (str): プロンプトの種類．
        seed (int): シード．

    Returns:
        dict: 分割名 → name_boxes の戻り値．
    """
    chains = {}
    for split in SPLITS:
        fkey = (vlm_name, scale, split)
        nkey = (vlm_name, scale, split, naming, vocab_name, mode)
        if nkey in cache:
            chains[split] = cache[nkey]
            continue
        if cache.get("vlm_name") != vlm_name:
            cache.pop("vlm", None)
            for k in [k for k in cache if isinstance(k, tuple) and len(k) == 3]:
                cache.pop(k)  # 別のVLMの画像特徴はGPUのメモリを空けるため破棄する
            torch.cuda.empty_cache()
            cache["vlm"], cache["vlm_name"] = load_model(vlm_name, seed), vlm_name
        if fkey not in cache:
            cache[fkey] = compute_features(cache["vlm"], ctx["coco"], data[split], scale)
        cache[nkey] = name_boxes(cache["vlm"], data[split], cache[fkey], naming, vocab_name, mode)
        chains[split] = cache[nkey]
    return chains


def run_config(name, config, ctx, data, cache, seed):
    """
    1構成について，lvis_dev で閾値を選び，lvis_test で評価して保存する（完了済みならスキップする）．

    Args:
        name (str): 構成名．
        config (tuple): CONFIGS の値．
        ctx (dict): load_context の戻り値．
        data (dict): 分割名 → prepare_images の戻り値．
        cache (dict): 画像特徴と命名結果のキャッシュ．
        seed (int): シード．

    Returns:
        dict: 予算（"B1"，"B2"，"B5"）と "hF" → lvis_test の指標と閾値．
    """
    out_dir = target_dir(name, "lvis_test", seed)
    if os.path.exists(os.path.join(out_dir, "config.json")):
        return {label: json.load(open(os.path.join(out_dir, f"metrics_{label}.json")))
                for label in [f"B{b}" for b in BUDGETS] + ["hF"]}
    naming, vocab_name, vlm_name, scale, mode, reject, stop = config
    start = time.time()
    chains = get_chains(cache, ctx, data, vlm_name, scale, naming, vocab_name, mode, seed)
    table = dev_grid(data["lvis_dev"], chains["lvis_dev"], naming, reject, stop)
    dev_dir = target_dir(name, "lvis_dev", seed)
    os.makedirs(dev_dir, exist_ok=True)
    table.to_csv(os.path.join(dev_dir, "grid.csv"), index=False)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for label, budget in [(f"B{b}", b) for b in BUDGETS] + [("hF", None)]:
        params = choose(table, budget)
        test = evaluate(data["lvis_test"], chains["lvis_test"], naming, params, stop, keep_records=(label == "B2"))
        records = test.pop("records", None)
        with open(os.path.join(out_dir, f"metrics_{label}.json"), "w") as f:
            json.dump(test | {"params": params}, f, indent=4)
        if records is not None:
            with open(os.path.join(out_dir, "records_B2.json"), "w") as f:
                json.dump(records, f)
        results[label] = test | {"params": params}
        print(name, label, params, {k: round(test[k], 4) for k in ["hP", "hF", "noncoco_detected", "noncoco_named",
                                                                    "noncoco_hR", "coco_hR", "extra_per_image",
                                                                    "extra_precision"]}, flush=True)
    config_dict = {"naming": naming, "vocab": vocab_name, "vlm": vlm_name, "crop_scale": scale, "prompt": mode,
                   "reject": reject, "stop": stop, "budgets": BUDGETS, "grid": GRID, "k_pool": K_POOL, "d_min": D_MIN,
                   "yolo_conf": YOLO_CONF, "yolo_tau": YOLO_TAU, "dev_images": DEV_IMAGES, "seed": seed,
                   "elapsed_sec": time.time() - start}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    return results


def hybrid_only(data, ctx, cache, seed):
    """
    比較のため，加える候補を用いないハイブリッド手法（YOLO11xのみ）の lvis_test の指標を計算する．

    Args:
        data (dict): 分割名 → prepare_images の戻り値．
        ctx (dict): load_context の戻り値．
        cache (dict): キャッシュ（B16，拡大率1.0，flat，wn の命名結果を用いる．YOLO11xのボックスの命名のみを用いる）．
        seed (int): シード．

    Returns:
        dict: lvis_test の指標．
    """
    chains = get_chains(cache, ctx, data, "B16", 1.0, "flat", "wn", "plain", seed)["lvis_test"]
    metrics = evaluate(data["lvis_test"], chains, "flat", {"k": 0, "o": 0.0, "gamma": 0.0, "contain": False, "tau_s": 0.0}, True)
    out_dir = target_dir("hybrid_only", "lvis_test", seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics


def main():
    """全構成を実行し，集計表を作成する（完了済みの構成はスキップ）．"""
    m10 = import_experiment_module("ex010_lvis_open_world_detection", "model")
    for seed in SEEDS:
        ctx = load_context(seed)
        data = {}
        for split in SPLITS:
            proposals = owlv2_proposals(ctx, split, lambda: m10.load_detector("owlv2", seed))
            ids = image_ids_for(ctx, split, seed)
            gts = ground_truth(ctx, split)
            data[split] = prepare_images(ctx, proposals, gts, ids)
        cache, rows = {}, []
        for name, config in CONFIGS.items():
            info = {"config": name, "naming": config[0], "vocab": config[1], "vlm": config[2], "crop_scale": config[3],
                    "prompt": config[4], "reject": config[5], "stop": config[6]}
            for label, metrics in run_config(name, config, ctx, data, cache, seed).items():
                params = metrics.pop("params")
                rows.append(info | {"selection": label} | {f"param_{k}": v for k, v in params.items()} | metrics)
            if name == "ex010_baseline":
                rows.append({"config": "hybrid_only", "selection": "-"} | hybrid_only(data, ctx, cache, seed))
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(OUTPUT_DIR, EXPERIMENT, "table_results.csv"), index=False)


if __name__ == "__main__":
    main()
