"""
ex015の実行スクリプト：再学習なしで未学習の物体と語彙を検出・命名する粒度可変アノテーションのやり直し（order_019）．

学習は行わない．以下の段階を順に行い，各段階では前の段階で選んだ設定を引き継ぐ．閾値等は検証用（dev）で選び，テスト用（test）で報告する．
- S0 原因の切り分け: 加えた領域のうちCOCO外の物体と重なるものについて，OWLv2にLVISの語彙（正解を含む）とキャプションの語彙を
     与えた場合の1位の正解率を測る（st：LVISの人手の注釈を用いない重み，ens：LVIS baseでファインチューニングした重み）
- S1 語の選び方: 加えた領域を，OWLv2（st）の領域の特徴とキャプションの語彙の確率をWordNetの上位語へ集約して命名し，
     規則（flat，deepest，backoff，greedy），温度 β，閾値 τ を選ぶ．YOLO11xのボックスはYOLO11xのクラスのみを出力する
- S2 文章と語彙: 単語の文章（plain，gloss，ens）と，人の役割を表す語（person の下位語）を語彙から除くかを選ぶ
- S3 YOLO11xのボックスの細分化: YOLO11xのクラスの下位語に限った確率で，同じ規則によりクラスより細かい語を選ぶ（閾値 τ_y）
- S4 加える領域の数と棄却: 加える領域の数 K と，物体らしさの閾値 o を選ぶ
- S5 より大きいOWLv2（st_large）: 領域のキャッシュがある場合のみ，S4までの設定で評価する

評価: 出力ボックスとLVISのGTをIoU 0.5以上で1対1に対応付け，ex013と同じ階層的適合率・再現率・F値（hP，hR，hF，GTと対応しない出力は除く）と，
LVISの federated な評価（GTと対応しない出力のうち，画像に存在しないと確認されたカテゴリ（neg）か，網羅的に注釈されたカテゴリの語またはその下位語を
出力したものを誤りとして hP に含める）による fed_hP，fed_hF を求める．閾値の選択には fed_hF を用いる．

出力: outputs/ex015_lvis_open_vocabulary_annotation/{段階}_{構成}/{分割}/{seed}/metrics.json，config.json
    diagnosis.json，grid_{段階}.csv，qualitative/，および直下の table_results.csv
"""
import itertools
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定論的なCUDAの行列計算に必要な設定

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from nltk.corpus import wordnet as wn  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from tqdm import tqdm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402
from data import (DATA_DIR, SPLITS, ancestors, build_images, caption_vocabulary, ex013_modules, load_context,  # noqa: E402
                  lvis_vocabulary, region_cache)
from model import DEVICE, Hierarchy, OWLv2Regions, query_features  # noqa: E402

# ===== 実験条件（order_019） =====
EXPERIMENT = "ex015_lvis_open_vocabulary_annotation"
SEEDS = [42]
MAIN_MODEL = "st"
IOU_THRESHOLD = 0.5
BETAS = [1.0, 2.0, 4.0, 8.0]          # ソフトマックスの温度（OWLv2のロジットの scale に掛ける倍率）
TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TAUS_YOLO = [0.3, 0.5, 0.7, 0.9, 1.01]  # 1.01 はクラスより細かい語を出力しない
KS = [10, 20, 50, 100]
OBJECTNESS = [0.0, 0.05, 0.1, 0.2]
BASE_K = 10                            # S1〜S3で加える領域の数（ex013・ex014と同じ）


def target_dir(name, split, seed):
    """
    1構成の保存先を返す．

    Args:
        name (str): 構成名．
        split (str): 分割名．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, name, split, str(seed))


# ===================== 確率と出力するSynset =====================

def vocab_variant(vocab, exclude_person):
    """
    語彙から人の役割を表す語を除く（person.n.01 自身は残す）．

    Args:
        vocab (list[dict]): caption_vocabulary の戻り値．
        exclude_person (bool): 除くか．

    Returns:
        list[dict]: 語彙．
    """
    return [v for v in vocab if not (exclude_person and v["person"])]


class Namer:
    """1つのOWLv2と語彙・文章の組について，領域ごとの語彙上の確率を計算し，規則に従って出力するSynsetを選ぶ．"""

    def __init__(self, owl, vocab, variant):
        """
        Args:
            owl (OWLv2Regions): OWLv2（文章の特徴と領域の埋め込みの計算に用いる）．
            vocab (list[dict]): 語彙．
            variant (str): 単語の文章の種類．
        """
        self.owl, self.vocab = owl, vocab
        self.hier = Hierarchy(vocab)
        self.query = query_features(owl, vocab, variant)
        self._under = {}

    def probs(self, feats, beta):
        """
        領域の特徴から語彙上の確率を計算する．

        Args:
            feats (torch.Tensor): 領域の特徴（N×D）．
            beta (float): 温度の倍率．

        Returns:
            torch.Tensor: 確率（N×V）．
        """
        e, scale, _ = self.owl.region_embeddings(feats)
        return (beta * scale[:, None] * (e @ self.query.T)).softmax(dim=1)

    def word_mask_under(self, synset):
        """
        指定したSynsetの下位語または自身である語彙のマスクを返す（キャッシュする）．

        Args:
            synset (str): Synset名．

        Returns:
            torch.Tensor: マスク（V，float）．
        """
        if synset not in self._under:
            self._under[synset] = torch.tensor([synset in ancestors(v["synset"]) for v in self.vocab],
                                               dtype=torch.float32, device=DEVICE)
        return self._under[synset]

    def added(self, feats, beta, rule, taus):
        """
        加えた領域の出力するSynsetを，全ての τ について求める．

        Args:
            feats (torch.Tensor): 領域の特徴（N×D）．
            beta (float): 温度の倍率．
            rule (str): 規則．
            taus (list[float]): 閾値．

        Returns:
            dict[float, list[str]]: τ → Synset名のリスト（長さN）．
        """
        if len(feats) == 0:
            return {t: [] for t in taus}
        out = self.hier.choose(self.probs(feats, beta), taus, rule)
        return {t: [self.hier.nodes[int(n)] for n in idx] for t, (idx, _) in out.items()}

    def yolo(self, feats, labels, beta, rule, taus):
        """
        YOLO11xのボックスについて，クラスの下位語に限った確率で，クラスより細かい語を全ての τ_y について求める．
        クラスのSynsetの確率は1となるため，閾値を超える細かい語が無い場合はクラスを出力する．

        Args:
            feats (torch.Tensor): YOLO11xのボックスに対応する領域の特徴（N×D）．
            labels (list[str]): YOLO11xのクラス名（長さN）．
            beta (float): 温度の倍率．
            rule (str): 規則．
            taus (list[float]): 閾値．

        Returns:
            dict[float, list[str]]: τ_y → Synset名のリスト（長さN）．
        """
        roots = [COCO_TO_SYNSET[l] for l in labels]
        out = {t: list(roots) for t in taus}
        if len(feats) == 0:
            return out
        p = self.probs(feats, beta)
        for k, root in enumerate(roots):
            mask = self.word_mask_under(root)
            if mask.sum() <= 1 or root not in self.hier.index:
                continue
            q = p[k:k + 1] * mask
            q = q / q.sum()
            r = "backoff" if rule in ("deepest", "flat") else rule
            start = torch.tensor([self.hier.index[root]], device=DEVICE) if r == "greedy" else None
            for t, (idx, _) in self.hier.choose(q, taus, r, start=start).items():
                node = self.hier.nodes[int(idx[0])]
                out[t][k] = node if root in ancestors(node) else root
        return out


# ===================== 評価指標 =====================

def evaluate(images, outputs, k_add, objectness, keep=None):
    """
    出力からLVISのGTに対する指標を計算する．

    Args:
        images (list[dict]): build_images の戻り値．
        outputs (list[list[str]]): 画像ごと・ボックスごとの出力するSynset名（ボックスの並びは images と同じ）．
        k_add (int): 加える領域の最大数．
        objectness (float): 加える領域の物体らしさの下限．
        keep (set[str] | None): 出力を記録する画像ID．

    Returns:
        dict: hP，hR，hF，fed_hP，fed_hF，strict_hP，strict_hF（GTと対応しない出力を全て誤りとする），{coco, noncoco}_{n, detected, named, fine, hR}，added_per_image，added_precision，
            noncoco_named_depth，n_false（federated な評価で誤りとした出力の数），（"outputs"）．
    """
    t13 = ex013_modules()[2]
    category_synset = None
    inter = pred = gt_total = fed_extra = n_false = strict_extra = 0
    stats = {k: {"n": 0, "detected": 0, "named": 0, "fine": 0, "inter": 0, "size": 0} for k in ["coco", "noncoco"]}
    n_added = n_added_ok = 0
    depths, kept = [], {}
    for img, out in zip(images, outputs):
        if category_synset is None:
            category_synset = img["category_synset"]
        sel, n_owl = [], 0
        for j, b in enumerate(img["boxes"]):
            if b["source"] == "owlv2":
                n_owl += 1
                if n_owl > k_add or b["score"] < objectness:
                    continue
            sel.append(j)
        if keep is not None and img["image_id"] in keep:
            kept[img["image_id"]] = [{"box": img["boxes"][j]["box"], "source": img["boxes"][j]["source"], "synset": out[j]}
                                     for j in sel]
        for g in img["gts"]:
            size = len(t13.evaluated_set(g["synset"]))
            gt_total += size
            stats[g["kind"]]["n"] += 1
            stats[g["kind"]]["size"] += size
        matched = {}
        if sel and len(img["gts"]):
            iou = img["iou"][sel]
            r, c = linear_sum_assignment(-iou)
            matched = {int(a): int(b) for a, b in zip(r, c) if iou[a, b] >= IOU_THRESHOLD}
        checked = img["checked_synsets"]
        for k, j in enumerate(sel):
            synset = out[j]
            added = img["boxes"][j]["source"] == "owlv2"
            n_added += added
            if k not in matched:
                strict_extra += len(t13.evaluated_set(synset))  # 厳しい評価: GTと対応しない出力を全て誤りとする
                # 存在しない，または網羅的に注釈されたカテゴリの語（またはその下位語）を出力した場合は誤りとする
                if ancestors(synset) & checked:
                    fed_extra += len(t13.evaluated_set(synset))
                    n_false += 1
                continue
            g = img["gts"][matched[k]]
            t = g["synset"]
            out_set = t13.evaluated_set(t) if t in ancestors(synset) else t13.evaluated_set(synset)
            hit = len(out_set & t13.evaluated_set(t))
            inter += hit
            pred += len(out_set)
            named = bool(out_set) and (synset in ancestors(t) or t in ancestors(synset))
            st = stats[g["kind"]]
            st["detected"] += 1
            st["named"] += named
            st["fine"] += t in ancestors(synset)
            st["inter"] += hit
            n_added_ok += added and named
            if named and g["kind"] == "noncoco":
                depths.append(wn.synset(synset).min_depth())
    h_p = inter / pred if pred else float("nan")
    h_r = inter / gt_total if gt_total else float("nan")
    f_p = inter / (pred + fed_extra) if pred + fed_extra else float("nan")
    s_p = inter / (pred + strict_extra) if pred + strict_extra else float("nan")
    res = {"strict_hP": s_p, "strict_hF": 2 * s_p * h_r / (s_p + h_r) if pred and s_p + h_r else 0.0, "hP": h_p, "hR": h_r, "hF": 2 * h_p * h_r / (h_p + h_r) if pred and h_p + h_r else 0.0,
           "fed_hP": f_p, "fed_hF": 2 * f_p * h_r / (f_p + h_r) if pred and f_p + h_r else 0.0, "n_false": n_false,
           "added_per_image": n_added / len(images), "added_precision": n_added_ok / n_added if n_added else float("nan"),
           "noncoco_named_depth": float(np.mean(depths)) if depths else float("nan")}
    for kind, st in stats.items():
        n = max(st["n"], 1)
        res |= {f"{kind}_n": st["n"], f"{kind}_detected": st["detected"] / n, f"{kind}_named": st["named"] / n,
                f"{kind}_fine": st["fine"] / n, f"{kind}_hR": st["inter"] / max(st["size"], 1)}
    if keep is not None:
        res["outputs"] = kept
    return res


def attach_checked(images, lvis):
    """
    画像ごとに，federated な評価で「出力すると誤り」と判定できるカテゴリのSynsetの集合を付ける．
    存在しないと確認されたカテゴリ（neg）と，画像に存在し網羅的に注釈されたカテゴリ（pos − not_exhaustive）である．

    Args:
        images (list[dict]): build_images の戻り値（"checked_synsets"，"category_synset" を追記する）．
        lvis (dict): load_lvis の戻り値．

    Returns:
        None
    """
    cs = lvis["category_synset"]
    for img in images:
        cats = img["neg"] | (img["pos"] - img["not_exhaustive"])
        img["checked_synsets"] = frozenset(cs[c] for c in cats if c in cs)
        img["category_synset"] = cs


# ===================== 出力の組み立て =====================

def assemble(images, added_out, yolo_out):
    """
    画像ごと・ボックスごとの出力するSynset名を組み立てる．

    Args:
        images (list[dict]): build_images の戻り値．
        added_out (list[list[str]]): 画像ごとの加えた領域の出力（長さは画像の加えた領域の数以下．足りない分はYOLO以外を None）．
        yolo_out (list[list[str]]): 画像ごとのYOLO11xのボックスの出力．

    Returns:
        list[list[str]]: 画像ごと・ボックスごとの出力．
    """
    out = []
    for img, a, y in zip(images, added_out, yolo_out):
        row = list(y) + list(a) + ["physical_entity.n.01"] * (len(img["boxes"]) - img["n_yolo"] - len(a))
        out.append(row)
    return out


def yolo_classes(images):
    """
    YOLO11xのボックスにYOLO11xのクラスのSynsetを出力する（細分化しない）．

    Args:
        images (list[dict]): build_images の戻り値．

    Returns:
        list[list[str]]: 画像ごとのYOLO11xのボックスの出力．
    """
    return [[COCO_TO_SYNSET[b["label"]] for b in img["boxes"][:img["n_yolo"]]] for img in images]


def added_outputs(namer, images, beta, rule, taus, k_max):
    """
    全画像の加えた領域（先頭 k_max 件）の出力を，全ての τ について求める．

    Args:
        namer (Namer): 命名を行うオブジェクト．
        images (list[dict]): build_images の戻り値．
        beta (float): 温度の倍率．
        rule (str): 規則．
        taus (list[float]): 閾値．
        k_max (int): 命名する加えた領域の最大数．

    Returns:
        dict[float, list[list[str]]]: τ → 画像ごとの加えた領域の出力．
    """
    chunks = [img["feats"][img["n_yolo"]:img["n_yolo"] + k_max] for img in images]
    sizes = [len(c) for c in chunks]
    feats = torch.cat(chunks)
    flat = {t: [] for t in taus}
    for b in range(0, len(feats), 1024):  # 画像をまとめて計算する
        res = namer.added(feats[b:b + 1024], beta, rule, taus)
        for t in taus:
            flat[t] += res[t]
    out = {t: [] for t in taus}
    for t in taus:
        pos = 0
        for n in sizes:
            out[t].append(flat[t][pos:pos + n])
            pos += n
    return out


def yolo_outputs(namer, images, beta, rule, taus):
    """
    全画像のYOLO11xのボックスの細分化した出力を，全ての τ_y について求める．

    Args:
        namer (Namer): 命名を行うオブジェクト．
        images (list[dict]): build_images の戻り値．
        beta (float): 温度の倍率．
        rule (str): 規則．
        taus (list[float]): 閾値．

    Returns:
        dict[float, list[list[str]]]: τ_y → 画像ごとのYOLO11xのボックスの出力．
    """
    out = {t: [] for t in taus}
    for img in images:
        labels = [b["label"] for b in img["boxes"][:img["n_yolo"]]]
        res = namer.yolo(img["feats"][:img["n_yolo"]], labels, beta, rule, taus)
        for t in taus:
            out[t].append(res[t])
    return out


# ===================== 原因の切り分け =====================

def diagnosis(images, owl, cap_vocab, k_add=BASE_K):
    """
    加えた領域（先頭 k_add 件）のうちCOCO外の物体とIoU 0.5以上で重なるものについて，1位の語の正解率を語彙ごとに求める．

    Args:
        images (list[dict]): テスト用の build_images の戻り値．
        owl (OWLv2Regions): OWLv2．
        cap_vocab (list[dict]): キャプションの語彙．
        k_add (int): 加える領域の数．

    Returns:
        dict: n，lvis_top1_exact，lvis_top1_on_path，caption_top1_exact，caption_top1_on_path，caption_top1_finer_or_exact．
    """
    lvis = lvis_vocabulary()
    q_lvis = query_features(owl, lvis, "plain")
    q_cap = query_features(owl, cap_vocab, "plain")
    rows = []
    for img in images:
        idx = [j for j in range(img["n_yolo"], min(len(img["boxes"]), img["n_yolo"] + k_add))]
        if not idx or not len(img["gts"]):
            continue
        e, scale, _ = owl.region_embeddings(img["feats"][idx])
        top_l = (e @ q_lvis.T).argmax(dim=1).tolist()
        top_c = (e @ q_cap.T).argmax(dim=1).tolist()
        for k, j in enumerate(idx):
            g = int(np.argmax(img["iou"][j]))
            if img["iou"][j][g] < IOU_THRESHOLD or img["gts"][g]["kind"] != "noncoco":
                continue
            t = img["gts"][g]["synset"]
            pl, pc = lvis[top_l[k]]["synset"], cap_vocab[top_c[k]]["synset"]
            rows.append((pl == t, pl in ancestors(t) or t in ancestors(pl), pc == t, pc in ancestors(t) or t in ancestors(pc),
                         t in ancestors(pc)))
    arr = np.array(rows, dtype=float)
    keys = ["lvis_top1_exact", "lvis_top1_on_path", "caption_top1_exact", "caption_top1_on_path", "caption_top1_finer_or_exact"]
    return {"n": len(rows)} | {k: float(arr[:, i].mean()) for i, k in enumerate(keys)}


# ===================== 実行 =====================

def best_row(rows, keys):
    """
    fed_hF が最大の行の設定を返す．

    Args:
        rows (list[dict]): 設定と指標の行．
        keys (list[str]): 設定の列名．

    Returns:
        dict: 設定．
    """
    best = max(rows, key=lambda r: r["fed_hF"])
    return {k: best[k] for k in keys}


def save(name, split, seed, metrics, config):
    """
    1構成の指標と設定を保存する．

    Args:
        name (str): 構成名．
        split (str): 分割名．
        seed (int): シード．
        metrics (dict): 指標．
        config (dict): 設定．

    Returns:
        None
    """
    d = target_dir(name, split, seed)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def run_model(ctx, model_key, seed, settings=None):
    """
    1つのOWLv2の重みについて，S1〜S4を順に行う（settings を与えた場合は，その設定でテスト用のみを評価する）．

    Args:
        ctx (dict): load_context の戻り値．
        model_key (str): OWLv2の重みの名前．
        seed (int): シード．
        settings (dict | None): S4までに選んだ設定．

    Returns:
        tuple[list[dict], dict]: テスト用の結果の行と，選んだ設定．
    """
    owl = OWLv2Regions(model_key)
    images = {}
    for s in SPLITS:
        images[s] = build_images(ctx, s, region_cache(ctx, s, model_key, None))
        attach_checked(images[s], ctx["lvis"])
    cap = caption_vocabulary()
    rows, grids = [], {}
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)

    def record(stage, name, cfg, yolo_out_t, added_out_t, k_add=BASE_K, objectness=0.0, split_list=SPLITS, keep=None):
        res = {}
        for s in split_list:
            m = evaluate(images[s], assemble(images[s], added_out_t[s], yolo_out_t[s]), k_add, objectness,
                         keep if s == "test" else None)
            outputs = m.pop("outputs", None)
            save(f"{stage}_{name}", s, seed, m, cfg | {"model": model_key, "k_add": k_add, "objectness": objectness})
            res[s] = (m, outputs)
        if "test" in res:
            rows.append({"stage": stage, "name": name, "model": model_key} | cfg | {"k_add": k_add, "objectness": objectness}
                        | res["test"][0])
        return res

    if settings is None:
        # S1: 語の選び方（YOLO11xのボックスはクラスのみ）
        namer = Namer(owl, cap, "plain")
        y_cls = {s: yolo_classes(images[s]) for s in SPLITS}
        grid = []
        for rule, beta in tqdm(list(itertools.product(["flat", "deepest", "backoff", "greedy"], BETAS)), desc="S1", leave=False):
            taus = [0.0] if rule == "flat" else TAUS
            out = added_outputs(namer, images["dev"], beta, rule, taus, BASE_K)
            for t in taus:
                m = evaluate(images["dev"], assemble(images["dev"], out[t], y_cls["dev"]), BASE_K, 0.0)
                grid.append({"rule": rule, "beta": beta, "tau": t} | m)
        grids["S1"] = pd.DataFrame(grid)
        for rule in ["flat", "deepest", "backoff", "greedy"]:
            cfg = best_row([g for g in grid if g["rule"] == rule], ["rule", "beta", "tau"])
            out = {s: added_outputs(namer, images[s], cfg["beta"], rule, [cfg["tau"]], BASE_K)[cfg["tau"]] for s in SPLITS}
            record("S1", rule, cfg | {"text": "plain", "exclude_person": False}, y_cls, out)
        s1 = best_row(grid, ["rule", "beta", "tau"])
        # S2: 文章と語彙
        grid = []
        namers = {}
        for text, excl in itertools.product(["plain", "gloss", "ens"], [False, True]):
            namers[(text, excl)] = Namer(owl, vocab_variant(cap, excl), text)
            for beta in BETAS:
                out = added_outputs(namers[(text, excl)], images["dev"], beta, s1["rule"], TAUS, BASE_K)
                for t in TAUS:
                    m = evaluate(images["dev"], assemble(images["dev"], out[t], y_cls["dev"]), BASE_K, 0.0)
                    grid.append({"text": text, "exclude_person": excl, "beta": beta, "tau": t, "rule": s1["rule"]} | m)
        grids["S2"] = pd.DataFrame(grid)
        for text, excl in itertools.product(["plain", "gloss", "ens"], [False, True]):
            cfg = best_row([g for g in grid if g["text"] == text and g["exclude_person"] == excl],
                           ["rule", "beta", "tau", "text", "exclude_person"])
            out = {s: added_outputs(namers[(text, excl)], images[s], cfg["beta"], cfg["rule"], [cfg["tau"]], BASE_K)[cfg["tau"]]
                   for s in SPLITS}
            record("S2", f"{text}_{'noperson' if excl else 'all'}", cfg, y_cls, out)
        s2 = best_row(grid, ["rule", "beta", "tau", "text", "exclude_person"])
        namer = namers[(s2["text"], s2["exclude_person"])]
        added_s2 = {s: added_outputs(namer, images[s], s2["beta"], s2["rule"], [s2["tau"]], max(KS))[s2["tau"]] for s in SPLITS}
        # S3: YOLO11xのボックスの細分化
        y_t = {s: yolo_outputs(namer, images[s], s2["beta"], s2["rule"], TAUS_YOLO) for s in SPLITS}
        grid = []
        for t in TAUS_YOLO:
            m = evaluate(images["dev"], assemble(images["dev"], added_s2["dev"], y_t["dev"][t]), BASE_K, 0.0)
            grid.append({"tau_yolo": t} | m)
        grids["S3"] = pd.DataFrame(grid)
        tau_y = best_row(grid, ["tau_yolo"])["tau_yolo"]
        s3 = s2 | {"tau_yolo": tau_y}
        record("S3", "yolo_refine", s3, {s: y_t[s][tau_y] for s in SPLITS}, added_s2)
        # S4: 加える領域の数と棄却
        grid = []
        for k, o in itertools.product(KS, OBJECTNESS):
            m = evaluate(images["dev"], assemble(images["dev"], added_s2["dev"], y_t["dev"][tau_y]), k, o)
            grid.append({"k_add": k, "objectness": o} | m)
        grids["S4"] = pd.DataFrame(grid)
        s4 = best_row(grid, ["k_add", "objectness"])
        settings = s3 | s4
        os.makedirs(root, exist_ok=True)
        for stage, g in grids.items():
            g.to_csv(os.path.join(root, f"grid_{stage}_{model_key}.csv"), index=False)
        q_path = os.path.join(OUTPUT_DIR, "ex013_open_vocabulary_granular", "qualitative", "ex012.json")
        keep = set(json.load(open(q_path))) if os.path.exists(q_path) else None
        res = record("S4", "final", settings, {s: y_t[s][tau_y] for s in SPLITS}, added_s2, settings["k_add"],
                     settings["objectness"], keep=keep)
        # K を変えた場合の曲線（棄却の閾値は選んだ値）
        for k in KS:
            record("S4", f"k{k}", settings | {"k_add": k}, {s: y_t[s][tau_y] for s in SPLITS}, added_s2, k, settings["objectness"])
        if res["test"][1]:
            save_qualitative(ctx, res["test"][1], model_key)
    else:
        # S5: 同じ設定で別の重みを評価する
        namer = Namer(owl, vocab_variant(cap, settings["exclude_person"]), settings["text"])
        y = {s: yolo_outputs(namer, images[s], settings["beta"], settings["rule"], [settings["tau_yolo"]])[settings["tau_yolo"]]
             for s in SPLITS}
        a = {s: added_outputs(namer, images[s], settings["beta"], settings["rule"], [settings["tau"]],
                              settings["k_add"])[settings["tau"]] for s in SPLITS}
        record("S5", model_key, settings, y, a, settings["k_add"], settings["objectness"])
    del owl
    torch.cuda.empty_cache()
    return rows, settings


def save_qualitative(ctx, outputs, model_key):
    """
    定性評価の画像の出力を保存し，図を描く．

    Args:
        ctx (dict): load_context の戻り値．
        outputs (dict): 画像ID → {"box"，"source"，"synset"} のリスト．
        model_key (str): OWLv2の重みの名前．

    Returns:
        None
    """
    q_dir = os.path.join(OUTPUT_DIR, EXPERIMENT, "qualitative")
    os.makedirs(q_dir, exist_ok=True)
    for items in outputs.values():
        for it in items:
            it["word"] = wn.synset(it["synset"]).lemma_names()[0].replace("_", " ")
    with open(os.path.join(q_dir, f"final_{model_key}.json"), "w") as f:
        json.dump(outputs, f, indent=1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (image_id, items) in zip(axes.flat, sorted(outputs.items())):
        ax.imshow(load_image(ctx["coco"], int(image_id)))
        ax.axis("off")
        ax.set_title(f"image {image_id}", fontsize=9)
        for it in items:
            x0, y0, x1, y1 = it["box"]
            color = "tab:blue" if it["source"] == "yolo" else "tab:red"
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
            ax.text(x0, y0, it["word"], fontsize=6, color="white", va="top", bbox=dict(facecolor=color, alpha=0.7, pad=1))
    fig.tight_layout()
    fig.savefig(os.path.join(q_dir, f"grid_final_{model_key}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    """S0〜S5を実行し，集計表を作成する．"""
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    os.makedirs(root, exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        start = time.time()
        ctx = load_context(seed)
        cap = caption_vocabulary()
        # S0: 原因の切り分け
        diag = {}
        for key in ["st", "ens"]:
            if not os.path.exists(os.path.join(DATA_DIR, f"regions_{key}_test_n{len(ctx['splits']['test'])}.pt")):
                continue
            owl = OWLv2Regions(key)
            images = build_images(ctx, "test", region_cache(ctx, "test", key, None))
            diag[key] = diagnosis(images, owl, cap)
            print("S0", key, diag[key], flush=True)
            del owl
            torch.cuda.empty_cache()
        with open(os.path.join(root, "diagnosis.json"), "w") as f:
            json.dump(diag, f, indent=4)
        rows, settings = run_model(ctx, MAIN_MODEL, seed)
        print("selected settings", settings, flush=True)
        # 参考: LVIS baseでファインチューニングした重み（ens）と，より大きい重み（st_large）を同じ設定で評価する
        for key in ["ens", "st_large"]:
            if all(os.path.exists(os.path.join(DATA_DIR, f"regions_{key}_{s}_n{len(ctx['splits'][s])}.pt")) for s in SPLITS):
                r, _ = run_model(ctx, key, seed, settings)
                rows += r
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(root, "table_results.csv"), index=False)
        with open(os.path.join(root, "selected_settings.json"), "w") as f:
            json.dump(settings | {"elapsed_sec": time.time() - start}, f, indent=4)
        cols = ["stage", "name", "model", "fed_hF", "hF", "noncoco_detected", "noncoco_named", "noncoco_fine", "noncoco_hR",
                "coco_named", "coco_fine", "added_per_image", "added_precision"]
        print(table[cols].round(4).to_string(), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
