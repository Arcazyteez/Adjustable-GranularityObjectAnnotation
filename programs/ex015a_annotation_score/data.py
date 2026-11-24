"""
ex015aのデータ：ex015の data，model，train を読み込み，ex015で選んだ設定で全ての出力（YOLO11xのボックスと加えた領域）の
単語，経路上の各単語の確率，およびスコアの計算に用いる特徴を求める（order_020）．
"""
import functools
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from tqdm import tqdm

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX015 = "ex015_lvis_open_vocabulary_annotation"
K_ADD = 100  # ex015で選んだ加える領域の数（スコアで棄却するため，物体らしさの下限は設けない）


def _load(path, name):
    """
    ファイルからモジュールを読み込む．

    Args:
        path (str): ファイルのパス．
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def ex015_modules():
    """
    ex015の data，model，train を読み込む．ex015の model，train は data，model という名前でモジュールを読み込むため，
    読み込み中のみ sys.modules をex015のものに置き換え，終了後に元に戻す．

    Returns:
        tuple[module, module, module]: ex015の data，model，train．
    """
    directory = os.path.join(PROGRAM_DIR, EX015)
    saved = {n: sys.modules.pop(n) for n in ("data", "model") if n in sys.modules}
    sys.path.insert(0, directory)
    try:
        d15 = _load(os.path.join(directory, "data.py"), "ex015_data")
        sys.modules["data"] = d15
        m15 = _load(os.path.join(directory, "model.py"), "ex015_model")
        sys.modules["model"] = m15
        t15 = _load(os.path.join(directory, "train.py"), "ex015_train")
    finally:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != directory]
        for n in ("data", "model"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return d15, m15, t15


def selected_settings():
    """
    ex015で検証用の画像により選んだ設定を返す．

    Returns:
        dict: rule，beta，tau，text，exclude_person 等．
    """
    t15 = ex015_modules()[2]
    with open(os.path.join(t15.OUTPUT_DIR, EX015, "selected_settings.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def evaluated_set(synset_name):
    """
    評価に用いるSynsetの集合（ex013と同じ定義，キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    return ex015_modules()[0].ex013_modules()[2].evaluated_set(synset_name)


@functools.lru_cache(maxsize=None)
def ancestors_of(synset_name):
    """
    Synset自身と全ての上位語の集合（ex013と同じ定義，キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    return ex015_modules()[0].ancestors(synset_name)


@functools.lru_cache(maxsize=None)
def max_depth(synset_name):
    """
    Synsetの max_depth を返す（キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        int: 深さ．
    """
    return wn.synset(synset_name).max_depth()


def load_images(ctx, split, model_key):
    """
    ex015と同じ出力ボックスの候補とGTを読み込み，federated な評価に用いる情報を付ける．

    Args:
        ctx (dict): ex015の load_context の戻り値．
        split (str): 分割名．
        model_key (str): OWLv2の重みの名前．

    Returns:
        list[dict]: ex015の build_images の戻り値（加えた領域は先頭 K_ADD 件に限る）．
    """
    d15, _, t15 = ex015_modules()
    images = d15.build_images(ctx, split, d15.region_cache(ctx, split, model_key, None))
    t15.attach_checked(images, ctx["lvis"])
    for img in images:
        n = min(len(img["boxes"]), img["n_yolo"] + K_ADD)
        img["boxes"], img["feats"], img["iou"] = img["boxes"][:n], img["feats"][:n], img["iou"][:n]
    return images


def _chains(namer, p, tau):
    """
    加えた領域について，上位語から貪欲に進んだ経路のうち出力する単語までの各Synsetと集約した確率を求める．

    Args:
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        p (torch.Tensor): 語彙上の確率（N×V）．
        tau (float): 閾値．

    Returns:
        list[list[tuple[str, float]]]: 領域ごとの（Synset名，集約した確率）の列（根を除く）．
    """
    hier = namer.hier
    node_p = p @ hier.member
    start = torch.full((len(p),), hier.root, device=p.device)
    path, probs = hier.greedy_path(p, node_p, start)
    ok = probs >= tau
    ok[:, 0] = True
    last = (ok.long().cumprod(dim=1).sum(dim=1) - 1).clamp(min=0).tolist()
    path, probs = path.tolist(), probs.tolist()
    out = []
    for r in range(len(p)):
        # greedy_path は進めなくなった後も最後のSynsetを繰り返すため，同じSynsetが続く部分を除く
        ks = [k for k in range(1, last[r] + 1) if path[r][k] != path[r][k - 1]]
        out.append([(hier.nodes[path[r][k]], probs[r][k]) for k in ks] or [(hier.nodes[path[r][0]], probs[r][0])])
    return out


def image_records(namer, img, beta, tau):
    """
    1枚の画像の全ての出力について，単語，経路上の確率，スコアの特徴を求める．

    Args:
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        img (dict): load_images の要素．
        beta (float): 温度の倍率．
        tau (float): 閾値．

    Returns:
        list[dict]: 出力ごとの {"synset"，"chain"，"source"，"det"（検出のスコア），"q"（出力した単語の集約した確率），
            "pmax"（語彙上の確率の最大値），"entropy"（正規化したエントロピー），"depth"（max_depth）}．
    """
    from common.wordnet_utils import COCO_TO_SYNSET

    if len(img["boxes"]) == 0:
        return []
    p = namer.probs(img["feats"], beta)
    pmax = p.max(dim=1).values.tolist()
    ent = (-(p * (p + 1e-12).log()).sum(dim=1) / math.log(p.shape[1])).tolist()
    node_p = p @ namer.hier.member
    n_yolo = img["n_yolo"]
    chains = _chains(namer, p[n_yolo:], tau) if len(p) > n_yolo else []
    recs = []
    for j, b in enumerate(img["boxes"]):
        if j < n_yolo:
            syn = COCO_TO_SYNSET[b["label"]]
            k = namer.hier.index.get(syn)
            q = float(node_p[j, k]) if k is not None else float("nan")
            chain = [(syn, q)]
        else:
            chain = chains[j - n_yolo]
            syn, q = chain[-1]
        recs.append({"synset": syn, "chain": chain, "source": b["source"], "det": float(b["score"]), "q": q,
                     "pmax": pmax[j], "entropy": ent[j], "depth": max_depth(syn)})
    return recs


def build_records(images, namer, settings):
    """
    全画像の出力の記録を求める．

    Args:
        images (list[dict]): load_images の戻り値．
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        settings (dict): selected_settings の戻り値．

    Returns:
        list[list[dict]]: 画像ごとの image_records の戻り値．
    """
    return [image_records(namer, img, settings["beta"], settings["tau"])
            for img in tqdm(images, desc="records", leave=False)]


def features(recs):
    """
    スコアの校正に用いる特徴を並べる．

    Args:
        recs (list[dict]): 出力の記録（平坦化したもの）．

    Returns:
        np.ndarray: 特徴（N×F，float32）．列は [logit(det)，log q，log pmax，entropy，depth/10，YOLOか，YOLOか×logit(det)，YOLOか×log q]．
    """
    det = np.clip(np.array([r["det"] for r in recs], dtype=np.float64), 1e-4, 1 - 1e-4)
    q = np.array([r["q"] for r in recs], dtype=np.float64)
    q = np.clip(np.nan_to_num(q, nan=1.0), 1e-6, 1.0)
    yolo = np.array([r["source"] == "yolo" for r in recs], dtype=np.float64)
    logit_det = np.log(det / (1 - det))
    cols = [logit_det, np.log(q), np.log(np.array([r["pmax"] for r in recs]) + 1e-12), np.array([r["entropy"] for r in recs]),
            np.array([r["depth"] for r in recs]) / 10.0, yolo, yolo * logit_det, yolo * np.log(q)]
    return np.stack(cols, axis=1).astype(np.float32)
