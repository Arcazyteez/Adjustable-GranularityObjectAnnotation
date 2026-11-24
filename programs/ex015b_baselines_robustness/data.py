"""
ex015bのデータ：ex015aの data，model，train を読み込み，命名の方法（greedy，flat，および LVIS の語彙を与えた場合）ごとに
全ての出力の記録を求める（order_021）．YOLO11xのボックスの記録は全ての方法で共通とし，加えた領域の命名のみを変える．
"""
import functools
import importlib.util
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX015A = "ex015a_annotation_score"
# 命名の方法：（語彙，規則）．flat は確率が最大の1語をそのまま出力し，greedy は上位語へ集約して確信度に応じて止める
METHODS = {"greedy": ("caption", "greedy"), "flat": ("caption", "flat"),
           "greedy_lvis": ("lvis", "greedy"), "flat_lvis": ("lvis", "flat")}


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
def ex015a_modules():
    """
    ex015aの data，model，train を読み込む．ex015aの model，train は data，model という名前でモジュールを読み込むため，
    読み込み中のみ sys.modules をex015aのものに置き換え，終了後に元に戻す．

    Returns:
        tuple[module, module, module]: ex015aの data，model，train．
    """
    directory = os.path.join(PROGRAM_DIR, EX015A)
    saved = {n: sys.modules.pop(n) for n in ("data", "model") if n in sys.modules}
    sys.path.insert(0, directory)
    try:
        da = _load(os.path.join(directory, "data.py"), "ex015a_data")
        sys.modules["data"] = da
        ma = _load(os.path.join(directory, "model.py"), "ex015a_model")
        sys.modules["model"] = ma
        ta = _load(os.path.join(directory, "train.py"), "ex015a_train")
    finally:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != directory]
        for n in ("data", "model"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return da, ma, ta


def make_namers(owl, settings):
    """
    キャプションの語彙と LVIS のカテゴリの語彙について，命名を行うオブジェクトを作る．

    Args:
        owl (ex015 model.OWLv2Regions): OWLv2．
        settings (dict): ex015で選んだ設定．

    Returns:
        dict[str, ex015 train.Namer]: "caption"，"lvis" → 命名を行うオブジェクト．
    """
    d15, _, t15 = ex015a_modules()[0].ex015_modules()
    cap = t15.vocab_variant(d15.caption_vocabulary(), settings["exclude_person"])
    return {"caption": t15.Namer(owl, cap, settings["text"]), "lvis": t15.Namer(owl, d15.lvis_vocabulary(), settings["text"])}


def added_chains(namer, p, rule, tau):
    """
    加えた領域について，規則に従って経路（粗い語から出力する語までのSynsetと集約した確率）を求める．

    Args:
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        p (torch.Tensor): 語彙上の確率（N×V）．
        rule (str): "greedy" または "flat"（確率が最大の語のみを出力し，経路はその語のみとする）．
        tau (float): greedy の閾値．

    Returns:
        list[list[tuple[str, float]]]: 領域ごとの（Synset名，集約した確率）の列．
    """
    if rule == "greedy":
        return ex015a_modules()[0]._chains(namer, p, tau)
    hier = namer.hier
    node = hier.node_of_word[p.argmax(dim=1)]
    q = (p @ hier.member)[torch.arange(len(p), device=p.device), node]
    return [[(hier.nodes[int(k)], float(v))] for k, v in zip(node.tolist(), q.tolist())]


def method_records(base_recs, namer, img, rule, beta, tau):
    """
    1枚の画像について，YOLO11xのボックスの記録は base_recs のまま，加えた領域の記録を指定した方法で求め直す．

    Args:
        base_recs (list[dict]): ex015a の image_records の戻り値（キャプションの語彙と greedy）．
        namer (ex015 train.Namer): 加えた領域の命名に用いるオブジェクト．
        img (dict): ex015a の load_images の要素．
        rule (str): "greedy" または "flat"．
        beta (float): 温度の倍率．
        tau (float): greedy の閾値．

    Returns:
        list[dict]: 出力ごとの記録（ex015a の image_records と同じ形式）．
    """
    da = ex015a_modules()[0]
    n_yolo = img["n_yolo"]
    if len(img["boxes"]) <= n_yolo:
        return list(base_recs)
    p = namer.probs(img["feats"][n_yolo:], beta)
    pmax = p.max(dim=1).values.tolist()
    ent = (-(p * (p + 1e-12).log()).sum(dim=1) / np.log(p.shape[1])).tolist()
    recs = list(base_recs[:n_yolo])
    for k, chain in enumerate(added_chains(namer, p, rule, tau)):
        syn, q = chain[-1]
        b = img["boxes"][n_yolo + k]
        recs.append({"synset": syn, "chain": chain, "source": b["source"], "det": float(b["score"]), "q": q,
                     "pmax": pmax[k], "entropy": ent[k], "depth": da.max_depth(syn)})
    return recs


def build_all_records(images, namers, settings):
    """
    全ての命名の方法について，全画像の出力の記録を求める．

    Args:
        images (list[dict]): ex015a の load_images の戻り値．
        namers (dict): make_namers の戻り値．
        settings (dict): ex015で選んだ設定．

    Returns:
        dict[str, list[list[dict]]]: 方法 → 画像ごとの記録．
    """
    da = ex015a_modules()[0]
    base = da.build_records(images, namers["caption"], settings)
    out = {"greedy": base}
    for name, (vocab, rule) in METHODS.items():
        if name == "greedy":
            continue
        out[name] = [method_records(b, namers[vocab], img, rule, settings["beta"], settings["tau"])
                     for b, img in tqdm(list(zip(base, images)), desc=name, leave=False)]
    return out
