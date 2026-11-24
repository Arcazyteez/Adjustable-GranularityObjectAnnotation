"""
ex016のデータ：ex015bの data，model，train を読み込み，出力ごとに「物体の全体らしさ」の特徴を求める（order_025）．

特徴は再学習なしに，ボックスの位置関係とWordNetの部分と全体の関係（part meronym）のみから求める．
- 幾何の特徴：他のボックスに含まれる割合，内側に含む加えた領域の数，画像内で最大のボックスに対する面積の比
- WordNetの特徴：
  - meronym：その領域を含む大きなボックスの語の「部分」に，その領域の語が当たるか（例：car のボックス内の wheel）
  - part_word：その領域の語が何かの部分である語か（part_holonym を持つ，または part.n.01 等の下位語）
"""
import functools
import importlib.util
import os
import sys

import numpy as np
from nltk.corpus import wordnet as wn

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX015B = "ex015b_baselines_robustness"
CONTAIN = 0.8          # 領域の面積のうち他のボックスに含まれる割合がこれ以上であれば「含まれる」とする
AREA_RATIO = 1.5       # 含むボックスは含まれる領域よりこの倍率以上大きいものに限る
# 部分を表す語の上位語（これらの下位語は「何かの部分」を表す）
PART_ROOTS = frozenset({"part.n.01", "part.n.02", "part.n.03", "component.n.03", "body_part.n.01"})
# 人のボックス（COCOの person）の語に加える，人の体の部分の全体を表すSynset（hand → homo.n.02 等）
PERSON_ALIASES = frozenset({"homo.n.02", "body.n.01", "human_body.n.01", "person.n.01"})
WHOLENESS_NAMES = ["contain_yolo", "contain_any", "log_n_inside", "log_rel_area", "meronym", "part_word"]
GEOM_COLS = [0, 1, 2, 3]
WORDNET_COLS = [4, 5]


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
def ex015b_modules():
    """
    ex015bの data，model，train を読み込む．ex015bの model，train は data，model という名前でモジュールを読み込むため，
    読み込み中のみ sys.modules をex015bのものに置き換え，終了後に元に戻す．

    Returns:
        tuple[module, module, module]: ex015bの data，model，train．
    """
    directory = os.path.join(PROGRAM_DIR, EX015B)
    saved = {n: sys.modules.pop(n) for n in ("data", "model") if n in sys.modules}
    sys.path.insert(0, directory)
    try:
        db = _load(os.path.join(directory, "data.py"), "ex015b_data")
        sys.modules["data"] = db
        mb = _load(os.path.join(directory, "model.py"), "ex015b_model")
        sys.modules["model"] = mb
        tb = _load(os.path.join(directory, "train.py"), "ex015b_train")
    finally:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != directory]
        for n in ("data", "model"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return db, mb, tb


@functools.lru_cache(maxsize=None)
def holonyms_of(synset_name):
    """
    Synsetが部分となる全体のSynsetの集合を返す．Synset自身とその直接の上位語の part_holonym を推移的にたどる（キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: 全体を表すSynset名の集合．
    """
    s = wn.synset(synset_name)
    out, stack = set(), [h for x in [s] + s.hypernyms() for h in x.part_holonyms()]
    while stack:
        h = stack.pop()
        if h.name() in out:
            continue
        out.add(h.name())
        stack += h.part_holonyms()
    return frozenset(out)


@functools.lru_cache(maxsize=None)
def is_part_word(synset_name):
    """
    Synsetが何かの部分を表す語か（part_holonym を持つ，または PART_ROOTS の下位語）を返す（キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        bool: 部分を表す語であれば True．
    """
    if holonyms_of(synset_name):
        return True
    s = wn.synset(synset_name)
    ups = {x.name() for x in s.closure(lambda t: t.hypernyms())} | {synset_name}
    return bool(ups & PART_ROOTS)


@functools.lru_cache(maxsize=None)
def whole_set(synset_name):
    """
    ボックスの語が表す全体のSynsetの集合（自身と全ての上位語．人の場合は PERSON_ALIASES を加える）を返す（キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    out = set(ex015b_modules()[0].ex015a_modules()[0].ancestors_of(synset_name))
    if "person.n.01" in out:
        out |= PERSON_ALIASES
    return frozenset(out)


def is_meronym(part_syn, whole_syns):
    """
    part_syn が whole_syns のいずれかの語の部分であるかを返す．

    Args:
        part_syn (str): 含まれる領域の語．
        whole_syns (list[str]): 含むボックスの語（経路上の全ての語）．

    Returns:
        bool: 部分であれば True．
    """
    hol = holonyms_of(part_syn)
    return bool(hol) and any(hol & whole_set(w) for w in whole_syns)


def containment_matrix(boxes):
    """
    ボックスの組ごとに，ボックスjの面積のうちボックスiに含まれる割合 C[j, i] を求める．

    Args:
        boxes (np.ndarray): ボックス（N×4，x0，y0，x1，y1）．

    Returns:
        tuple[np.ndarray, np.ndarray]: C（N×N，対角は0），面積（N）．
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    w = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    c = w * h / np.maximum(area[:, None], 1e-9)
    np.fill_diagonal(c, 0.0)
    return c, area


def wholeness_features(img, recs):
    """
    1枚の画像の出力ごとに，物体の全体らしさの特徴を求める．YOLO11xのボックスは全て0とする（スコアは加えた領域のみ変える）．

    Args:
        img (dict): 画像（"boxes"，"n_yolo"）．
        recs (list[dict]): 出力の記録（"synset"，"chain"，"source"）．

    Returns:
        np.ndarray: 特徴（出力数×6，float32）．列は WHOLENESS_NAMES の順．
    """
    n = len(recs)
    out = np.zeros((n, len(WHOLENESS_NAMES)), dtype=np.float32)
    if n == 0:
        return out
    c, area = containment_matrix([b["box"] for b in img["boxes"]])
    larger = area[None, :] >= AREA_RATIO * area[:, None]   # larger[j, i]：iはjより十分大きい
    inside = (c >= CONTAIN) & larger                         # inside[j, i]：jはiに含まれる
    is_yolo = np.array([r["source"] == "yolo" for r in recs])
    max_area = max(area.max(), 1e-9)
    for j in np.where(~is_yolo)[0]:
        cy = c[j, is_yolo] * larger[j, is_yolo]
        out[j, 0] = cy.max() if cy.size else 0.0
        out[j, 1] = (c[j] * larger[j]).max()
        out[j, 2] = np.log1p((inside[:, j] & ~is_yolo).sum())
        out[j, 3] = np.log(max(area[j], 1e-9) / max_area)
        containers = np.where(inside[j])[0]
        out[j, 4] = float(any(is_meronym(recs[j]["synset"], [s for s, _ in recs[i]["chain"]]) for i in containers))
        out[j, 5] = float(is_part_word(recs[j]["synset"]))
    return out


def image_wholeness(images, records):
    """
    全画像の出力の全体らしさの特徴を，平坦化した順に並べる．

    Args:
        images (list[dict]): 画像．
        records (list[list[dict]]): 画像ごとの記録．

    Returns:
        np.ndarray: 特徴（全出力数×6）．
    """
    rows = [wholeness_features(img, recs) for img, recs in zip(images, records)]
    return np.concatenate(rows) if rows else np.zeros((0, len(WHOLENESS_NAMES)), dtype=np.float32)


def build_records(images, namers, settings):
    """
    greedy（提案）と flat（階層を用いない）の2つの命名の方法について，全画像の出力の記録を求める．

    Args:
        images (list[dict]): ex015a の load_images の戻り値．
        namers (dict): ex015b の make_namers の戻り値．
        settings (dict): ex015で選んだ設定．

    Returns:
        dict[str, list[list[dict]]]: 方法 → 画像ごとの記録．
    """
    db = ex015b_modules()[0]
    da = db.ex015a_modules()[0]
    base = da.build_records(images, namers["caption"], settings)
    flat = [db.method_records(b, namers["caption"], img, "flat", settings["beta"], settings["tau"])
            for b, img in zip(base, images)]
    return {"greedy": base, "flat": flat}
