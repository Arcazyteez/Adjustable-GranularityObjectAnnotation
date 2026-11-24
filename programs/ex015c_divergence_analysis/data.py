"""
ex015cのデータ：ex015bの data（ex015，ex015a を含む）を読み込み，テスト用の画像でCOCO外のGTと対応した加えた領域と，
その領域の語彙上の確率を集める（order_026）．

対応付けは ex015b の paired_naming と同じく，検出のスコアの順にGTへ対応付け，IoU 0.5以上でCOCO外のGTと対応した加えた領域を対象とする．
"""
import functools
import importlib.util
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX015B = "ex015b_baselines_robustness"


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
def ex015b_data():
    """
    ex015bの data を読み込む（ex015a，ex015 のモジュールは ex015b data の ex015a_modules から得る）．

    Returns:
        module: ex015bの data．
    """
    directory = os.path.join(PROGRAM_DIR, EX015B)
    saved = {n: sys.modules.pop(n) for n in ("data", "model") if n in sys.modules}
    sys.path.insert(0, directory)
    try:
        db = _load(os.path.join(directory, "data.py"), "ex015b_data")
    finally:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != directory]
        sys.modules.update(saved)
    return db


def modules():
    """
    ex015，ex015a のモジュールを返す．

    Returns:
        tuple: (ex015 data，ex015 train，ex015a data，ex015a model)．
    """
    da, ma, _ = ex015b_data().ex015a_modules()
    d15, _, t15 = da.ex015_modules()
    return d15, t15, da, ma


def matched_noncoco(images, records):
    """
    検出のスコアの順に出力をGTへ対応付け，COCO外のGTと対応した加えた領域を集める．

    Args:
        images (list[dict]): テスト用の画像（ex015a の load_images）．
        records (list[list[dict]]): 画像ごとの greedy の記録．

    Returns:
        list[dict]: 領域ごとの {"img"（画像の番号），"j"（出力の番号），"gt"（GTの番号），"gt_synset"，"rec"（記録）}．
    """
    _, _, _, ma = modules()
    out = []
    for i, (img, recs) in enumerate(zip(images, records)):
        matched = ma.match_greedy(img["iou"], np.array([r["det"] for r in recs]))
        for j, g in enumerate(matched):
            if recs[j]["source"] != "owlv2" or g < 0 or img["gts"][g]["kind"] != "noncoco":
                continue
            out.append({"img": i, "j": j, "gt": int(g), "gt_synset": img["gts"][g]["synset"], "rec": recs[j]})
    return out


def region_probs(namer, images, items, beta):
    """
    対象の領域ごとに，語彙上の確率と上位語へ集約した確率を求める．

    Args:
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        images (list[dict]): テスト用の画像．
        items (list[dict]): matched_noncoco の戻り値．
        beta (float): 温度の倍率．

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 語彙上の確率（N×V），Synsetの確率（N×Synset数）．
    """
    ps = []
    for it in tqdm(items, desc="probs", leave=False):
        ps.append(namer.probs(images[it["img"]]["feats"][it["j"]:it["j"] + 1], beta))
    p = torch.cat(ps) if ps else torch.zeros(0, len(namer.vocab))
    return p, p @ namer.hier.member
