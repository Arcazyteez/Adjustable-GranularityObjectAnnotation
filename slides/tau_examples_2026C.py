"""
スライド（slides_2026C.pptx）の「τ を変えると出力の粒度が変わる」例を作るスクリプト（order_044）．

ex020 の正解ボックスの特徴（Alpha-CLIP context1.5，test，キャッシュ済み）と WordNet∩VG の木から，
部分木の確率の合計による命名（式(2)）を τ ごとに再計算し，τ によって出力が細かくなる例を選んで JSON に保存する．
新しい学習は行わない（語彙の文章の特徴のみ文章エンコーダで求める）．

実行: .venv_pytorch/bin/python slides/tau_examples_2026C.py
出力: slides/media_2026C/tau_examples.json
"""
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX020 = os.path.join(ROOT, "programs", "ex020_coco_owl_alphaclip_granularity")
sys.path.insert(0, EX020)
import train as E20  # noqa: E402
from data import build_tree, eval_set  # noqa: E402
from model import TreeNamer  # noqa: E402

EMB = "AlphaCLIP-ViT-B16_context1.5"
TEMPERATURE = 1.0
TAUS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
OUT = os.path.join(ROOT, "slides", "media_2026C", "tau_examples.json")


def outputs_by_tau(namer, cos_row, taus, temperature):
    """
    1つの領域について，τ ごとの部分木の確率の合計による出力を求める．

    Args:
        namer (TreeNamer): WordNet∩VG の木の命名器．
        cos_row (np.ndarray): 領域と全ての語のコサイン類似度（長さW）．
        taus (list[float]): 閾値 τ の列．
        temperature (float): 温度 T．

    Returns:
        list[str | None]: τ ごとの出力のノード名（長さ len(taus)）．
    """
    return [namer.subtree_mass(cos_row[None, :], temperature, t)[0] for t in taus]


def is_refining(nodes, fine):
    """
    τ を下げるにつれて出力が同じ経路上で細かくなり，最後に細かい正解に一致するかを判定する．

    Args:
        nodes (list[str | None]): τ の降順に並べた出力のノード名．
        fine (str): 細かい正解の Synset 名．

    Returns:
        bool: 異なる語が3つ以上あり，全てが細かい正解の上位語（または正解自身）であれば True．
    """
    path = eval_set(fine) | {fine}
    words = [n for n in dict.fromkeys(nodes) if n is not None]
    return len(words) >= 3 and nodes[-1] == fine and all(n in path for n in words)


def main():
    """
    test の正解ボックスから，τ によって出力が細かくなる例を全て列挙して JSON に保存する．

    Returns:
        None
    """
    coco = E20.E19_DATA.load_coco()
    splits = E20.E19_DATA.load_splits(coco)
    nouns = E20.caption_nouns(coco, splits["test"], "test")
    instances = E20.fine_instances(coco, splits["test"], nouns)
    tree = build_tree()
    feats = torch.load(os.path.join(ROOT, "datasets", "ex019_coco_owl_alphaclip_wordnet", f"feat_gt_test_{EMB}.pt"),
                       weights_only=False)
    pool = E20.E19_TRAIN.EmbedderPool()
    namer = TreeNamer(pool.get(EMB), tree, [s for _, s in E20.E19_DATA.coco80_vocab()])
    examples = []
    for i, items in tqdm(instances.items(), desc="tau examples", leave=False):
        cos = namer.cosine(feats[i]["feats"])
        boxes, _ = E20.E19_DATA.gt_instances(coco, i)
        for g, cls, fines in items:
            if not fines:
                continue
            nodes = outputs_by_tau(namer, cos[g], TAUS, TEMPERATURE)
            fine = next((f for f in fines if is_refining(nodes, f)), None)
            if fine is None:
                continue
            examples.append({"image_id": int(i), "gt_index": int(g), "class": cls, "fine": fine,
                             "box": [float(v) for v in np.asarray(boxes[g])],
                             "outputs": [{"tau": t, "synset": n, "word": tree["words"].get(n) if n else None}
                                         for t, n in zip(TAUS, nodes)]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"embedder": EMB, "T": TEMPERATURE, "taus": TAUS, "examples": examples}, f, indent=1, ensure_ascii=False)
    print(len(examples), "examples ->", OUT)


if __name__ == "__main__":
    main()
