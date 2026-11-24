"""
スライド（slides_2026C.pptx）の説明用の例を作るスクリプト（order_050）．

ex019・ex020 の正解ボックスの特徴（test，キャッシュ済み）から，次の2種類の例を求めて JSON に保存する．新しい学習は行わない．
  1. naming：同じ物体（Alpha-CLIP context1.5）に対する，前回の決め方（WordNet の下位語を1段ずつ下る，ex020 の保存済みの経路）と
     今回の決め方（WordNet∩VG の木で枝ごとに確率を足す，T = 1，τ = 0.05）の経路と確率
  2. classifier：COCO の80クラスに対する CLIP（ViT-B/32，切り抜き）と Alpha-CLIP（context1.5，crop）の上位3件の予測

実行: .venv_pytorch/bin/python slides/explain_examples_2026C.py
出力: slides/media_2026C/explain_examples.json
"""
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "programs", "ex020_coco_owl_alphaclip_granularity"))
import train as E20  # noqa: E402
from data import build_tree, eval_set  # noqa: E402
from model import TreeNamer  # noqa: E402

FEAT_DIR = os.path.join(ROOT, "datasets", "ex019_coco_owl_alphaclip_wordnet")
CHAIN_PATH = os.path.join(ROOT, "datasets", "ex020_coco_owl_alphaclip_granularity", "chains_gt_test_AlphaCLIP-ViT-B16_context1.5.json")
OUT = os.path.join(ROOT, "slides", "media_2026C", "explain_examples.json")
ALPHA = "AlphaCLIP-ViT-B16_context1.5"
CLASSIFIERS = ["CLIP-ViT-B32", "AlphaCLIP-ViT-B16_crop", "AlphaCLIP-ViT-B16_context1.5"]
T, TAU = 1.0, 0.05


def softmax(z):
    """
    最後の軸のソフトマックス．

    Args:
        z (np.ndarray): ロジット（B×K）．

    Returns:
        np.ndarray: 確率（B×K）．
    """
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def subtree_path(namer, cos_row):
    """
    今回の決め方の出力と，根から出力までの経路の各ノードの部分木の確率の合計を求める．

    Args:
        namer (TreeNamer): WordNet∩VG の木の命名器．
        cos_row (np.ndarray): 領域と全ての語のコサイン類似度（長さW）．

    Returns:
        list[dict]: 根に近い順の {"node"，"word"，"mass"}（枠のノードは除く．出力が無ければ空）．
    """
    mass = np.asarray((namer.subtree.T @ namer.softmax(cos_row[None, :], T).T).T)[0]
    out = namer.subtree_mass(cos_row[None, :], T, TAU)[0]
    pos = {n: j for j, n in enumerate(namer.nodes)}
    path, u = [], out
    while u is not None and u != "root":
        if not u.startswith("frame:"):
            path.append({"node": u, "word": namer.tree["words"][u], "mass": float(mass[pos[u]])})
        u = namer.tree["parent"].get(u)
    return path[::-1]


def is_correct(node, fine):
    """
    出力が細かい正解そのもの，またはその上位語（深さ4以上）であるかを判定する．

    Args:
        node (str | None): 出力の Synset 名．
        fine (str): 細かい正解の Synset 名．

    Returns:
        bool: 正しければ True．
    """
    return node is not None and (node == fine or node in eval_set(fine))


def topk(probs, names, k=3):
    """
    確率の上位 k 件を (名前, 確率) で返す．

    Args:
        probs (np.ndarray): 確率（長さ80）．
        names (list[str]): クラス名（長さ80）．
        k (int): 件数．

    Returns:
        list[list]: [名前, 確率] のリスト（長さ k）．
    """
    j = np.argsort(-probs)[:k]
    return [[names[i], float(probs[i])] for i in j]


def main():
    """
    説明用の例を全ての正解インスタンスについて求め，JSON に保存する．

    Returns:
        None
    """
    coco = E20.E19_DATA.load_coco()
    splits = E20.E19_DATA.load_splits(coco)
    nouns = E20.caption_nouns(coco, splits["test"], "test")
    instances = E20.fine_instances(coco, splits["test"], nouns)
    tree = build_tree()
    pool = E20.E19_TRAIN.EmbedderPool()
    namer = TreeNamer(pool.get(ALPHA), tree, [s for _, s in E20.E19_DATA.coco80_vocab()])
    chains = json.load(open(CHAIN_PATH))
    classes = [w for w, _ in E20.E19_DATA.coco80_vocab()]
    feats = {e: torch.load(os.path.join(FEAT_DIR, f"feat_gt_test_{e}.pt"), weights_only=False) for e in CLASSIFIERS}
    text = {e: pool.get(e).text_features(classes) for e in CLASSIFIERS}
    naming, classifier = [], []
    for i, items in tqdm(instances.items(), desc="examples", leave=False):
        boxes, labels = E20.E19_DATA.gt_instances(coco, i)
        cos = namer.cosine(feats[ALPHA][i]["feats"])
        probs = {e: softmax(100.0 * (feats[e][i]["feats"].float() @ text[e].T).numpy()) for e in CLASSIFIERS}
        for g, cls, fines in items:
            box = [float(v) for v in np.asarray(boxes[g])]
            classifier.append({"image_id": int(i), "gt_index": int(g), "class": cls, "box": box,
                               **{e: topk(probs[e][g], classes) for e in CLASSIFIERS}})
            if not fines:
                continue
            path = subtree_path(namer, cos[g])
            old = chains[str(i)]["chains"][g]
            fine = max(fines, key=lambda f: len(eval_set(f)))
            naming.append({"image_id": int(i), "gt_index": int(g), "class": cls, "fine": fine, "box": box,
                           "old": [{"word": c["word"], "synset": c["synset"], "score": c["score"], "n": c["n_candidates"],
                                    "correct": is_correct(c["synset"], fine)} for c in old],
                           "new": [{**p, "correct": is_correct(p["node"], fine)} for p in path]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"T": T, "tau": TAU, "naming": naming, "classifier": classifier}, open(OUT, "w"), ensure_ascii=False, indent=0)
    print(len(naming), "naming,", len(classifier), "classifier ->", OUT)


if __name__ == "__main__":
    main()
