"""
ex011のモデル定義：COCO外の物体の命名に用いるSigLIP，キャプションで絞り込んだWordNetの木，上から下への命名．

- VLM: SigLIP-ViT-B/16（ex009と同じ）と SigLIP SO400M（ViT-SO400M-14-SigLIP-384，webli）
- 木: WordNetの physical_entity.n.01 以下の名詞のうち，物体の部分でなく，いずれかのレンマがCOCO train2017の
      キャプションに出現するSynset（以下，語として使えるSynset）と，その上位語のみからなる木
- 上から下への命名: キャプションで頻出するカテゴリの語（出発点，animal，vehicle，food 等）とCOCOの80語の中から
  最初の語を選び，その後は木の子の中からロジットが最大の子を選び，葉に達するまでたどる．
  木の最上位（thing，object 等）から選ぶとVLMが抽象的な語を判別できないため，出発点を頻出するカテゴリの語とする．
  各段階のソフトマックス（兄弟の間での正規化）とシグモイドの確信度を記録し，停止は評価時に閾値で決める
- プロンプト: "a photo of a {語}"（plain），または上位語を含む "a photo of a {語}, a type of {親の語}"（hier）
"""
import functools
import json
import os
import sys

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import pad_to_square  # noqa: E402
from common.paths import set_seed  # noqa: E402
from common.vlm import SigLIPModel, load_vlm, vlm_logits  # noqa: E402
from common.wordnet_utils import hyponym_candidates  # noqa: E402
from data import DATA_DIR, caption_counts, caption_nouns, is_part_noun  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLMS = {"B16": "SigLIP-ViT-B16", "SO400M": "ViT-SO400M-14-SigLIP-384"}
TREE_ROOT = "physical_entity.n.01"
ROOT_MIN_COUNT = 300   # 出発点の語のキャプションでの出現回数の下限
ROOT_MIN_DEPTH = 5     # 出発点の語の最小の深さ（thing，object 等の抽象的な語を除く）
ROOT_MIN_DESCENDANTS = 3


def load_model(name, seed=0):
    """
    SigLIPをインスタンス化する．

    Args:
        name (str): "B16" または "SO400M"．
        seed (int): 乱数シード．

    Returns:
        SigLIPModel: VLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if name == "B16":
        return load_vlm(VLMS[name], DEVICE)
    vlm = SigLIPModel(arch=VLMS[name], pretrained="webli", device=DEVICE)
    vlm.name = "SigLIP-SO400M"
    return vlm


def context_crop(image, box, scale=1.0):
    """
    ボックスの中心を保って縦横を scale 倍に広げた範囲を切り出し，黒で正方形にパディングする．

    Args:
        image (PIL.Image.Image): 画像．
        box (list[float]): xyxy形式のボックス．
        scale (float): 拡大率（1.0は拡大しない）．

    Returns:
        PIL.Image.Image: 正方形のクロップ．
    """
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * scale, (y1 - y0) * scale
    x0, y0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x1, y1 = min(image.size[0], int(np.ceil(cx + w / 2))), min(image.size[1], int(np.ceil(cy + h / 2)))
    crop = image.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))
    return pad_to_square(crop)


def prompt_word(synset, captions):
    """
    Synsetをプロンプトに入れる語に変換する（キャプションに出現するレンマのうち先頭のもの，無ければ先頭のレンマ）．

    Args:
        synset (Synset): Synset．
        captions (frozenset[str]): キャプションの語の集合．

    Returns:
        str: 語．
    """
    lemmas = [lemma.replace("_", " ") for lemma in synset.lemma_names()]
    return next((lemma for lemma in lemmas if lemma.lower() in captions), lemmas[0])


@functools.lru_cache(maxsize=None)
def caption_tree():
    """
    キャプションで絞り込んだWordNetの木を作る（キャッシュする）．

    Returns:
        dict: "children"（Synset名 → 子のSynset名のリスト），"word"（Synset名 → プロンプトの語），
            "usable"（語として使えるSynset名のリスト）．
    """
    path = os.path.join(DATA_DIR, "caption_tree.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    captions = caption_nouns()
    root = wn.synset(TREE_ROOT)
    usable = []
    for s in root.closure(lambda x: x.hyponyms()):
        lemmas = [lemma.lower().replace("_", " ") for lemma in s.lemma_names()]
        if any(lemma in captions for lemma in lemmas) and not is_part_noun(s.name()):
            usable.append(s)
    keep = {TREE_ROOT}
    for s in usable:
        for path_ in s.hypernym_paths():
            if root in path_:
                keep.update(x.name() for x in path_[path_.index(root):])
    children = {name: sorted(h.name() for h in wn.synset(name).hyponyms() if h.name() in keep) for name in keep}
    tree = {"children": children, "word": {name: prompt_word(wn.synset(name), captions) for name in keep},
            "usable": sorted(s.name() for s in usable)}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tree, f)
    return tree


def prompts(words, parent_word, mode):
    """
    候補語のプロンプトに入れる文字列を返す（テンプレート "a photo of a {}" に埋め込まれる）．

    Args:
        words (list[str]): 候補語．
        parent_word (str | None): 親の語．
        mode (str): "plain" または "hier"．

    Returns:
        list[str]: 文字列のリスト．
    """
    if mode == "hier" and parent_word:
        return [f"{w}, a type of {parent_word}" for w in words]
    return list(words)


def select(vlm, feature, candidates, words, parent_word, mode):
    """
    候補の中からロジットが最大のものを選び，ソフトマックスとシグモイドの確信度を返す．

    Args:
        vlm (SigLIPModel): VLM．
        feature (torch.Tensor): クロップの画像特徴（D）．
        candidates (list[str]): 候補のSynset名．
        words (list[str]): 候補の語．
        parent_word (str | None): 親の語．
        mode (str): プロンプトの種類．

    Returns:
        tuple[int, float, float]: 選んだ候補の番号，ソフトマックスの確信度，シグモイドの確信度．
    """
    logits = vlm_logits(vlm, feature, prompts(words, parent_word, mode))
    best = int(logits.argmax())
    return best, float(logits.softmax(dim=-1)[best]), float(logits[best].sigmoid())


@functools.lru_cache(maxsize=None)
def top_down_roots():
    """
    上から下への命名の出発点となる語を返す．キャプションに ROOT_MIN_COUNT 回以上出現し，深さが ROOT_MIN_DEPTH 以上で，
    木の中に語として使える下位語を ROOT_MIN_DESCENDANTS 個以上持つSynsetと，COCOの80クラスのSynsetである
    （同じ語のSynsetは，語として使える下位語が最も多いもののみ残す）．

    Returns:
        list[dict]: {"synset"，"word"} のリスト．
    """
    from common.wordnet_utils import COCO_TO_SYNSET
    tree, counts = caption_tree(), caption_counts()
    usable = set(tree["usable"])
    best = {}
    for name in tree["usable"]:
        synset, word = wn.synset(name), tree["word"][name]
        if synset.min_depth() < ROOT_MIN_DEPTH or counts.get(word.lower(), 0) < ROOT_MIN_COUNT:
            continue
        n_desc = sum(1 for x in synset.closure(lambda y: y.hyponyms()) if x.name() in usable)
        if n_desc >= ROOT_MIN_DESCENDANTS and n_desc > best.get(word, ("", -1))[1]:
            best[word] = (name, n_desc)
    roots = {word: name for word, (name, _) in best.items()}
    roots |= {word: name for word, name in COCO_TO_SYNSET.items()}
    return [{"synset": name, "word": word} for word, name in sorted(roots.items())]


def top_down_chain(vlm, feature, mode="plain"):
    """
    出発点の語から1つを選び，キャプションで絞り込んだ木を葉までたどる（上から下への命名）．

    Args:
        vlm (SigLIPModel): VLM．
        feature (torch.Tensor): クロップの画像特徴（D）．
        mode (str): 深さ1以降のプロンプトの種類．

    Returns:
        list[dict]: 各段階の {"synset"，"word"，"soft"，"sig"，"n_candidates"}（先頭が出発点の語）．
    """
    tree, roots = caption_tree(), top_down_roots()
    best, soft, sig = select(vlm, feature, [r["synset"] for r in roots], [r["word"] for r in roots], None, "plain")
    node = roots[best]["synset"]
    chain = [{"synset": node, "word": roots[best]["word"], "soft": soft, "sig": sig, "n_candidates": len(roots)}]
    while tree["children"].get(node):
        cands = tree["children"][node]
        words = [tree["word"][c] for c in cands]
        best, soft, sig = select(vlm, feature, cands, words, chain[-1]["word"], mode)
        node = cands[best]
        chain.append({"synset": node, "word": words[best], "soft": soft, "sig": sig, "n_candidates": len(cands)})
    return chain


def flat_chain(vlm, feature, vocab, mode="plain"):
    """
    語彙から深さ0の語を選び，WordNetの下位語を最後までたどる（ex010と同じ命名）．

    Args:
        vlm (SigLIPModel): VLM．
        feature (torch.Tensor): クロップの画像特徴（D）．
        vocab (list[dict]): 深さ0の語彙．
        mode (str): 深さ1以降のプロンプトの種類．

    Returns:
        list[dict]: 各段階の {"synset"，"word"，"soft"，"sig"，"n_candidates"}（先頭が深さ0）．
    """
    best, soft, sig = select(vlm, feature, [v["synset"] for v in vocab], [v["word"] for v in vocab], None, "plain")
    chain = [{"synset": vocab[best]["synset"], "word": vocab[best]["word"], "soft": soft, "sig": sig,
              "n_candidates": len(vocab)}]
    return chain + hyponym_chain(vlm, feature, vocab[best]["synset"], vocab[best]["word"], mode)


def hyponym_chain(vlm, feature, synset_name, word, mode="plain"):
    """
    Synsetの下位語を，下位語の集合が空になるまでたどる（深さ1以降）．

    Args:
        vlm (SigLIPModel): VLM．
        feature (torch.Tensor): クロップの画像特徴（D）．
        synset_name (str): 出発点のSynset名．
        word (str): 出発点の語．
        mode (str): プロンプトの種類．

    Returns:
        list[dict]: 各段階の {"synset"，"word"，"soft"，"sig"，"n_candidates"}．
    """
    synset, chain = wn.synset(synset_name), []
    while True:
        words, synsets = hyponym_candidates(synset)
        if not words:
            return chain
        best, soft, sig = select(vlm, feature, [s.name() for s in synsets], words, word, mode)
        synset, word = synsets[best], words[best]
        chain.append({"synset": synset.name(), "word": word, "soft": soft, "sig": sig, "n_candidates": len(words)})
