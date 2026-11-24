"""
ex013のモデル定義：検出の補完（OWLv2），WordNetの絞り込み，先読みによる下位語の選択，文章・画像埋め込みモデルの強化．

下位語のたどり方（descend）は，進捗報告2026Bのアルゴリズムと同じく，単語の集合から類似度が最大の単語を選び，
その下位語の集合に置き換えることを葉に達するまで繰り返す．各段階のコサイン類似度とソフトマックス（候補の中で正規化した確率）を
記録し，停止の判断は評価時に閾値を変えて行う．
- 絞り込み（pruned=True）: COCO train2017のキャプションに出現する語を MIN_VISIBLE 個以上含む枝と，
  キャプションに出現する深さ LEAF_DEPTH 以上の語のみを下位語として残す（役割を表す小さな枝を除く）
- 先読み（lookahead=True）: 候補の単語の得点を，自身とその下位語の類似度の最大値とする
- プロンプト: "a photo of a {w}"（plain），または "a photo of a {w}, a type of {親の語}"（hier）
"""
import functools
import os
import sys

import numpy as np
import open_clip
import torch
from nltk.corpus import wordnet as wn
from transformers import Owlv2ForObjectDetection, Owlv2Processor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.detectors import class_agnostic_nms  # noqa: E402
from common.paths import set_seed  # noqa: E402
from data import caption_terms  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDERS = {"SigLIP-B16": ("ViT-B-16-SigLIP", "webli"), "SigLIP-SO400M": ("ViT-SO400M-14-SigLIP-384", "webli")}
WORDNET_TOPS = ["person.n.01", "animal.n.01", "plant.n.02", "food.n.01", "artifact.n.01", "natural_object.n.01"]
MIN_VISIBLE = 3
LEAF_DEPTH = 8
OWL_MAX_BOXES = 100


class OWLv2Objectness:
    """OWLv2（google/owlv2-base-patch16-ensemble）の物体らしさのスコアによる，クラスに依存しない領域の検出．"""

    def __init__(self, checkpoint="google/owlv2-base-patch16-ensemble"):
        """
        Args:
            checkpoint (str): Hugging Face Hubのチェックポイント名．
        """
        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint).eval().to(DEVICE)

    @torch.no_grad()
    def detect(self, image):
        """
        1枚の画像の領域を検出する（IoU 0.5のNMSの後，スコアの上位 OWL_MAX_BOXES 件）．
        OWLv2は画像を右下に0埋めした正方形として扱うため，ボックスを長辺の長さで復元する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K）．
        """
        inputs = self.processor(text=[["a photo of an object"]], images=image, return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        scores = out.objectness_logits[0].sigmoid()
        cx, cy, w, h = (out.pred_boxes[0] * max(image.size)).T
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        limit = torch.tensor([image.size[0], image.size[1], image.size[0], image.size[1]], device=boxes.device)
        boxes = torch.minimum(boxes.clamp(min=0), limit)
        boxes, scores = class_agnostic_nms(boxes, scores, 0.5)
        order = np.argsort(-scores, kind="stable")[:OWL_MAX_BOXES]
        return {"boxes": boxes[order].tolist(), "scores": scores[order].tolist()}


class Embedder:
    """文章・画像埋め込みモデル F（open_clip の SigLIP）．"""

    def __init__(self, name, seed=0):
        """
        Args:
            name (str): EMBEDDERS のキー．
            seed (int): 乱数シード．
        """
        set_seed(seed)
        arch, pretrained = EMBEDDERS[name]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.eval().to(DEVICE)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.name = name
        self.scale = float(self.model.logit_scale.exp())
        self._text = {}

    @torch.no_grad()
    def image_features(self, images, batch_size=64):
        """
        画像の特徴ベクトルを計算する（SO400Mは画像エンコーダを半精度で計算する）．

        Args:
            images (list[PIL.Image.Image]): 画像（長さN）．
            batch_size (int): バッチサイズ．

        Returns:
            torch.Tensor: L2正規化した特徴ベクトル（N×D）．
        """
        feats = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]]).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.name == "SigLIP-SO400M"):
                feats.append(self.model.encode_image(x).float())
        feats = torch.cat(feats)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_features(self, texts):
        """
        文章の特徴ベクトルを計算する（文章の集合ごとにキャッシュする）．

        Args:
            texts (tuple[str]): 文章（長さV）．

        Returns:
            torch.Tensor: L2正規化した特徴ベクトル（V×D）．
        """
        if texts not in self._text:
            feats = torch.cat([self.model.encode_text(self.tokenizer(list(texts[i:i + 256])).to(DEVICE)).float()
                               for i in range(0, len(texts), 256)])
            self._text[texts] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text[texts]


def context_crop(image, box, scale=1.0):
    """
    ボックスの中心を保って縦横を scale 倍に広げた範囲を切り抜き，黒で正方形にパディングする．

    Args:
        image (PIL.Image.Image): 画像．
        box (list[float]): xyxy形式のボックス．
        scale (float): 拡大率．

    Returns:
        PIL.Image.Image: 切り抜き画像．
    """
    from PIL import Image

    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * scale, (y1 - y0) * scale
    x0, y0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x1, y1 = min(image.size[0], int(np.ceil(cx + w / 2))), min(image.size[1], int(np.ceil(cy + h / 2)))
    region = image.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))
    side = max(region.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(region, ((side - region.size[0]) // 2, (side - region.size[1]) // 2))
    return canvas


def is_visible(synset):
    """
    Synsetのいずれかのレンマが，COCO train2017のキャプションに出現するかを判定する．

    Args:
        synset (Synset): Synset．

    Returns:
        bool: 出現するか．
    """
    terms = caption_terms()
    return any(lemma.lower().replace("_", " ") in terms for lemma in synset.lemma_names())


@functools.lru_cache(maxsize=None)
def visible_count(synset_name):
    """
    Synsetとその全ての下位語のうち，キャプションに出現するSynsetの数を返す．

    Args:
        synset_name (str): Synset名．

    Returns:
        int: 数．
    """
    s = wn.synset(synset_name)
    return int(is_visible(s)) + sum(visible_count(h.name()) for h in s.hyponyms())


def word_of(synset, pruned):
    """
    Synsetをプロンプトに入れる単語に変換する（絞り込みでは，キャプションに出現するレンマを優先する）．

    Args:
        synset (Synset): Synset．
        pruned (bool): 絞り込みを行うか．

    Returns:
        str: 単語．
    """
    lemmas = [lemma.replace("_", " ") for lemma in synset.lemma_names()]
    if pruned:
        terms = caption_terms()
        return next((lemma for lemma in lemmas if lemma.lower() in terms), lemmas[0])
    return lemmas[0]


@functools.lru_cache(maxsize=None)
def children(synset_name, pruned):
    """
    Synsetの下位語の集合を返す（表記が重複する語は先頭のみ残す）．

    Args:
        synset_name (str): Synset名．
        pruned (bool): 絞り込みを行うか．

    Returns:
        tuple[tuple[str, str], ...]: (Synset名, 単語) の組．
    """
    out, words = [], set()
    for h in sorted(wn.synset(synset_name).hyponyms(), key=lambda s: s.name()):
        if pruned and not (visible_count(h.name()) >= MIN_VISIBLE or (is_visible(h) and h.min_depth() >= LEAF_DEPTH)):
            continue
        w = word_of(h, pruned)
        if w not in words:
            words.add(w)
            out.append((h.name(), w))
    return tuple(out)


def prompt_texts(words, parent_word, mode):
    """
    単語をテンプレートに埋め込んだ文章を返す．

    Args:
        words (list[str]): 単語．
        parent_word (str | None): 親の単語．
        mode (str): "plain" または "hier"．

    Returns:
        tuple[str, ...]: 文章．
    """
    if mode == "hier" and parent_word:
        return tuple(f"a photo of a {w}, a type of {parent_word}" for w in words)
    return tuple(f"a photo of a {w}" for w in words)


class Grandchildren:
    """先読みに用いる，ある単語の集合の各単語の下位語の文章の特徴ベクトルをまとめて保持する．"""

    def __init__(self, embedder, cands, pruned, mode):
        """
        Args:
            embedder (Embedder): F．
            cands (list[tuple[str, str]]): 候補の (Synset名, 単語)．
            pruned (bool): 絞り込みを行うか．
            mode (str): プロンプトの種類．
        """
        feats, owners = [], []
        for k, (name, word) in enumerate(cands):
            kids = children(name, pruned)
            if kids:
                feats.append(embedder.text_features(prompt_texts([w for _, w in kids], word, mode)))
                owners += [k] * len(kids)
        self.feats = torch.cat(feats) if feats else None
        self.owners = torch.tensor(owners, device=DEVICE) if owners else None
        self.n = len(cands)


_GRANDCHILDREN = {}


def grandchildren(embedder, cands, pruned, mode):
    """
    候補の集合に対する Grandchildren を返す（キャッシュする）．

    Args:
        embedder (Embedder): F．
        cands (list[tuple[str, str]]): 候補の (Synset名, 単語)．
        pruned (bool): 絞り込みを行うか．
        mode (str): プロンプトの種類．

    Returns:
        Grandchildren: 下位語の特徴ベクトル．
    """
    key = (id(embedder), tuple(c for c, _ in cands), pruned, mode)
    if key not in _GRANDCHILDREN:
        _GRANDCHILDREN[key] = Grandchildren(embedder, cands, pruned, mode)
    return _GRANDCHILDREN[key]


def descend(embedder, feature, start, pruned=False, lookahead=False, mode="plain", root=None):
    """
    単語の集合から類似度（先読みでは得点）が最大の単語を選び，その下位語の集合に置き換えることを葉に達するまで繰り返す．

    Args:
        embedder (Embedder): F．
        feature (torch.Tensor): 切り抜き画像の特徴ベクトル（D）．
        start (list[str]): 初期の語彙のSynset名（root が与えられた場合は用いない）．
        pruned (bool): 絞り込みを行うか．
        lookahead (bool): 先読みを行うか．
        mode (str): プロンプトの種類．
        root (str | None): 深さ0のSynset名（YOLO11xのクラス）．与えた場合，深さ0は選ばずにこのSynsetとする．

    Returns:
        list[dict]: 各段階の {"synset"，"word"，"cos"（選んだ単語のコサイン類似度），"soft"（候補の中でのソフトマックス），
            "n_candidates"}．root を与えた場合の深さ0は soft = 1．
    """
    steps = []
    if root is not None:
        word = word_of(wn.synset(root), pruned)
        cos = float(embedder.text_features(prompt_texts([word], None, "plain")) @ feature)
        steps.append({"synset": root, "word": word, "cos": cos, "soft": 1.0, "n_candidates": 1})
        cands = list(children(root, pruned))
    else:
        cands = [(s, word_of(wn.synset(s), pruned)) for s in start]
    while cands:
        parent_word = steps[-1]["word"] if steps else None
        cos = embedder.text_features(prompt_texts([w for _, w in cands], parent_word, mode)) @ feature
        score = cos
        if lookahead:
            gc = grandchildren(embedder, cands, pruned, mode)
            if gc.feats is not None:
                g = gc.feats @ feature
                best_g = torch.full((gc.n,), -1.0, device=DEVICE).scatter_reduce(0, gc.owners, g, reduce="amax")
                score = torch.maximum(cos, best_g)
        j = int(score.argmax())
        soft = float((embedder.scale * score).softmax(dim=-1)[j])
        steps.append({"synset": cands[j][0], "word": cands[j][1], "cos": float(cos[j]), "soft": soft,
                      "n_candidates": len(cands)})
        cands = list(children(cands[j][0], pruned))
    return steps
