"""
ex014のモデル定義：下から確率を集める命名（提案1），OWLv2による命名（提案2），文章の工夫（提案3）．

- 提案1（hierarchical_choice）: 語彙 V の各語の確率 p_v をソフトマックスで求め，WordNetの各Synset n の確率を
  P(n) = Σ_{v ∈ V, n は v の上位語または自身} p_v とする．P(n) ≥ τ のSynsetのうち最も深いもの（max_depth が最大，同じ深さでは P が最大）を出力する
- 提案2（OWLv2Regions）: OWLv2の領域の特徴と単語の文章の特徴から，OWLv2のクラスの判定と同じ式で領域ごとのロジットを求める
- 提案3（text_features）: 複数のテンプレートと定義文を用いた文章の特徴の平均を単語の特徴とする
"""
import functools
import os
import sys

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import ex013_modules  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPLATES = ["a photo of a {w}", "a photo of the {w}", "a close-up photo of a {w}", "a {w}, which is {g}"]


def gloss_head(gloss):
    """
    定義文の最初の節（セミコロンまで）を返す．

    Args:
        gloss (str): WordNetの定義文．

    Returns:
        str: 定義文の先頭．
    """
    return gloss.split(";")[0].strip()


@torch.no_grad()
def text_features(embedder, vocab, ensemble=False):
    """
    語彙の各語の文章の特徴ベクトルを返す（ensemble=True は複数のテンプレートと定義文の平均）．

    Args:
        embedder (ex013 の Embedder): SigLIP．
        vocab (list[dict]): {"word"，"gloss"} のリスト（長さV）．
        ensemble (bool): 文章の工夫を行うか．

    Returns:
        torch.Tensor: L2正規化した特徴ベクトル（V×D）．
    """
    templates = TEMPLATES if ensemble else TEMPLATES[:1]
    feats = 0
    for t in templates:
        texts = tuple(t.format(w=v["word"], g=gloss_head(v["gloss"])) for v in vocab)
        feats = feats + embedder.text_features(texts)
    feats = feats / len(templates)
    return feats / feats.norm(dim=-1, keepdim=True)


class Hierarchy:
    """語彙の各語からWordNetの上位語への所属を表す行列と，各Synsetの深さを保持する．"""

    def __init__(self, vocab):
        """
        Args:
            vocab (list[dict]): {"synset"} のリスト（長さV）．
        """
        d13 = ex013_modules()[0]
        nodes = sorted({a for v in vocab for a in d13.ancestors(v["synset"])})
        index = {n: k for k, n in enumerate(nodes)}
        rows, cols = [], []
        for i, v in enumerate(vocab):
            for a in d13.ancestors(v["synset"]):
                rows.append(i)
                cols.append(index[a])
        self.nodes = nodes
        self.member = torch.zeros(len(vocab), len(nodes), device=DEVICE)
        self.member[rows, cols] = 1.0
        # min_depth は上位語より下位語の方が小さくなる場合がある（dog と canine 等）ため，上位語より必ず大きい max_depth を用いる
        self.depth = torch.tensor([wn.synset(n).max_depth() for n in nodes], device=DEVICE, dtype=torch.float32)

    def choose(self, probs, tau):
        """
        各領域について，確率が τ 以上のSynsetのうち最も深いもの（同じ深さでは確率が最大）を返す．

        Args:
            probs (torch.Tensor): 語彙上の確率（N×V）．
            tau (float): 閾値．

        Returns:
            list[tuple[str, float]]: (Synset名, そのSynsetの確率) のリスト（長さN）．
        """
        node_p = probs @ self.member
        score = torch.where(node_p >= tau, self.depth + node_p, torch.full_like(node_p, -1.0))
        best = score.argmax(dim=1)
        return [(self.nodes[int(b)], float(node_p[i, b])) for i, b in enumerate(best)]


class OWLv2Regions:
    """OWLv2（google/owlv2-base-patch16-ensemble）の領域の検出と，領域の特徴によるクラスのロジットの計算．"""

    def __init__(self, checkpoint="google/owlv2-base-patch16-ensemble"):
        """
        Args:
            checkpoint (str): Hugging Face Hubのチェックポイント名．
        """
        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint).eval().to(DEVICE)
        self._text = {}

    @torch.no_grad()
    def regions(self, image, max_boxes=100):
        """
        ex013の OWLv2Objectness と同じ手順で領域を検出し，領域の特徴も返す．

        Args:
            image (PIL.Image.Image): 入力画像．
            max_boxes (int): 残す領域の数．

        Returns:
            tuple[list[list[float]], torch.Tensor]: xyxy形式のボックス（スコアの降順）と，領域の特徴（K×768）．
        """
        pixel = self.processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        feature_map, _ = self.model.image_embedder(pixel)
        b, h, w, d = feature_map.shape
        feats = feature_map.reshape(b, h * w, d)
        boxes = self.model.box_predictor(feats, feature_map)[0]
        scores = self.model.objectness_predictor(feats)[0].sigmoid()
        cx, cy, bw, bh = (boxes * max(image.size)).T
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)
        limit = torch.tensor([image.size[0], image.size[1], image.size[0], image.size[1]], device=DEVICE)
        xyxy = torch.minimum(xyxy.clamp(min=0), limit)
        keep = nms(xyxy.float(), scores.float(), 0.5)[:max_boxes]
        return xyxy[keep].tolist(), feats[0, keep]

    @torch.no_grad()
    def query_features(self, vocab):
        """
        語彙の各語の文章「a photo of a {w}」のOWLv2の文章の特徴を返す（キャッシュする）．

        Args:
            vocab (list[dict]): {"word"} のリスト．

        Returns:
            torch.Tensor: 文章の特徴（V×512）．
        """
        key = tuple(v["word"] for v in vocab)
        if key not in self._text:
            feats = []
            for i in range(0, len(key), 512):
                tok = self.processor.tokenizer([f"a photo of a {w}" for w in key[i:i + 512]], padding="max_length",
                                               truncation=True, max_length=16, return_tensors="pt").to(DEVICE)
                feats.append(self.model.owlv2.get_text_features(**tok))
            self._text[key] = torch.cat(feats)
        return self._text[key]

    @torch.no_grad()
    def logits(self, region_feats, vocab):
        """
        領域の特徴と語彙の文章の特徴から，OWLv2のクラスの判定と同じ式でロジットを計算する．

        Args:
            region_feats (torch.Tensor): 領域の特徴（N×768）．
            vocab (list[dict]): 語彙．

        Returns:
            torch.Tensor: ロジット（N×V）．
        """
        if len(region_feats) == 0:
            return torch.zeros(0, len(vocab), device=DEVICE)
        query = self.query_features(vocab)
        out = self.model.class_head(region_feats[None], query[None], None)[0][0]
        return out.float()
