"""
ex015のモデル定義：OWLv2の領域の特徴による命名と，WordNet上の確率の集約による粒度の決定．

- OWLv2Regions: OWLv2を1回適用し，物体らしさの上位の領域（ボックス，領域の特徴）を得る．領域の特徴と単語の文章の特徴から，
  OWLv2のクラスの判定と同じ式 l = (cos + shift) * scale でロジットを計算する
- 重み（CHECKPOINTS）:
    st:       google/owlv2-base-patch16（Web上の画像と文章の対による自己学習のみ．LVISの人手のアノテーションを用いていない）
    ens:      google/owlv2-base-patch16-ensemble（LVIS baseでファインチューニングした重みとの平均．参考）
    st_large: google/owlv2-large-patch14（自己学習のみのL/14）
- 単語の文章（TEXT_VARIANTS）: plain "a photo of a {w}"，gloss "{w}, {WordNetの定義文}"（16トークンで打ち切り），ens（両者の特徴の平均）
- Hierarchy: 語彙の各語の確率 p_v（ソフトマックス）をWordNetの上位語へ足し上げ P(n) = Σ_{v: n は v の上位語または自身} p_v とし，
  出力するSynsetを以下の規則で選ぶ
    deepest: P(n) ≥ τ のSynsetのうち最も深いもの（ex014の規則）
    backoff: 確率が最大の語から出発し，P(n) ≥ τ となるまで上位語へさかのぼる
    greedy:  起点から，集約した確率が最大の下位語へ進むことを繰り返し，下位語の確率が自身の語の確率以下，または τ 未満になったら止まる
"""
import os
import sys

import torch
from nltk.corpus import wordnet as wn
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import MAX_REGIONS, ROOT, ancestors  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS = {"st": "google/owlv2-base-patch16", "ens": "google/owlv2-base-patch16-ensemble",
               "st_large": "google/owlv2-large-patch14"}
TEXT_VARIANTS = ["plain", "gloss", "ens"]
RULES = ["deepest", "backoff", "greedy"]
MAX_TOKENS = 16  # OWLv2の文章エンコーダの最大トークン数


class OWLv2Regions:
    """OWLv2による領域の検出と，領域の特徴・文章の特徴の計算．"""

    def __init__(self, key):
        """
        Args:
            key (str): CHECKPOINTS のキー．
        """
        self.processor = Owlv2Processor.from_pretrained(CHECKPOINTS[key])
        self.model = Owlv2ForObjectDetection.from_pretrained(CHECKPOINTS[key]).eval().to(DEVICE)
        self.half = key.endswith("large")  # L/14は計算量が大きいため半精度で計算する
        self.key = key
        self._text = {}

    @torch.no_grad()
    def forward(self, image):
        """
        1枚の画像の全ての領域（パッチごとの予測）と，物体らしさの上位の領域の番号を返す．
        OWLv2は画像を右下に埋めた正方形として扱うため，ボックスを長辺の長さで復元する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 全ての領域のボックス（M×4，xyxy），
                物体らしさ（M），領域の特徴（M×D），IoU 0.5のNMSの後の物体らしさの上位 MAX_REGIONS 件の番号．
        """
        pixel = self.processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            feature_map, _ = self.model.image_embedder(pixel)
            b, h, w, d = feature_map.shape
            feats = feature_map.reshape(b, h * w, d)
            boxes = self.model.box_predictor(feats, feature_map)[0]
            scores = self.model.objectness_predictor(feats)[0].sigmoid()
        boxes, scores, feats = boxes.float(), scores.float(), feats[0].float()
        cx, cy, bw, bh = (boxes * max(image.size)).T
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)
        limit = torch.tensor([image.size[0], image.size[1], image.size[0], image.size[1]], device=DEVICE)
        xyxy = torch.minimum(xyxy.clamp(min=0), limit)
        keep = nms(xyxy, scores, 0.5)[:MAX_REGIONS]
        return xyxy, scores, feats, keep

    @torch.no_grad()
    def text_features(self, texts):
        """
        文章の特徴ベクトルを計算する（文章の集合ごとにキャッシュする）．

        Args:
            texts (tuple[str]): 文章（長さV）．MAX_TOKENS を超える部分は打ち切る．

        Returns:
            torch.Tensor: L2正規化した特徴ベクトル（V×E）．
        """
        if texts not in self._text:
            feats = []
            for i in range(0, len(texts), 512):
                tok = self.processor.tokenizer(list(texts[i:i + 512]), padding="max_length", truncation=True,
                                               max_length=MAX_TOKENS, return_tensors="pt").to(DEVICE)
                feats.append(self.model.owlv2.get_text_features(**tok).float())
            f = torch.cat(feats)
            self._text[texts] = f / f.norm(dim=-1, keepdim=True)
        return self._text[texts]

    @torch.no_grad()
    def region_embeddings(self, feats):
        """
        領域の特徴を，文章の特徴と比較する空間に写し，ロジットの scale と shift を計算する（OWLv2のクラスの判定と同じ計算）．

        Args:
            feats (torch.Tensor): 領域の特徴（N×D）．

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: L2正規化した埋め込み（N×E），scale（N），shift（N）．
        """
        head = self.model.class_head
        feats = feats.to(DEVICE).float()
        e = head.dense0(feats)
        e = e / (torch.linalg.norm(e, dim=-1, keepdim=True) + 1e-6)
        scale = head.elu(head.logit_scale(feats))[:, 0] + 1
        shift = head.logit_shift(feats)[:, 0]
        return e, scale, shift


def gloss_head(gloss):
    """
    定義文の最初の節（セミコロンまで）を返す．

    Args:
        gloss (str): WordNetの定義文．

    Returns:
        str: 定義文の先頭．
    """
    return gloss.split(";")[0].strip()


def query_features(owl, vocab, variant):
    """
    語彙の各語の文章の特徴を返す．

    Args:
        owl (OWLv2Regions): OWLv2．
        vocab (list[dict]): {"word"，"gloss"} のリスト（長さV）．
        variant (str): "plain"，"gloss"，"ens"．

    Returns:
        torch.Tensor: L2正規化した特徴ベクトル（V×E）．
    """
    plain = owl.text_features(tuple(f"a photo of a {v['word']}" for v in vocab))
    if variant == "plain":
        return plain
    gloss = owl.text_features(tuple(f"{v['word']}, {gloss_head(v['gloss'])}" for v in vocab))
    if variant == "gloss":
        return gloss
    f = plain + gloss
    return f / f.norm(dim=-1, keepdim=True)


class Hierarchy:
    """語彙の各語からWordNetの上位語への所属と，上位語・下位語の関係を保持し，出力するSynsetを選ぶ．"""

    def __init__(self, vocab):
        """
        Args:
            vocab (list[dict]): {"synset"} のリスト（長さV）．
        """
        nodes = sorted({a for v in vocab for a in ancestors(v["synset"])})
        index = {n: k for k, n in enumerate(nodes)}
        rows, cols = [], []
        for i, v in enumerate(vocab):
            for a in ancestors(v["synset"]):
                rows.append(i)
                cols.append(index[a])
        self.nodes, self.index = nodes, index
        self.member = torch.zeros(len(vocab), len(nodes), device=DEVICE)
        self.member[rows, cols] = 1.0
        # min_depth は上位語より下位語の方が小さくなる場合があるため，上位語より必ず大きい max_depth を用いる
        self.depth = torch.tensor([wn.synset(n).max_depth() for n in nodes], device=DEVICE, dtype=torch.float32)
        self.child = torch.zeros(len(nodes), len(nodes), dtype=torch.bool, device=DEVICE)
        pr, ch = [], []
        for n in nodes:
            for h in wn.synset(n).hyponyms():
                if h.name() in index:
                    pr.append(index[n])
                    ch.append(index[h.name()])
        self.child[pr, ch] = True
        # 各Synsetに対応する語彙の番号（語彙に無いSynsetは -1）
        self.word_of_node = torch.full((len(nodes),), -1, dtype=torch.long, device=DEVICE)
        self.word_of_node[[index[v["synset"]] for v in vocab]] = torch.arange(len(vocab), device=DEVICE)
        self.node_of_word = torch.tensor([index[v["synset"]] for v in vocab], device=DEVICE)
        self.root = index[ROOT]
        self.max_steps = int(self.depth.max().item()) + 1

    def under(self, synset_name):
        """
        指定したSynsetの下位語または自身であるSynsetのマスクを返す．

        Args:
            synset_name (str): Synset名．

        Returns:
            torch.Tensor: マスク（Synset数，bool）．
        """
        return torch.tensor([synset_name in ancestors(n) for n in self.nodes], device=DEVICE)

    def greedy_path(self, p_leaf, node_p, start):
        """
        起点から，集約した確率が最大の下位語へ進むことを繰り返した経路を返す．
        下位語の確率の最大値が自身の語の確率以下の場合は，その語で止まる．

        Args:
            p_leaf (torch.Tensor): 語彙上の確率（N×V）．
            node_p (torch.Tensor): Synsetの確率（N×Synset数）．
            start (torch.Tensor): 起点のSynsetの番号（N）．

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 経路上のSynsetの番号（N×L，経路が終わった後は最後のSynsetを繰り返す）と確率（N×L）．
        """
        n = len(p_leaf)
        rows = torch.arange(n, device=DEVICE)
        cur, active = start.clone(), torch.ones(n, dtype=torch.bool, device=DEVICE)
        path, probs = [cur], [node_p[rows, cur]]
        for _ in range(self.max_steps):
            child_p = torch.where(self.child[cur], node_p, torch.full_like(node_p, -1.0))
            best_p, best = child_p.max(dim=1)
            word = self.word_of_node[cur]
            self_p = torch.where(word >= 0, p_leaf[rows, word.clamp(min=0)], torch.zeros(n, device=DEVICE))
            active = active & (best_p > self_p) & (best_p > 0)
            if not active.any():
                break
            cur = torch.where(active, best, cur)
            path.append(cur)
            probs.append(node_p[rows, cur])
        return torch.stack(path, dim=1), torch.stack(probs, dim=1)

    def choose(self, p_leaf, taus, rule, start=None, allowed=None):
        """
        各領域について，規則に従って出力するSynsetを選ぶ．

        Args:
            p_leaf (torch.Tensor): 語彙上の確率（N×V，各行の和は1）．
            taus (list[float]): 閾値 τ のリスト．
            rule (str): "deepest"，"backoff"，"greedy"，"flat"（確率が最大の語をそのまま出力する．τ は用いない）．
            start (torch.Tensor | None): greedy の起点のSynsetの番号（N）．None は ROOT．
            allowed (torch.Tensor | None): 出力してよいSynsetのマスク（N×Synset数，bool）．None は全て．

        Returns:
            dict[float, tuple[torch.Tensor, torch.Tensor]]: τ →（選んだSynsetの番号（N），そのSynsetの確率（N））．
        """
        n = len(p_leaf)
        rows = torch.arange(n, device=DEVICE)
        node_p = p_leaf @ self.member
        out = {}
        if rule == "flat":
            top = self.node_of_word[p_leaf.argmax(dim=1)]
            return {tau: (top, node_p[rows, top]) for tau in taus}
        if rule == "greedy":
            start = torch.full((n,), self.root, device=DEVICE) if start is None else start
            path, probs = self.greedy_path(p_leaf, node_p, start)
            for tau in taus:
                ok = probs >= tau
                ok[:, 0] = True  # 起点は常に出力できる
                last = (ok.long().cumprod(dim=1).sum(dim=1) - 1).clamp(min=0)
                out[tau] = (path[rows, last], probs[rows, last])
            return out
        cand = torch.ones_like(node_p, dtype=torch.bool) if allowed is None else allowed
        if rule == "backoff":
            cand = cand & (self.member[p_leaf.argmax(dim=1)] > 0)
        for tau in taus:
            score = torch.where(cand & (node_p >= tau), self.depth + node_p, torch.full_like(node_p, -1.0))
            best = score.argmax(dim=1)
            out[tau] = (best, node_p[rows, best])
        return out
