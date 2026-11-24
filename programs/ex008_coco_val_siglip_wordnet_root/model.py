"""
ex008のモデル定義：CLIP-ViT-B/32（ソフトマックス）とSigLIP-ViT-B/16（シグモイド）によるカテゴリのアノテーション．

VLMは候補語彙 V 上のロジット l_v = t * cos(x, w_v) + b を計算する（CLIPは b = 0）．
- ソフトマックス: p_v = exp(l_v) / Σ_u exp(l_u)（候補語彙の中で正規化する，論文の式(2)）
- シグモイド:     p_v = σ(l_v) = 1 / (1 + exp(-l_v))（語ごとに独立，SigLIPの学習時の目的関数と同じ形）
argmax はどちらも同じであるため，WordNetの下位語の展開で選ばれる語は同じで，確信度のみが異なる．
"""
import os
import sys

import open_clip
import torch
from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import set_seed  # noqa: E402
from common.vlm import load_vlm  # noqa: E402
from common.wordnet_utils import hyponym_candidates  # noqa: E402
from data import load_ex007  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPLATE = "a photo of a {}"
# VLMの名前 → 使用できる正規化関数
VLM_NORMALIZATIONS = {"CLIP-ViT-B32": ["softmax"], "SigLIP-ViT-B16": ["softmax", "sigmoid"]}


class SigLIPModel:
    """open_clipによるSigLIP（ViT-B-16-SigLIP，webli）のゼロショット分類．"""

    def __init__(self, arch="ViT-B-16-SigLIP", pretrained="webli", device="cuda", template=TEMPLATE):
        """
        モデルを読み込む．

        Args:
            arch (str): open_clipのアーキテクチャ名．
            pretrained (str): 学習済みの重みの名前．
            device (str): 推論デバイス．
            template (str): プロンプトのテンプレート．
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.name = "SigLIP-ViT-B16"
        self.device = device
        self.template = template
        self._text_cache = {}

    @torch.no_grad()
    def text_features(self, vocab, batch_size=256):
        """
        候補語彙のテキスト特徴を計算する（語彙ごとにキャッシュする）．

        Args:
            vocab (list[str]): 候補語彙（長さV）．
            batch_size (int): バッチサイズ．

        Returns:
            torch.Tensor: L2正規化済みテキスト特徴（V×D）．
        """
        key = tuple(vocab)
        if key not in self._text_cache:
            prompts = [self.template.format(w.replace("_", " ")) for w in vocab]
            feats = []
            for i in range(0, len(prompts), batch_size):
                tokens = self.tokenizer(prompts[i:i + batch_size]).to(self.device)
                feats.append(self.model.encode_text(tokens).float())
            feats = torch.cat(feats)
            self._text_cache[key] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text_cache[key]

    @torch.no_grad()
    def image_features(self, images, batch_size=64):
        """
        画像特徴を計算する．

        Args:
            images (list[PIL.Image.Image]): 画像のリスト（長さN）．
            batch_size (int): バッチサイズ．

        Returns:
            torch.Tensor: L2正規化済み画像特徴（N×D）．
        """
        feats = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]]).to(self.device)
            feats.append(self.model.encode_image(x).float())
        feats = torch.cat(feats)
        return feats / feats.norm(dim=-1, keepdim=True)


def load_model(name, seed=0):
    """
    カテゴリのアノテーションに用いるVLMをインスタンス化する．

    Args:
        name (str): "CLIP-ViT-B32" または "SigLIP-ViT-B16"．
        seed (int): 乱数シード．

    Returns:
        CLIPModel | SigLIPModel: image_features，text_features，model（logit_scale，logit_bias）を持つVLM．
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if name == "SigLIP-ViT-B16":
        return SigLIPModel(device=DEVICE)
    return load_vlm(name, DEVICE)


def load_detr(seed=0):
    """
    ex007と同じDETR（ResNet-50，閾値0.5）をインスタンス化する．

    Args:
        seed (int): 乱数シード．

    Returns:
        DETRDetector: detect(image) を持つ検出器．
    """
    return load_ex007()[1].load_detr(seed)


def load_yolo(seed=0):
    """
    ex007と同じYOLO11x（閾値0.25）をインスタンス化する．

    Args:
        seed (int): 乱数シード．

    Returns:
        YOLODetector: detect(image) を持つ検出器．
    """
    return load_ex007()[1].load_yolo(seed)


@torch.no_grad()
def logits_from_features(vlm, image_features, vocab):
    """
    画像特徴と候補語彙のロジット l_v = t * cos(x, w_v) + b を計算する．

    Args:
        vlm (CLIPModel | SigLIPModel): VLM．
        image_features (torch.Tensor): L2正規化済みの画像特徴（D または N×D）．
        vocab (list[str]): 候補語彙（長さV）．

    Returns:
        torch.Tensor: ロジット（V または N×V）．
    """
    logits = vlm.model.logit_scale.exp() * image_features @ vlm.text_features(vocab).T
    bias = getattr(vlm.model, "logit_bias", None)
    return logits + bias if bias is not None else logits


def normalize(logits, normalization):
    """
    ロジットを確信度に変換する．

    Args:
        logits (torch.Tensor): ロジット（... × V）．
        normalization (str): "softmax"（候補語彙の中で正規化）または "sigmoid"（語ごとに独立）．

    Returns:
        torch.Tensor: 確信度（... × V）．
    """
    return logits.softmax(dim=-1) if normalization == "softmax" else logits.sigmoid()


def hierarchical_chain(vlm, image_feature, root_word, root_synset_name):
    """
    深さ0の単語から出発し，下位語集合が空になるまでWordNetの下位語をたどる（論文の式(3)）．
    各段階の確信度はソフトマックスとシグモイドの両方を記録する（選ばれる語は argmax のため同じ）．

    Args:
        vlm (CLIPModel | SigLIPModel): VLM．
        image_feature (torch.Tensor): クロップのL2正規化済みの画像特徴（D）．
        root_word (str): 深さ0の単語．
        root_synset_name (str): 深さ0の単語に対応するSynset名．

    Returns:
        list[dict]: 各深さの {"depth", "word", "synset", "conf_softmax", "conf_sigmoid", "n_candidates"} のリスト
            （深さ0の確信度は None）．
    """
    synset = wn.synset(root_synset_name)
    chain = [{"depth": 0, "word": root_word, "synset": root_synset_name, "conf_softmax": None, "conf_sigmoid": None,
              "n_candidates": None}]
    while True:
        words, synsets = hyponym_candidates(synset)
        if not words:
            return chain
        logits = logits_from_features(vlm, image_feature, words)
        best = int(logits.argmax())
        synset = synsets[best]
        chain.append({"depth": len(chain), "word": words[best], "synset": synset.name(),
                      "conf_softmax": float(normalize(logits, "softmax")[best]),
                      "conf_sigmoid": float(normalize(logits, "sigmoid")[best]), "n_candidates": len(words)})
