"""
ex019：検出器 D と埋め込みモデル F（order_029）．

- 新しい提案手法：OWLv2 の objectness（クラス非依存）→ Alpha-CLIP（ViT-B/16）→ WordNet
- 古い提案手法：DETR（閾値0.5）→ CLIP（ViT-B/32，切り抜き画像）→ WordNet
Alpha-CLIP（Sun et al., CVPR 2024）は，CLIP の画像エンコーダのパッチ埋め込みに αチャネル用の畳み込み conv1_alpha を
加えたモデルである．公式の実装はパッケージの導入が必要なため，open_clip の CLIP ViT-B/16（OpenAI，QuickGELU）に
conv1_alpha を加え，公式の重み（clip_b16_grit+mim_fultune_4xe.pth，画像エンコーダのみ）を読み込んで同じ計算を行う．
文章エンコーダは元の CLIP のものを用いる（Alpha-CLIP の学習でも固定されている）．

注意：open_clip の "ViT-B-32" に pretrained="openai" を与えると活性化関数が GELU になり，OpenAI の学習時の QuickGELU と
一致しない（ex017 はこの設定だった）．本実験では全ての CLIP に "-quickgelu" の設定を用いる．
"""
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from data import DATASET_DIR, load_ex017

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_CLIP_WEIGHT = os.path.join(DATASET_DIR, "weights", "clip_b16_grit+mim_fultune_4xe.pth")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ALPHA_MEAN, ALPHA_STD = 0.5, 0.26           # 公式の実装のαの正規化
OWL_MAX_BOXES = 300
OWL_MIN_SCORE = 0.001


class AlphaPatchEmbed(nn.Module):
    """パッチ埋め込み conv1 の出力に conv1_alpha(α) を加える層（Alpha-CLIP の VisionTransformer.forward と同じ計算）．"""

    def __init__(self, conv1, width, patch_size):
        """
        Args:
            conv1 (nn.Conv2d): 元の CLIP の RGB のパッチ埋め込み．
            width (int): 埋め込みの次元．
            patch_size (int): パッチの一辺．
        """
        super().__init__()
        self.conv1 = conv1
        self.conv1_alpha = nn.Conv2d(1, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.alpha = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 正規化したRGB画像（B×3×H×W）．

        Returns:
            torch.Tensor: パッチの埋め込み（B×width×H/p×W/p）．
        """
        return self.conv1(x) + self.conv1_alpha(self.alpha.to(x.dtype))


def clip_embedder(name, arch):
    """
    open_clip の OpenAI CLIP（QuickGELU）を ex017 の OpenCLIPEmbedder として読み込む（切り抜き画像を入力とする）．

    Args:
        name (str): 表示名（例: "CLIP-ViT-B32"）．
        arch (str): open_clip のアーキテクチャ名（例: "ViT-B-32-quickgelu"）．

    Returns:
        OpenCLIPEmbedder: 埋め込みモデル．
    """
    return load_ex017()[1].OpenCLIPEmbedder(name, arch, "openai")


class AlphaCLIPEmbedder:
    """Alpha-CLIP（ViT-B/16）．RGB画像とαマスクから画像の特徴を求め，文章の特徴とスコアは元の CLIP と同じ計算を行う．"""

    def __init__(self, name="AlphaCLIP-ViT-B16", weight=ALPHA_CLIP_WEIGHT, device=DEVICE, batch_size=128):
        """
        Args:
            name (str): 表示名．
            weight (str): Alpha-CLIP の画像エンコーダの重みのパス．
            device (str): 推論デバイス．
            batch_size (int): 特徴計算のバッチサイズ．
        """
        self.base = clip_embedder(name, "ViT-B-16-quickgelu")
        visual = self.base.model.visual
        visual.conv1 = AlphaPatchEmbed(visual.conv1, visual.conv1.out_channels, visual.conv1.kernel_size[0])
        state = torch.load(weight, map_location="cpu", weights_only=False)
        mapped = {}
        for k, v in state.items():
            k = k.replace("attn.in_proj.weight", "attn.in_proj_weight").replace("attn.in_proj.bias", "attn.in_proj_bias")
            k = {"conv1.weight": "conv1.conv1.weight", "conv1_alpha.weight": "conv1.conv1_alpha.weight"}.get(k, k)
            mapped[k] = v
        visual.load_state_dict(mapped, strict=True)
        self.base.model = self.base.model.eval().to(device)
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def image_features(self, images, alphas):
        """
        RGB画像とαマスクから画像の特徴を求める．

        Args:
            images (list[PIL.Image.Image]): 224×224 のRGB画像（長さN，data.make_alpha_input の出力）．
            alphas (list[np.ndarray]): 224×224 のαマスク（0〜1，長さN）．

        Returns:
            torch.Tensor: L2正規化した特徴（N×D，float32，CPU）．
        """
        feats = []
        visual = self.base.model.visual
        for i in range(0, len(images), self.batch_size):
            x = torch.from_numpy(np.stack([np.asarray(im, np.float32) / 255 for im in images[i:i + self.batch_size]]))
            x = (x.permute(0, 3, 1, 2).to(self.device) - self.mean) / self.std
            a = torch.from_numpy(np.stack(alphas[i:i + self.batch_size]))[:, None].to(self.device)
            visual.conv1.alpha = (a - ALPHA_MEAN) / ALPHA_STD
            f = self.base.model.encode_image(x).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        visual.conv1.alpha = None
        return torch.cat(feats) if feats else torch.zeros(0, 1)

    def text_features(self, words):
        """
        単語の特徴（プロンプト "a photo of a {w}"）．

        Args:
            words (list[str]): 単語のリスト（長さV）．

        Returns:
            torch.Tensor: L2正規化した特徴（V×D）．
        """
        return self.base.text_features(words)

    def scores(self, image_feats, words, mode="softmax"):
        """
        画像と各単語のスコア（ex017 の OpenCLIPEmbedder.scores と同じ定義）．

        Args:
            image_feats (torch.Tensor): image_features の出力（N×D）．
            words (list[str]): 候補語彙（長さV）．
            mode (str): "cosine" または "softmax"．

        Returns:
            np.ndarray: スコア（N×V）．
        """
        return self.base.scores(image_feats, words, mode)


class OWLv2Objectness:
    """OWLv2（google/owlv2-base-patch16-ensemble）の物体らしさのスコアによるクラス非依存の検出（ex018 で選んだ構成）．"""

    name = "OWLv2"

    def __init__(self, checkpoint="google/owlv2-base-patch16-ensemble", device=DEVICE):
        """
        Args:
            checkpoint (str): Hugging Face Hub のチェックポイント名．
            device (str): 推論デバイス．
        """
        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint).eval().to(device)
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        objectness のスコアで検出する（クラス非依存の NMS IoU 0.5 の後，上位 OWL_MAX_BOXES 件，スコア OWL_MIN_SCORE 以上）．
        OWLv2 は画像を右下に0埋めした正方形として扱うため，ボックスを長辺の長さで復元する．
        テキストのクエリは objectness に影響しないが，モデルの入力に必要なため1語だけ与える．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（空のリスト）．
        """
        inputs = self.processor(text=[["a photo of an object"]], images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        cx, cy, w, h = (out.pred_boxes[0] * max(image.size)).T
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        limit = torch.tensor([image.size[0], image.size[1]] * 2, device=boxes.device, dtype=boxes.dtype)
        boxes = torch.minimum(boxes.clamp(min=0), limit)
        scores = out.objectness_logits[0].sigmoid()
        mask = scores >= OWL_MIN_SCORE
        b, s = load_ex017()[1].class_agnostic_nms(boxes[mask], scores[mask], 0.5)
        return {"boxes": b[:OWL_MAX_BOXES], "scores": s[:OWL_MAX_BOXES], "labels": []}


def granular_annotation(embedder, image_feat, vocab, theta=-np.inf, mode="softmax", max_depth=None, top_k=5):
    """
    1つの領域の特徴から語を選び，θ 以上の間 WordNet の下位語へ進む（ex017 の granular_annotation，進捗報告の Alg.）．

    Args:
        embedder: 埋め込みモデル（scores を持つ）．
        image_feat (torch.Tensor): 領域の特徴（1×D）．
        vocab (list[tuple[str, str]]): 初期の語彙．
        theta (float): θ．
        mode (str): スコアの種類．
        max_depth (int | None): 最大の深さ．
        top_k (int): 記録する上位の候補の数．

    Returns:
        list[dict]: 深さの浅い順の出力（"depth"，"word"，"synset"，"score"，"n_candidates"，"top"）．
    """
    return load_ex017()[1].granular_annotation(embedder, image_feat, vocab, theta, mode, max_depth, top_k)
