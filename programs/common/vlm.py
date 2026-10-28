"""
視覚言語モデル f: 画像 → 候補語彙集合上のスコア（論文の式(1)）の実装．

- ResNet50IN1k: ImageNet-1k学習済みの閉域モデル．候補語をレンマ化してクラス名と完全一致した場合のみそのクラス確率をスコアとし，一致しない語には0を割り当てる．
- CLIPModel: open_clipのOpenAI学習済みCLIP．"a photo of a {w}" のプロンプトで画像特徴とのコサイン類似度をソフトマックスで正規化する．
"""
import numpy as np
import open_clip
import torch
import torchvision
from nltk.corpus import wordnet as wn


def lemmatize_phrase(phrase):
    """
    語句を小文字化し，末尾の語を名詞としてレンマ化する．

    Args:
        phrase (str): 単語または複合語（例: "hot dogs"）．

    Returns:
        str: レンマ化した語句（例: "hot dog"）．
    """
    tokens = phrase.lower().replace("_", " ").split()
    if not tokens:
        return ""
    tokens[-1] = wn.morphy(tokens[-1], wn.NOUN) or tokens[-1]
    return " ".join(tokens)


class ResNet50IN1k:
    """ImageNet-1k学習済みResNet-50による閉域の単語スコアリング．"""

    name = "ResNet50-IN1k"

    def __init__(self, device="cuda"):
        """
        モデルを読み込む．

        Args:
            device (str): 推論デバイス．
        """
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnet50(weights=weights).eval().to(device)
        self.preprocess = weights.transforms()
        self.class_names = [lemmatize_phrase(c) for c in weights.meta["categories"]]
        self.device = device

    @torch.no_grad()
    def image_probs(self, images, batch_size=64):
        """
        画像ごとの1000クラスの確率ベクトルを計算する．

        Args:
            images (list[PIL.Image.Image]): 画像のリスト（長さN）．
            batch_size (int): バッチサイズ．

        Returns:
            np.ndarray: 確率行列（N×1000）．
        """
        outputs = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]]).to(self.device)
            outputs.append(self.model(x).softmax(dim=-1).float().cpu().numpy())
        return np.concatenate(outputs)

    def word_to_classes(self, vocab):
        """
        候補語ごとに，レンマ化後のクラス名が完全一致するImageNetクラスのインデックスを求める．

        Args:
            vocab (list[str]): 候補語彙（長さV）．

        Returns:
            list[list[int]]: 各候補語に対応するクラスインデックスのリスト（長さV）．
        """
        return [[c for c, name in enumerate(self.class_names) if name == lemmatize_phrase(w)] for w in vocab]

    def score(self, images, vocab):
        """
        候補語彙上のスコア f(X) を計算する．

        Args:
            images (list[PIL.Image.Image]): 画像のリスト（長さN）．
            vocab (list[str]): 候補語彙（長さV）．

        Returns:
            np.ndarray: スコア行列（N×V）．一致クラスが無い語は0．
        """
        probs = self.image_probs(images)
        scores = np.zeros((len(images), len(vocab)), dtype=np.float32)
        for v, classes in enumerate(self.word_to_classes(vocab)):
            if classes:
                scores[:, v] = probs[:, classes].max(axis=1)
        return scores


class CLIPModel:
    """open_clipによるCLIPのゼロショット分類．"""

    def __init__(self, arch="ViT-B-32", device="cuda", template="a photo of a {}"):
        """
        モデルを読み込む．

        Args:
            arch (str): open_clipのアーキテクチャ名（"RN50" または "ViT-B-32"）．
            device (str): 推論デバイス．
            template (str): プロンプトのテンプレート．
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
        self.model = self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.name = {"RN50": "CLIP-RN50", "ViT-B-32": "CLIP-ViT-B32"}[arch]
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

    @torch.no_grad()
    def score(self, images, vocab):
        """
        候補語彙上のスコア f(X)（ソフトマックス確率）を計算する．

        Args:
            images (list[PIL.Image.Image]): 画像のリスト（長さN）．
            vocab (list[str]): 候補語彙（長さV）．

        Returns:
            np.ndarray: スコア行列（N×V），各行の和は1．
        """
        logits = self.model.logit_scale.exp() * self.image_features(images) @ self.text_features(vocab).T
        return logits.softmax(dim=-1).cpu().numpy()


def load_vlm(name, device="cuda"):
    """
    名前からVLMを生成する．

    Args:
        name (str): "ResNet50-IN1k"，"CLIP-RN50"，"CLIP-ViT-B32" のいずれか．
        device (str): 推論デバイス．

    Returns:
        ResNet50IN1k | CLIPModel: score(images, vocab) を持つモデル．
    """
    if name == "ResNet50-IN1k":
        return ResNet50IN1k(device)
    return CLIPModel({"CLIP-RN50": "RN50", "CLIP-ViT-B32": "ViT-B-32"}[name], device)
