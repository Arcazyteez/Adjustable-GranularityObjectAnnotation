"""
ex012のモデル定義：進捗報告（2026B）の「Alg. 粒度可変アノテーション」．

- 物体認識モデル D: YOLO11x（Ultralytics），DETR（facebook/detr-resnet-50）．ボックス・スコア・COCOのクラスを出力する
- 文章・画像埋め込みモデル F: CLIP ViT-B/32（openai），SigLIP ViT-B/16（webli）．L2正規化した特徴ベクトルを出力する
- アルゴリズム（granular_annotation）:
    v = F(crop(X, B_i))
    while True:
        j* = argmax_j cos(v, F(w_j)),  w_j ∈ W
        if cos(v, F(w_j*)) < θ: break
        yield (B_i, w_j*)
        W ← w_j* の下位語の集合（WordNet）
        if W = ∅: break
"""
import os
import shutil
import sys

import numpy as np
import open_clip
import torch
from nltk.corpus import wordnet as wn
from transformers import DetrForObjectDetection, DetrImageProcessor
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import DATASET_DIR, set_seed  # noqa: E402
from data import DATA_DIR  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTION_MIN_SCORE = 0.05     # キャッシュする検出の最小スコア
EMBEDDERS = {"CLIP": ("ViT-B-32", "openai"), "SigLIP": ("ViT-B-16-SigLIP", "webli")}
TEMPLATE = "a photo of a {}"
# 定性評価の初期の語彙：WordNetの最上位カテゴリ（noun.Tops）のうち物体を表すSynset
WORDNET_TOPS = ["person.n.01", "animal.n.01", "plant.n.02", "food.n.01", "artifact.n.01", "natural_object.n.01"]


class YOLODetector:
    """COCOの80クラスで学習済みのYOLO11x（Close Vocabularyモデル）．"""

    def __init__(self, name="yolo11x", conf=DETECTION_MIN_SCORE):
        """
        Args:
            name (str): Ultralyticsのモデル名．
            conf (float): 検出スコアの下限．
        """
        weight = os.path.join(DATA_DIR, "weights", f"{name}.pt")
        if not os.path.exists(weight):
            os.makedirs(os.path.dirname(weight), exist_ok=True)
            local = os.path.join(DATASET_DIR, "ex005_coco_val500_yolo_comparison", "weights", f"{name}.pt")
            if os.path.exists(local):
                shutil.copy(local, weight)  # 同じ公開重みのファイルを再ダウンロードせずに用いる
        self.model = YOLO(weight)
        self.conf = conf

    @torch.no_grad()
    def detect(self, image):
        """
        1枚の画像を検出する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（K，COCOのクラス名）．
        """
        r = self.model.predict(image, conf=self.conf, device=DEVICE, verbose=False)[0]
        names = [self.model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
        return {"boxes": r.boxes.xyxy.cpu().numpy().tolist(), "scores": r.boxes.conf.cpu().numpy().tolist(), "labels": names}


class DETRDetector:
    """COCOの80クラスで学習済みのDETR（ResNet-50，Close Vocabularyモデル）．"""

    def __init__(self, checkpoint="facebook/detr-resnet-50", threshold=DETECTION_MIN_SCORE):
        """
        Args:
            checkpoint (str): Hugging Face Hubのチェックポイント名．
            threshold (float): 検出スコアの下限．
        """
        self.processor = DetrImageProcessor.from_pretrained(checkpoint, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(checkpoint, revision="no_timm").eval().to(DEVICE)
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, image):
        """
        1枚の画像を検出する．スコアは「物体なし」以外のクラスの最大確率である．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（K，COCOのクラス名）．
        """
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        r = self.processor.post_process_object_detection(outputs, threshold=self.threshold,
                                                         target_sizes=[image.size[::-1]])[0]
        labels = [self.model.config.id2label[int(c)] for c in r["labels"].cpu().numpy()]
        keep = [k for k, name in enumerate(labels) if name != "N/A"]  # COCOで欠番のクラスID
        boxes, scores = r["boxes"].cpu().numpy().tolist(), r["scores"].cpu().numpy().tolist()
        return {"boxes": [boxes[k] for k in keep], "scores": [scores[k] for k in keep], "labels": [labels[k] for k in keep]}


def load_detector(name, seed=0):
    """
    物体認識モデル D をインスタンス化する．

    Args:
        name (str): "yolo" または "detr"．
        seed (int): 乱数シード．

    Returns:
        YOLODetector | DETRDetector: 検出器．
    """
    set_seed(seed)
    return YOLODetector() if name == "yolo" else DETRDetector()


class Embedder:
    """文章・画像埋め込みモデル F（open_clip の CLIP または SigLIP）．"""

    def __init__(self, name, seed=0):
        """
        Args:
            name (str): "CLIP" または "SigLIP"．
            seed (int): 乱数シード．
        """
        set_seed(seed)
        arch, pretrained = EMBEDDERS[name]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.eval().to(DEVICE)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.name = name
        self._text_cache = {}

    @torch.no_grad()
    def image_features(self, images, batch_size=64):
        """
        画像の特徴ベクトルを計算する．

        Args:
            images (list[PIL.Image.Image]): 画像（長さN）．
            batch_size (int): バッチサイズ．

        Returns:
            torch.Tensor: L2正規化した特徴ベクトル（N×D）．
        """
        feats = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]]).to(DEVICE)
            feats.append(self.model.encode_image(x).float())
        feats = torch.cat(feats)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_features(self, words):
        """
        単語の特徴ベクトルを計算する（単語の集合ごとにキャッシュする）．

        Args:
            words (list[str]): 単語（長さV）．

        Returns:
            torch.Tensor: L2正規化した特徴ベクトル（V×D）．
        """
        key = tuple(words)
        if key not in self._text_cache:
            tokens = self.tokenizer([TEMPLATE.format(w) for w in words]).to(DEVICE)
            feats = self.model.encode_text(tokens).float()
            self._text_cache[key] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text_cache[key]


def crop(image, box):
    """
    ボックスで切り抜いた画像を，長辺に合わせて黒で正方形にパディングする．

    Args:
        image (PIL.Image.Image): 画像．
        box (list[float]): xyxy形式のボックス．

    Returns:
        PIL.Image.Image: 切り抜き画像．
    """
    from PIL import Image

    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    region = image.crop((x0, y0, x1, y1))
    side = max(region.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(region, ((side - region.size[0]) // 2, (side - region.size[1]) // 2))
    return canvas


def synset_word(synset):
    """
    Synsetを単語（先頭のレンマ，アンダースコアは空白）に変換する．

    Args:
        synset (Synset): Synset．

    Returns:
        str: 単語．
    """
    return synset.lemma_names()[0].replace("_", " ")


def hyponym_set(synset):
    """
    Synsetの直接の下位語の集合を返す（表記が重複する語は先頭のみ残す）．

    Args:
        synset (Synset): Synset．

    Returns:
        tuple[list[str], list[Synset]]: 単語と対応するSynset．
    """
    words, synsets = [], []
    for h in sorted(synset.hyponyms(), key=lambda s: s.name()):
        w = synset_word(h)
        if w not in words:
            words.append(w)
            synsets.append(h)
    return words, synsets


def best_word(embedder, feature, words):
    """
    単語の集合の中から，切り抜き画像と最も類似した単語を選ぶ．

    Args:
        embedder (Embedder): F．
        feature (torch.Tensor): 切り抜き画像の特徴ベクトル（D）．
        words (list[str]): 単語の集合．

    Returns:
        tuple[int, float]: 選んだ単語の番号とコサイン類似度．
    """
    sims = embedder.text_features(words) @ feature
    j = int(sims.argmax())
    return j, float(sims[j])


def granular_annotation(embedder, feature, initial_synsets, theta):
    """
    進捗報告のアルゴリズムで，1つのボックスに粗い語から細かい語までの単語を付ける．

    Args:
        embedder (Embedder): F．
        feature (torch.Tensor): 切り抜き画像の特徴ベクトル（D）．
        initial_synsets (list[str]): 初期の語彙 W のSynset名．
        theta (float): 類似と判断する下限値．

    Returns:
        list[dict]: 出力した各段階の {"word"，"synset"，"similarity"，"n_candidates"}（空なら出力なし）．
    """
    synsets = [wn.synset(s) for s in initial_synsets]
    words = [synset_word(s) for s in synsets]
    out = []
    while words:
        j, sim = best_word(embedder, feature, words)
        if sim < theta:
            break
        out.append({"word": words[j], "synset": synsets[j].name(), "similarity": sim, "n_candidates": len(words)})
        words, synsets = hyponym_set(synsets[j])
    return out


def coco_similarities(embedder, features, class_names):
    """
    定量評価のため，切り抜き画像とCOCOの80クラスの単語のコサイン類似度を計算する（下位語は生成しない）．

    Args:
        embedder (Embedder): F．
        features (torch.Tensor): 切り抜き画像の特徴ベクトル（N×D）．
        class_names (list[str]): COCOのクラス名（80）．

    Returns:
        np.ndarray: コサイン類似度（N×80）．
    """
    if len(features) == 0:
        return np.zeros((0, len(class_names)), np.float32)
    return (features @ embedder.text_features(class_names).T).cpu().numpy()
