"""
ex017：モデルと粒度可変アノテーションのアルゴリズム（order_027）．

進捗報告_2026B.pdf の「Alg. 粒度可変アノテーション」を実装する．
- 物体認識モデル D（DETR，YOLO11x）：画像 X からボックスの集合 ℬ = D(X) を1回だけ求める．
  比較のため，テキストクエリを入力とするオープンボキャブラリ物体検出モデル（OWL-ViT，Grounding DINO）も用意する．
- 埋め込みモデル F（CLIP-RN50，CLIP-ViT-B/32，SigLIP-ViT-B/16）：切り抜き画像と単語の特徴ベクトルを求める．
  比較対象として閉域モデル ResNet50-IN1k も同じインターフェースで用意する（論文3.5.1節）．
- granular_annotation：類似度が最大の単語を選び，θ 以上の間 WordNet の下位語へ進む．
  スコアは進捗報告のコサイン類似度（"cosine"）と，論文の候補語彙内のソフトマックス（"softmax"）から選ぶ．
"""
import os
import sys
import time

import numpy as np
import open_clip
import torch
import torchvision
from nltk.stem import WordNetLemmatizer
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, DetrForObjectDetection,
                          DetrImageProcessor, OwlViTForObjectDetection, OwlViTProcessor)

from data import DATASET_DIR, PROJECT_ROOT, hyponym_vocab, make_crop

sys.path.append(os.path.join(PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
from machine_learning_utils import set_seed  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "a photo of a {}"       # 論文3.5.2節のプロンプト
YOLO_WEIGHT_DIR = os.path.join(DATASET_DIR, "weights")


# ---------------------------------------------------------------------------
# 埋め込みモデル F
# ---------------------------------------------------------------------------
class OpenCLIPEmbedder:
    """open_clip の CLIP / SigLIP．画像と単語の特徴ベクトル（L2正規化）とスコアを計算する．"""

    def __init__(self, name, arch, pretrained, device=DEVICE, batch_size=64):
        """
        モデルを読み込む．

        Args:
            name (str): 表示名（例: "CLIP-ViT-B32"）．
            arch (str): open_clip のアーキテクチャ名．
            pretrained (str): 学習済みの重みの名前．
            device (str): 推論デバイス．
            batch_size (int): 特徴計算のバッチサイズ．
        """
        self.name = name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.device = device
        self.batch_size = batch_size
        self.is_siglip = "SigLIP" in arch
        self._text_cache = {}

    @torch.no_grad()
    def image_features(self, images):
        """
        画像の特徴ベクトル v_X = F(X) を求める．

        Args:
            images (list[PIL.Image.Image]): 画像（切り抜き画像）のリスト（長さN）．

        Returns:
            torch.Tensor: L2正規化した特徴（N×D，float32，CPU）．
        """
        feats = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + self.batch_size]]).to(self.device)
            feats.append(torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(feats) if feats else torch.zeros(0, 1)

    @torch.no_grad()
    def text_features(self, words):
        """
        単語の集合 𝒲 の特徴ベクトル 𝒱 = F(𝒲) を求める（プロンプト "a photo of a {w}"，単語ごとにキャッシュ）．

        Args:
            words (list[str]): 単語のリスト（長さV）．

        Returns:
            torch.Tensor: L2正規化した特徴（V×D，float32，CPU）．
        """
        new = [w for w in dict.fromkeys(words) if w not in self._text_cache]
        for i in range(0, len(new), 256):
            chunk = new[i:i + 256]
            tokens = self.tokenizer([PROMPT.format(w) for w in chunk]).to(self.device)
            feats = torch.nn.functional.normalize(self.model.encode_text(tokens).float(), dim=-1).cpu()
            self._text_cache.update(zip(chunk, feats))
        return torch.stack([self._text_cache[w] for w in words])

    def scores(self, image_feats, words, mode="cosine"):
        """
        画像と各単語のスコアを計算する．

        "cosine" は進捗報告の similarity（コサイン類似度），"softmax" は論文の式(1)の f（候補語彙内で
        logit_scale·cos にソフトマックスを適用した確信度），"sigmoid" は SigLIP の確率（t·cos + b のシグモイド）．

        Args:
            image_feats (torch.Tensor): image_features の出力（N×D）．
            words (list[str]): 候補語彙（長さV）．
            mode (str): "cosine"，"softmax"，"sigmoid" のいずれか．

        Returns:
            np.ndarray: スコア（N×V，float32）．
        """
        cos = image_feats @ self.text_features(words).T
        if mode == "cosine":
            return cos.numpy()
        logits = self.model.logit_scale.detach().exp().cpu().float() * cos
        if mode == "softmax":
            return logits.softmax(dim=-1).numpy()
        bias = self.model.logit_bias.detach().cpu().float() if self.is_siglip else 0.0
        return torch.sigmoid(logits + bias).numpy()


class ResNet50IN1k:
    """ImageNet-1kで学習済みのResNet50（閉域モデル，論文3.5.1節）．候補語がクラス名に一致する場合のみ確率を与える．"""

    name = "ResNet50-IN1k"

    def __init__(self, device=DEVICE, batch_size=64):
        """
        モデルを読み込み，クラス名をレンマ化して索引を作る．

        Args:
            device (str): 推論デバイス．
            batch_size (int): 推論のバッチサイズ．
        """
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnet50(weights=weights).eval().to(device)
        self.preprocess = weights.transforms()
        self.device = device
        self.batch_size = batch_size
        self.lemmatizer = WordNetLemmatizer()
        self.class_index = {}
        for i, c in enumerate(weights.meta["categories"]):
            self.class_index.setdefault(self.lemmatize(c), i)

    def lemmatize(self, phrase):
        """
        語句を小文字化し，各単語を名詞としてレンマ化する．

        Args:
            phrase (str): 語句（例: "Running shoes"）．

        Returns:
            str: レンマ化した語句（例: "running shoe"）．
        """
        return " ".join(self.lemmatizer.lemmatize(w) for w in phrase.lower().replace("_", " ").split())

    @torch.no_grad()
    def image_features(self, images):
        """
        1000クラスの確率ベクトルを求める（CLIPの特徴ベクトルの代わりに用いる）．

        Args:
            images (list[PIL.Image.Image]): 画像のリスト（長さN）．

        Returns:
            torch.Tensor: 確率（N×1000，float32，CPU）．
        """
        probs = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + self.batch_size]]).to(self.device)
            probs.append(self.model(x).softmax(dim=-1).float().cpu())
        return torch.cat(probs)

    def scores(self, image_feats, words, mode="softmax"):
        """
        候補語ごとのスコア．レンマ化してクラス名と完全一致する語は対応するクラスの確率，一致しない語は0とする．

        Args:
            image_feats (torch.Tensor): image_features の出力（N×1000）．
            words (list[str]): 候補語彙（長さV）．
            mode (str): 使わない（他の埋め込みモデルとインターフェースを揃えるための引数）．

        Returns:
            np.ndarray: スコア（N×V，float32）．
        """
        out = np.zeros((image_feats.shape[0], len(words)), dtype=np.float32)
        for j, w in enumerate(words):
            idx = self.class_index.get(self.lemmatize(w))
            if idx is not None:
                out[:, j] = image_feats[:, idx].numpy()
        return out


EMBEDDERS = {
    "ResNet50-IN1k": lambda: ResNet50IN1k(),
    "CLIP-RN50": lambda: OpenCLIPEmbedder("CLIP-RN50", "RN50", "openai"),
    "CLIP-ViT-B32": lambda: OpenCLIPEmbedder("CLIP-ViT-B32", "ViT-B-32", "openai"),
    "SigLIP-ViT-B16": lambda: OpenCLIPEmbedder("SigLIP-ViT-B16", "ViT-B-16-SigLIP", "webli"),
}


def load_embedder(name, seed=0):
    """
    埋め込みモデル F をインスタンス化する．

    Args:
        name (str): EMBEDDERS のキー．
        seed (int): 乱数シード（推論のみだが再現性のため固定する）．

    Returns:
        OpenCLIPEmbedder | ResNet50IN1k: 埋め込みモデル．
    """
    set_seed(seed)
    return EMBEDDERS[name]()


# ---------------------------------------------------------------------------
# 物体認識モデル D
# ---------------------------------------------------------------------------
def class_agnostic_nms(boxes, scores, iou_threshold=0.5):
    """
    クラスを区別しない NMS で重複するボックスを除く．

    Args:
        boxes (torch.Tensor): xyxy形式のボックス（K×4）．
        scores (torch.Tensor): スコア（K）．
        iou_threshold (float): 重複とみなす IoU．

    Returns:
        tuple[np.ndarray, np.ndarray]: 残したボックス（K'×4）とスコア（K'），スコアの降順．
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    keep = torchvision.ops.nms(boxes.float().cpu(), scores.float().cpu(), iou_threshold)
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()


class DETRDetector:
    """DETR（facebook/detr-resnet-50）．提案手法ではボックスのみを用い，Close Vocabularyモデルとしてはクラスも用いる．"""

    name = "DETR"

    def __init__(self, threshold=0.5, device=DEVICE):
        """
        モデルを読み込む．

        Args:
            threshold (float): 検出スコアの閾値（論文4.6節の0.5）．
            device (str): 推論デバイス．
        """
        ckpt = "facebook/detr-resnet-50"
        self.processor = DetrImageProcessor.from_pretrained(ckpt, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(ckpt, revision="no_timm").eval().to(device)
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        1回の推論でボックスを検出する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（長さKのCOCOクラス名）．スコアの降順．
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        result = self.processor.post_process_object_detection(
            self.model(**inputs), threshold=self.threshold, target_sizes=[image.size[::-1]])[0]
        order = result["scores"].argsort(descending=True).cpu()
        labels = [self.model.config.id2label[int(i)] for i in result["labels"].cpu()[order]]
        return {"boxes": result["boxes"].cpu()[order].numpy().reshape(-1, 4),
                "scores": result["scores"].cpu()[order].numpy(), "labels": labels}


class YOLODetector:
    """COCOの80クラスで学習済みの YOLO11x（Ultralytics．Close Vocabularyモデル）．"""

    name = "YOLO11x"

    def __init__(self, conf=0.25, device=DEVICE):
        """
        モデルを読み込む（重みは DATASET_DIR/weights/yolo11x.pt．無ければ自動でダウンロードする）．

        Args:
            conf (float): 検出スコアの閾値（Ultralyticsの既定値）．
            device (str): 推論デバイス．
        """
        from ultralytics import YOLO
        os.makedirs(YOLO_WEIGHT_DIR, exist_ok=True)
        weight = os.path.join(YOLO_WEIGHT_DIR, "yolo11x.pt")
        if not os.path.exists(weight):
            cwd = os.getcwd()
            os.chdir(YOLO_WEIGHT_DIR)
            YOLO("yolo11x.pt")
            os.chdir(cwd)
        self.model = YOLO(weight)
        self.conf = conf
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        1回の推論でボックスとクラスを検出する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（長さKのCOCOクラス名）．スコアの降順．
        """
        r = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        order = r.boxes.conf.argsort(descending=True).cpu()
        return {"boxes": r.boxes.xyxy.cpu()[order].numpy().reshape(-1, 4), "scores": r.boxes.conf.cpu()[order].numpy(),
                "labels": [r.names[int(c)] for c in r.boxes.cls.cpu()[order]]}


class OWLViTDetector:
    """テキストクエリを入力とする OWL-ViT（論文4.3節の比較対象）．"""

    name = "OWL-ViT"

    def __init__(self, queries, threshold=0.1, device=DEVICE):
        """
        モデルを読み込む．

        Args:
            queries (list[str]): テキストクエリとするカテゴリ名．
            threshold (float): 検出スコアの閾値．
            device (str): 推論デバイス．
        """
        ckpt = "google/owlvit-base-patch32"
        self.processor = OwlViTProcessor.from_pretrained(ckpt)
        self.model = OwlViTForObjectDetection.from_pretrained(ckpt).eval().to(device)
        self.queries = [PROMPT.format(q) for q in queries]
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        テキストクエリを与えて検出し，クラス非依存のボックスに変換する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（空のリスト）．
        """
        inputs = self.processor(text=[self.queries], images=image, return_tensors="pt").to(self.device)
        result = self.processor.post_process_object_detection(
            self.model(**inputs), threshold=self.threshold, target_sizes=torch.tensor([image.size[::-1]]))[0]
        boxes, scores = class_agnostic_nms(result["boxes"], result["scores"])
        return {"boxes": boxes.reshape(-1, 4), "scores": scores, "labels": []}


class GroundingDINODetector:
    """テキストクエリを入力とする Grounding DINO（論文4.3節の比較対象）．"""

    name = "Grounding DINO"

    def __init__(self, queries, box_threshold=0.35, text_threshold=0.25, chunk_size=40, device=DEVICE):
        """
        モデルを読み込む．

        Args:
            queries (list[str]): テキストクエリとするカテゴリ名．
            box_threshold (float): ボックスのスコアの閾値．
            text_threshold (float): テキストとの対応のスコアの閾値．
            chunk_size (int): 1回の推論に含めるカテゴリ数（テキスト長の上限256トークンのため分割する）．
            device (str): 推論デバイス．
        """
        ckpt = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(ckpt).eval().to(device)
        self.prompts = [". ".join(queries[i:i + chunk_size]) + "." for i in range(0, len(queries), chunk_size)]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image):
        """
        カテゴリのチャンクごとに検出し，統合してクラス非依存のボックスに変換する．

        Args:
            image (PIL.Image.Image): 入力画像．

        Returns:
            dict: "boxes"（K×4，xyxy），"scores"（K），"labels"（空のリスト）．
        """
        boxes, scores = [], []
        for prompt in self.prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            result = self.processor.post_process_grounded_object_detection(
                self.model(**inputs), inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[image.size[::-1]])[0]
            boxes.append(result["boxes"].cpu())
            scores.append(result["scores"].cpu())
        b, s = class_agnostic_nms(torch.cat(boxes), torch.cat(scores))
        return {"boxes": b.reshape(-1, 4), "scores": s, "labels": []}


def load_detector(name, queries=None, seed=0):
    """
    物体認識モデル D をインスタンス化する．

    Args:
        name (str): "DETR"，"YOLO11x"，"OWL-ViT"，"Grounding DINO" のいずれか．
        queries (list[str] | None): OWL-ViT と Grounding DINO のテキストクエリ．
        seed (int): 乱数シード．

    Returns:
        DETRDetector | YOLODetector | OWLViTDetector | GroundingDINODetector: 検出器．
    """
    set_seed(seed)
    if name == "DETR":
        return DETRDetector()
    if name == "YOLO11x":
        return YOLODetector()
    if name == "OWL-ViT":
        return OWLViTDetector(queries)
    return GroundingDINODetector(queries)


def timed_detect(detector, image):
    """
    検出を実行し，所要時間も返す．

    Args:
        detector: load_detector の戻り値．
        image (PIL.Image.Image): 入力画像．

    Returns:
        tuple[dict, float]: detect の戻り値と，所要時間（秒）．
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    det = detector.detect(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return det, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Alg. 粒度可変アノテーション（進捗報告_2026B.pdf）
# ---------------------------------------------------------------------------
def granular_annotation(embedder, image_feat, vocab, theta=-np.inf, mode="cosine", max_depth=None, top_k=5):
    """
    1つの切り抜き画像の特徴 v_{X_i} に対し，単語を選んで WordNet の下位語へ進む（Alg. の While ループ）．

    各段階で 𝒱 = F(𝒲) との similarity が最大の単語 w_{j*} を選び，similarity(v_{X_i}, v_{j*}) < θ なら終了する．
    θ 以上なら (B_i, w_{j*}) を出力し，𝒲 を w_{j*} の下位語の集合に置き換える．下位語が無ければ終了する．
    θ = −∞ のとき論文3.4節の設定（下位語が無くなるか最大深さまで進む）になる．

    Args:
        embedder (OpenCLIPEmbedder | ResNet50IN1k): 埋め込みモデル F．
        image_feat (torch.Tensor): 切り抜き画像の特徴（1×D）．
        vocab (list[tuple[str, str | None]]): 初期の語彙 𝒲 の (単語, Synset名) のリスト．
        theta (float): 類似と判断する下限値 θ．
        mode (str): スコアの種類（"cosine"，"softmax"，"sigmoid"）．
        max_depth (int | None): 最大の深さ（None なら制限しない）．
        top_k (int): 記録する上位の候補の数．

    Returns:
        list[dict]: 深さの浅い順の出力．各要素は "depth"，"word"，"synset"，"score"，"n_candidates"，
                    "top"（上位 top_k の [単語, スコア]）を持つ．
    """
    chain, depth = [], 0
    while vocab and (max_depth is None or depth <= max_depth):
        words = [w for w, _ in vocab]
        s = embedder.scores(image_feat, words, mode)[0]
        j = int(np.argmax(s))
        if s[j] < theta:
            break
        order = np.argsort(-s)[:top_k]
        chain.append({"depth": depth, "word": words[j], "synset": vocab[j][1], "score": float(s[j]),
                      "n_candidates": len(words), "top": [[words[k], float(s[k])] for k in order]})
        vocab = hyponym_vocab(vocab[j][1])
        depth += 1
    return chain


def annotate_image(detector, embedder, image, vocab, theta=-np.inf, mode="cosine", max_depth=None):
    """
    Alg. 粒度可変アノテーション全体：ℬ = D(X) を1回だけ求め，各ボックスの切り抜き画像に granular_annotation を適用する．

    Args:
        detector: 物体認識モデル D（load_detector の戻り値）．
        embedder: 埋め込みモデル F（load_embedder の戻り値）．
        image (PIL.Image.Image): アノテーションしたい画像 X．
        vocab (list[tuple[str, str | None]]): 初期の語彙 𝒲．
        theta (float): 類似と判断する下限値 θ．
        mode (str): スコアの種類．
        max_depth (int | None): 最大の深さ．

    Returns:
        list[dict]: ボックスごとの "box"（長さ4のリスト，xyxy），"det_score"，"chain"（granular_annotation の出力）．
                    chain が空（深さ0で θ 未満）のボックスも含む．
    """
    det = detector.detect(image)
    if len(det["boxes"]) == 0:
        return []
    feats = embedder.image_features([make_crop(image, b) for b in det["boxes"]])
    return [{"box": [float(v) for v in b], "det_score": float(sc),
             "chain": granular_annotation(embedder, feats[i:i + 1], vocab, theta, mode, max_depth)}
            for i, (b, sc) in enumerate(zip(det["boxes"], det["scores"]))]
