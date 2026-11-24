"""
ex013のデータ定義：再学習なしでCOCO外の語彙の物体を検出し，粒度可変のアノテーションを行う評価（order_015）．

- 画像: ex012と同じ検証用（dev）とテスト用（test）の分割のうち，LVIS v1 val のアノテーションがある画像
- GT: LVISのアノテーション（ボックスとWordNetのSynset）．COCOのクラスのSynsetの上位語・下位語（自身を含む）でなく，
      レンマ名もCOCOのクラス名と一致しないものを「COCO外の物体」，それ以外を「COCOの物体」とする
- 領域: YOLO11x（ex012の検出結果のキャッシュ）と，OWLv2の物体らしさによる領域（本実験で推論してキャッシュする）
- キャプション: COCO train2017のキャプション（WordNetの絞り込みに用いる．評価に用いるval2017とは重ならない）
"""
import collections
import functools
import importlib.util
import json
import os
import re
import sys

from nltk.corpus import wordnet as wn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.paths import COCO_DIR, DATASET_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(DATASET_DIR, "ex013_open_vocabulary_granular")
LVIS_PATH = os.path.join(DATASET_DIR, "lvis", "lvis_v1_val.json")
SPLITS = ["dev", "test"]
CAPTION_MIN_COUNT = 5


def import_experiment_module(experiment, module):
    """
    他の実験ディレクトリのモジュールを，名前の衝突を避けて読み込む．

    Args:
        experiment (str): 実験ディレクトリ名．
        module (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    path = os.path.join(PROGRAM_DIR, experiment, f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"{experiment}_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.lru_cache(maxsize=None)
def ex012_data():
    """
    ex012の data モジュール（COCO，分割，検出結果のキャッシュ）を返す．

    Returns:
        module: ex012の data．
    """
    return import_experiment_module("ex012_wordnet_granular_annotation", "data")


@functools.lru_cache(maxsize=None)
def ancestors(synset_name):
    """
    Synset自身と全ての上位語の集合を返す．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    s = wn.synset(synset_name)
    return frozenset([s.name()] + [x.name() for x in s.closure(lambda y: y.hypernyms())])


@functools.lru_cache(maxsize=None)
def is_coco_object(synset_name):
    """
    LVISのSynsetがCOCOの物体（COCOのクラスのSynsetの上位語・下位語・自身，またはレンマ名がクラス名と一致）かを判定する．

    Args:
        synset_name (str): Synset名．

    Returns:
        bool: COCOの物体であるか．
    """
    coco = set(COCO_TO_SYNSET.values())
    names = {n.replace(" ", "_") for n in COCO_TO_SYNSET}
    s = wn.synset(synset_name)
    if ancestors(synset_name) & coco or set(s.lemma_names()) & names:
        return True
    return any(synset_name in ancestors(c) for c in coco)


def load_lvis():
    """
    LVISのアノテーションを画像ごとに読み込む（WordNet 3.0に存在するSynsetのみ）．

    Returns:
        dict[int, list[dict]]: 画像ID → {"box"（xyxy），"synset"，"kind"（"coco" または "noncoco"）} のリスト．
    """
    with open(LVIS_PATH) as f:
        lvis = json.load(f)
    synset_of = {}
    for c in lvis["categories"]:
        try:
            synset_of[c["id"]] = wn.synset(c["synset"]).name()
        except Exception:  # WordNet 3.0に存在しないSynset名
            continue
    per_image = {}
    for a in lvis["annotations"]:
        if a["category_id"] not in synset_of:
            continue
        x, y, w, h = a["bbox"]
        s = synset_of[a["category_id"]]
        per_image.setdefault(a["image_id"], []).append(
            {"box": [x, y, x + w, y + h], "synset": s, "kind": "coco" if is_coco_object(s) else "noncoco"})
    return per_image


def load_context(seed=42):
    """
    COCO，LVIS，分割（LVISのアノテーションがある画像のみ），YOLO11xの検出結果を読み込む．

    Returns:
        dict: "coco"，"lvis"，"splits"，"yolo"．
    """
    d12 = ex012_data()
    coco, _ = d12.load_dataset()
    lvis = load_lvis()
    splits = {k: [i for i in v if i in lvis] for k, v in d12.make_splits(coco, seed).items()}
    yolo = d12.cached_detections("yolo", coco, sorted(coco.getImgIds()), None)
    return {"coco": coco, "lvis": lvis, "splits": splits, "yolo": yolo}


def owlv2_proposals(ctx, split, detector_factory):
    """
    分割の画像にOWLv2の物体らしさによる領域の検出を適用し，キャッシュする．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．
        detector_factory (callable): キャッシュが無い場合に detect(image) を持つ検出器を返す関数．

    Returns:
        dict: 画像ID（str）→ {"boxes"，"scores"}（スコアの降順）．
    """
    ids = ctx["splits"][split]
    path = os.path.join(DATA_DIR, f"owlv2_{split}_n{len(ids)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    detector = detector_factory()
    out = {str(i): detector.detect(load_image(ctx["coco"], i)) for i in tqdm(ids, desc=f"owlv2 {split}", leave=False)}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


@functools.lru_cache(maxsize=None)
def caption_terms():
    """
    COCO train2017のキャプションに CAPTION_MIN_COUNT 回以上出現する語（1語と2語の連続，名詞はレンマ化）の集合を返す．

    Returns:
        frozenset[str]: 小文字の語の集合．
    """
    path = os.path.join(DATA_DIR, f"caption_terms_min{CAPTION_MIN_COUNT}.json")
    if os.path.exists(path):
        with open(path) as f:
            return frozenset(json.load(f))
    with open(os.path.join(COCO_DIR, "annotations", "captions_train2017.json")) as f:
        captions = [a["caption"].lower() for a in json.load(f)["annotations"]]
    counts = collections.Counter()
    for caption in tqdm(captions, desc="captions", leave=False):
        tokens = [wn.morphy(t, wn.NOUN) or t for t in re.findall(r"[a-z]+", caption)]
        counts.update(tokens)
        counts.update(" ".join(p) for p in zip(tokens, tokens[1:]))
    terms = sorted(t for t, c in counts.items() if c >= CAPTION_MIN_COUNT)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(terms, f)
    return frozenset(terms)
