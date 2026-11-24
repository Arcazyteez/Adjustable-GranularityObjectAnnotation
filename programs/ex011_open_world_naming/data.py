"""
ex011のデータ定義：COCO外の物体の命名の改善（order_012）に用いるデータ．

- 分割: lvis_dev（閾値の選択），lvis_test（報告）．いずれもex007と同じ分割
- GT: COCOの物体（COCOのGT）とCOCO外の物体（LVISのアノテーションのうちCOCOのカテゴリに属さないもの）．ex010の ground_truth
- 領域の候補: OWLv2の物体らしさのスコア．lvis_test はex010のキャッシュを用い，lvis_dev は新たに推論してキャッシュする
- 語彙:
  - wn:     ex008のWordNetの同じ深さの語彙（681語）
  - wn_cap: wn から物体の部分を表す語（WordNetで部分・全体の関係の部分側にある語）を除き，
            COCO train2017のキャプションに出現する名詞に限ったもの
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

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(DATASET_DIR, "ex011_open_world_naming")
EX010_DATA_DIR = os.path.join(DATASET_DIR, "ex010_lvis_open_world_detection")
SPLITS = ["lvis_dev", "lvis_test"]
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
def ex010_data():
    """
    ex010の data モジュール（GT，語彙，ex009の読み込み）を返す．

    Returns:
        module: ex010の data．
    """
    return import_experiment_module("ex010_lvis_open_world_detection", "data")


def load_context(seed=42):
    """
    COCO，LVIS，分割，YOLO11xとDETRの検出結果を読み込む（ex009の load_context）．

    Args:
        seed (int): 分割の乱数シード．

    Returns:
        dict: "coco"，"lvis"，"splits"，"detections"．
    """
    return ex010_data().ex009_data().load_context(seed, None, None)


def owlv2_proposals(ctx, split, detector_factory):
    """
    分割の画像に対するOWLv2の領域の候補を返す（lvis_test はex010のキャッシュ，それ以外は推論してキャッシュする）．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．
        detector_factory (callable): キャッシュが無い場合に検出器を返す関数．

    Returns:
        dict: 画像ID（str）→ {"boxes"，"scores"}．
    """
    ids = ctx["splits"][split]
    path = os.path.join(EX010_DATA_DIR if split == "lvis_test" else DATA_DIR, f"owlv2_{split}_n{len(ids)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    detector = detector_factory()
    out = {}
    for image_id in tqdm(ids, desc=f"owlv2 {split}", leave=False):
        boxes, scores = detector.detect(load_image(ctx["coco"], image_id))
        out[str(image_id)] = {"boxes": [list(map(float, b)) for b in boxes], "scores": [float(s) for s in scores]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def ground_truth(ctx, split):
    """
    分割の画像ごとのGT（COCOの物体とCOCO外の物体）を返す．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．

    Returns:
        dict[str, list[dict]]: 画像ID → {"box"，"kind"，"name" または "synset"}．
    """
    return {str(i): ex010_data().ground_truth(ctx, i) for i in ctx["splits"][split]}


@functools.lru_cache(maxsize=None)
def caption_counts():
    """
    COCO train2017のキャプションにおける語（1語および2語の連続，名詞はレンマ化）の出現回数を返す（キャッシュする）．

    Returns:
        dict[str, int]: 小文字の語（2語の場合は空白区切り）→ 出現回数（CAPTION_MIN_COUNT 回以上のもの）．
    """
    path = os.path.join(DATA_DIR, f"caption_counts_min{CAPTION_MIN_COUNT}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    with open(os.path.join(COCO_DIR, "annotations", "captions_train2017.json")) as f:
        captions = [a["caption"].lower() for a in json.load(f)["annotations"]]
    counts = collections.Counter()
    for caption in tqdm(captions, desc="captions", leave=False):
        tokens = [wn.morphy(t, wn.NOUN) or t for t in re.findall(r"[a-z]+", caption)]
        counts.update(tokens)
        counts.update(" ".join(p) for p in zip(tokens, tokens[1:]))
    out = {t: c for t, c in counts.items() if c >= CAPTION_MIN_COUNT}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def caption_nouns():
    """
    COCO train2017のキャプションに CAPTION_MIN_COUNT 回以上出現する語の集合を返す．

    Returns:
        frozenset[str]: 小文字の語の集合．
    """
    return frozenset(caption_counts())


@functools.lru_cache(maxsize=None)
def is_part_noun(synset_name):
    """
    Synsetが物体の部分を表す語か（WordNetで，自身が他のSynsetの部分（part_holonyms）として登録されているか）を判定する．
    上位語まで調べると，plate（上位語のflatwareがtablewareの部分）のような物体まで除かれるため，自身のみを調べる．

    Args:
        synset_name (str): Synset名．

    Returns:
        bool: 部分を表す語であるか．
    """
    return bool(wn.synset(synset_name).part_holonyms())


def usable_word(synset_name, captions):
    """
    命名に用いる語として適切か（物体の部分でなく，いずれかのレンマがキャプションに出現する）を判定する．

    Args:
        synset_name (str): Synset名．
        captions (frozenset[str]): caption_nouns の戻り値．

    Returns:
        bool: 適切であるか．
    """
    synset = wn.synset(synset_name)
    lemmas = [lemma.lower().replace("_", " ") for lemma in synset.lemma_names()]
    return not is_part_noun(synset_name) and any(lemma in captions for lemma in lemmas)


def vocabulary(name):
    """
    深さ0の命名に用いる語彙を返す．

    Args:
        name (str): "wn"（ex008の681語）または "wn_cap"（部分の語を除き，キャプションの名詞に限る．COCOの80語は常に含める）．

    Returns:
        list[dict]: {"word"，"synset"，"coco"} のリスト．
    """
    vocab = ex010_data().wordnet_vocab()
    if name == "wn":
        return vocab
    captions = caption_nouns()
    return [v for v in vocab if v["coco"] is not None or usable_word(v["synset"], captions)]
