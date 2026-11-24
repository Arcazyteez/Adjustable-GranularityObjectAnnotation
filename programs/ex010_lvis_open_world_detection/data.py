"""
ex010のデータ定義：COCO外の物体の検出と，検出した領域の命名・粒度変更の評価データ（order_011の要求2）．

- 評価画像: ex007と同じ lvis_test（COCO 2017 valのうち，検証用と重複しないLVISのアノテーションがある2148枚）
- COCOの物体: COCOのGT（80クラス）
- COCO外の物体: LVISのアノテーションのうち，SynsetがCOCOのカテゴリの上位語・下位語（自身を含む）でなく，
  レンマ名もCOCOのカテゴリ名と一致しないもの（ex008の is_non_coco_synset）
- 命名の語彙:
  - wn:   COCOの80語 + WordNetにおいてCOCOのカテゴリと同じ深さにある語彙（ex008，k=2，681語）
  - lvis: LVIS v1のカテゴリ名（語彙が既知の場合の参考）
"""
import functools
import importlib.util
import json
import os
import re
import sys

from nltk.corpus import wordnet as wn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import gt_instances, load_image  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402
from common.wordnet_utils import synset_word  # noqa: E402

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(DATASET_DIR, "ex010_lvis_open_world_detection")
LVIS_PATH = os.path.join(DATASET_DIR, "lvis", "lvis_v1_val.json")
EVAL_SPLIT = "lvis_test"


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
def ex009_data():
    """
    ex009の data モジュール（COCO，LVIS，分割，検出結果の読み込み）を返す．

    Returns:
        module: ex009の data．
    """
    return import_experiment_module("ex009_coco_siglip_wordnet_annotation", "data")


@functools.lru_cache(maxsize=None)
def ex008_data():
    """
    ex008の data モジュール（COCO外のSynsetの判定，WordNetの同じ深さの語彙）を返す．

    Returns:
        module: ex008の data．
    """
    return import_experiment_module("ex008_coco_val_siglip_wordnet_root", "data")


def ground_truth(ctx, image_id):
    """
    1枚の画像のCOCOの物体とCOCO外の物体のGTを返す．

    Args:
        ctx (dict): ex009の load_context の戻り値．
        image_id (int): 画像ID．

    Returns:
        list[dict]: {"box"（xyxy），"kind"（"coco" または "noncoco"），"name"（COCOのカテゴリ名）または "synset"（LVIS）}．
    """
    boxes, names = gt_instances(ctx["coco"], image_id)
    gts = [{"box": list(map(float, b)), "kind": "coco", "name": n} for b, n in zip(boxes, names)]
    gts += [{"box": a["box"], "kind": "noncoco", "synset": a["synset"]} for a in ctx["lvis"].get(image_id, [])
            if ex008_data().is_non_coco_synset(a["synset"])]
    return gts


def wordnet_vocab():
    """
    COCOの80語とWordNetの同じ深さの語彙（k=2）を返す．

    Returns:
        list[dict]: {"word"，"synset"，"coco"} のリスト（COCOの80語を含む681語）．
    """
    return ex008_data().build_wordnet_root_vocab(2, 1)


@functools.lru_cache(maxsize=None)
def lvis_vocab():
    """
    LVIS v1のカテゴリ名を語彙として返す（WordNet 3.0に存在するSynsetのみ，表記の重複は先頭のみ残す）．

    Returns:
        list[dict]: {"word"，"synset"，"coco"（None）} のリスト．
    """
    with open(LVIS_PATH) as f:
        categories = json.load(f)["categories"]
    vocab, words = [], set()
    for c in sorted(categories, key=lambda c: c["id"]):
        try:
            synset = wn.synset(c["synset"])
        except Exception:  # WordNet 3.0に存在しないSynset名
            continue
        word = re.sub(r"_\(.*\)$", "", c["name"]).replace("_", " ") or synset_word(synset)
        if word in words:
            continue
        words.add(word)
        vocab.append({"word": word, "synset": synset.name(), "coco": None})
    return vocab


def cached_detections(name, coco, image_ids, detector_factory):
    """
    検出器を全画像に1回ずつ適用し，結果をキャッシュする．

    Args:
        name (str): 検出器の名前（キャッシュのファイル名）．
        coco (COCO): instancesのCOCOオブジェクト．
        image_ids (list[int]): 画像ID．
        detector_factory (callable): キャッシュが無い場合に detect(image) -> (boxes, scores) を持つ検出器を返す関数．

    Returns:
        dict: 画像ID（str）→ {"boxes": K×4，"scores": K}（スコアの降順）．
    """
    path = os.path.join(DATA_DIR, f"{name}_{EVAL_SPLIT}_n{len(image_ids)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    detector = detector_factory()
    out = {}
    for image_id in tqdm(image_ids, desc=name, leave=False):
        boxes, scores = detector.detect(load_image(coco, image_id))
        out[str(image_id)] = {"boxes": [list(map(float, b)) for b in boxes], "scores": [float(s) for s in scores]}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out
