"""
ex018：OWLの領域検出の確認に用いるデータ（order_028）．

- 画像：ex017のテスト用（test）500枚のうち，LVIS v1 val の注釈がある画像
- GT：LVISのボックスとWordNetのSynset．COCOのクラスのSynsetの上位語・下位語・自身（またはレンマ名がクラス名と一致）を
  「COCOの物体」，それ以外を「COCO外の物体」とする
- YOLO11x と OWL-ViT の検出結果：ex017のキャッシュ（`datasets/ex017_coco_granular_annotation/`）を読み込む
ex017の data.py は他のモジュールに依存しないため，ファイルのパスから別名で読み込む．
"""
import functools
import importlib.util
import json
import os

import numpy as np
import torch
from nltk.corpus import wordnet as wn

EX_NAME = "ex018_owl_region_check"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", EX_NAME)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", EX_NAME)
EX017_DIR = os.path.join(PROJECT_ROOT, "programs", "ex017_coco_granular_annotation")
EX017_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "ex017_coco_granular_annotation")
LVIS_PATH = os.path.join(PROJECT_ROOT, "datasets", "lvis", "lvis_v1_val.json")


@functools.lru_cache(maxsize=None)
def ex017_data():
    """
    ex017の data.py を "ex017_data" という名前で読み込む．

    Returns:
        module: ex017の data モジュール．
    """
    spec = importlib.util.spec_from_file_location("ex017_data", os.path.join(EX017_DIR, "data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_coco():
    """
    COCO 2017 val の注釈を読み込む（ex017の load_coco）．

    Returns:
        dict: ex017の load_coco の戻り値．
    """
    return ex017_data().load_coco()


def load_image(coco, image_id):
    """
    COCOの画像をRGBで読み込む．

    Args:
        coco (dict): load_coco の戻り値．
        image_id (int): 画像ID．

    Returns:
        PIL.Image.Image: RGB画像（W×H）．
    """
    return ex017_data().load_image(coco, image_id)


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
    coco_to_synset = ex017_data().COCO_TO_SYNSET
    coco = set(coco_to_synset.values())
    names = {n.replace(" ", "_") for n in coco_to_synset}
    if ancestors(synset_name) & coco or set(wn.synset(synset_name).lemma_names()) & names:
        return True
    return any(synset_name in ancestors(c) for c in coco)


def load_lvis():
    """
    LVISのアノテーションを画像ごとに読み込む（WordNet 3.0に存在するSynsetのみ）．

    Returns:
        dict[int, list[dict]]: 画像ID → {"box"（xyxy），"synset"，"name"（LVISのカテゴリ名），
                               "kind"（"coco" または "noncoco"）} のリスト．
    """
    with open(LVIS_PATH) as f:
        lvis = json.load(f)
    category = {}
    for c in lvis["categories"]:
        try:
            category[c["id"]] = (wn.synset(c["synset"]).name(), c["name"])
        except Exception:  # WordNet 3.0に存在しないSynset名
            continue
    per_image = {}
    for a in lvis["annotations"]:
        if a["category_id"] not in category:
            continue
        x, y, w, h = a["bbox"]
        synset, name = category[a["category_id"]]
        per_image.setdefault(a["image_id"], []).append(
            {"box": [x, y, x + w, y + h], "synset": synset, "name": name,
             "kind": "coco" if is_coco_object(synset) else "noncoco"})
    return per_image


def test_image_ids(lvis):
    """
    ex017のテスト用の画像のうち，LVISの注釈がある画像のIDを返す．

    Args:
        lvis (dict): load_lvis の戻り値．

    Returns:
        list[int]: 画像IDのリスト（ex017の順序）．
    """
    with open(os.path.join(EX017_DATASET_DIR, "splits.json")) as f:
        splits = json.load(f)
    return [i for i in splits["test"] if i in lvis]


def gt_arrays(lvis, image_id):
    """
    1枚の画像のLVISのGTを配列にする．

    Args:
        lvis (dict): load_lvis の戻り値．
        image_id (int): 画像ID．

    Returns:
        tuple[np.ndarray, list[str], list[str]]: ボックス（G×4，xyxy），種類（"coco"/"noncoco"，長さG），カテゴリ名（長さG）．
    """
    items = lvis.get(image_id, [])
    boxes = np.array([g["box"] for g in items], np.float32).reshape(-1, 4)
    return boxes, [g["kind"] for g in items], [g["name"] for g in items]


def load_ex017_detections(detector_name):
    """
    ex017のテスト用の検出結果のキャッシュを読み込む．

    Args:
        detector_name (str): "YOLO11x" または "OWL-ViT"．

    Returns:
        dict[int, dict]: 画像ID → {"boxes"（K×4），"scores"（K），...}．
    """
    return torch.load(os.path.join(EX017_DATASET_DIR, f"detections_{detector_name}_test.pt"), weights_only=False)

