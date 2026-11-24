"""
ex015のデータ定義：再学習なしで未学習の物体と語彙を検出・命名する粒度可変アノテーションのやり直し（order_019）．

- 画像と分割: ex013・ex014と同じ（COCO 2017 val のうちLVIS v1のアノテーションがある画像．検証用 dev，テスト用 test）
- GT: LVISのアノテーション（ボックス，WordNetのSynset，カテゴリID，頻度 r/c/f）．
      「COCOの物体」「COCO外の物体」の区別はex013と同じ（ex013の is_coco_object）
- LVISの画像ごとの情報: 存在しないと確認されたカテゴリ（neg），網羅的に注釈されていないカテゴリ（not_exhaustive）
- 領域のキャッシュ: OWLv2を各画像に1回適用し，物体らしさの上位 MAX_REGIONS 件の領域（ボックス，スコア，領域の特徴）と，
      YOLO11xの各ボックスにIoUが最大の領域の特徴を保存する
- 語彙: ex014と同じ，COCO train2017のキャプションに5回以上出現するレンマを持つ物理的な実体のSynset（7806語）に，
      キャプションでの出現回数を加えたもの
"""
import collections
import functools
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from torchvision.ops import box_iou as tv_box_iou
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.paths import COCO_DIR, DATASET_DIR  # noqa: E402

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX013 = "ex013_open_vocabulary_granular"
DATA_DIR = os.path.join(DATASET_DIR, "ex015_lvis_open_vocabulary_annotation")
LVIS_PATH = os.path.join(DATASET_DIR, "lvis", "lvis_v1_val.json")
SPLITS = ["dev", "test"]
YOLO_SCORE = 0.25
MAX_REGIONS = 100
ROOT = "physical_entity.n.01"
CAPTION_MIN_COUNT = 5


def _load(path, name):
    """
    ファイルからモジュールを読み込む．

    Args:
        path (str): ファイルのパス．
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def ex013_modules():
    """
    ex013の data，model，train を読み込む．ex013の model，train は data，model という名前でモジュールを読み込むため，
    読み込み中のみ sys.modules をex013のものに置き換え，終了後に元に戻す．

    Returns:
        tuple[module, module, module]: ex013の data，model，train．
    """
    directory = os.path.join(PROGRAM_DIR, EX013)
    saved = {n: sys.modules.pop(n) for n in ("data", "model") if n in sys.modules}
    try:
        d13 = _load(os.path.join(directory, "data.py"), "ex013_data")
        sys.modules["data"] = d13
        m13 = _load(os.path.join(directory, "model.py"), "ex013_model")
        sys.modules["model"] = m13
        t13 = _load(os.path.join(directory, "train.py"), "ex013_train")
    finally:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != directory]
        for n in ("data", "model"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return d13, m13, t13


def ancestors(synset_name):
    """
    Synset自身と全ての上位語の集合を返す（ex013と同じ定義）．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    return ex013_modules()[0].ancestors(synset_name)


def load_lvis():
    """
    LVISのアノテーションと，画像ごとの評価に用いる情報を読み込む（WordNet 3.0に存在するSynsetのカテゴリのみ）．

    Returns:
        dict: "gts"（画像ID → {"box"，"synset"，"kind"，"category"，"freq"} のリスト），
            "neg"・"not_exhaustive"（画像ID → カテゴリIDの集合），"category_synset"（カテゴリID → Synset名）．
    """
    d13 = ex013_modules()[0]
    with open(LVIS_PATH) as f:
        lvis = json.load(f)
    synset_of, freq_of = {}, {}
    for c in lvis["categories"]:
        try:
            synset_of[c["id"]] = wn.synset(c["synset"]).name()
        except Exception:  # WordNet 3.0に存在しないSynset名
            continue
        freq_of[c["id"]] = c["frequency"]
    gts = {}
    for a in lvis["annotations"]:
        c = a["category_id"]
        if c not in synset_of:
            continue
        x, y, w, h = a["bbox"]
        s = synset_of[c]
        gts.setdefault(a["image_id"], []).append({"box": [x, y, x + w, y + h], "synset": s, "category": c, "freq": freq_of[c],
                                                  "kind": "coco" if d13.is_coco_object(s) else "noncoco"})
    neg = {im["id"]: set(im.get("neg_category_ids", [])) for im in lvis["images"]}
    not_exhaustive = {im["id"]: set(im.get("not_exhaustive_category_ids", [])) for im in lvis["images"]}
    return {"gts": gts, "neg": neg, "not_exhaustive": not_exhaustive, "category_synset": synset_of}


def load_context(seed=42):
    """
    COCO，LVIS，分割，YOLO11xの検出結果を読み込む（分割とYOLO11xの検出結果はex013・ex012と同じもの）．

    Args:
        seed (int): 分割の乱数シード．

    Returns:
        dict: "coco"，"lvis"（load_lvis の戻り値），"splits"，"yolo"．
    """
    d13 = ex013_modules()[0]
    base = d13.load_context(seed)
    return {"coco": base["coco"], "lvis": load_lvis(), "splits": base["splits"], "yolo": base["yolo"]}


def yolo_boxes(ctx, image_id):
    """
    1枚の画像のYOLO11xのボックス（検出スコアが YOLO_SCORE 以上）を返す．

    Args:
        ctx (dict): load_context の戻り値．
        image_id (int | str): 画像ID．

    Returns:
        list[dict]: {"box"，"score"，"source"（"yolo"），"label"（COCOのクラス名）}．
    """
    d = ctx["yolo"][str(image_id)]
    return [{"box": b, "score": s, "source": "yolo", "label": l}
            for b, s, l in zip(d["boxes"], d["scores"], d["labels"]) if s >= YOLO_SCORE]


def region_cache(ctx, split, model_key, owl_factory):
    """
    分割の画像にOWLv2を1回ずつ適用し，領域（ボックス，物体らしさ，領域の特徴）とYOLO11xのボックスに対応する領域の特徴をキャッシュする．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．
        model_key (str): OWLv2の重みの名前（キャッシュのファイル名に用いる）．
        owl_factory (callable): キャッシュが無い場合に OWLv2Regions を返す関数．

    Returns:
        dict: 画像ID（str）→ {"boxes"（K×4），"scores"（K），"feats"（K×D，半精度），
            "yolo_feats"（YOLO11xのボックス数×D，半精度），"yolo_iou"（対応する領域とのIoU）}．
    """
    ids = ctx["splits"][split]
    path = os.path.join(DATA_DIR, f"regions_{model_key}_{split}_n{len(ids)}.pt")
    if os.path.exists(path):
        return torch.load(path)
    owl = owl_factory()
    out = {}
    for i in tqdm(ids, desc=f"{model_key} regions {split}", leave=False):
        boxes, scores, feats, keep = owl.forward(load_image(ctx["coco"], i))
        yolo = torch.tensor([b["box"] for b in yolo_boxes(ctx, i)], dtype=torch.float32, device=boxes.device).reshape(-1, 4)
        if len(yolo):
            iou, best = tv_box_iou(yolo, boxes.float()).max(dim=1)
        else:
            iou, best = torch.zeros(0), torch.zeros(0, dtype=torch.long, device=boxes.device)
        out[str(i)] = {"boxes": boxes[keep].tolist(), "scores": scores[keep].tolist(), "feats": feats[keep].half().cpu(),
                       "yolo_feats": feats[best].half().cpu(), "yolo_iou": iou.tolist()}
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(out, path)
    return out


@functools.lru_cache(maxsize=None)
def caption_counts():
    """
    COCO train2017のキャプションにおける語（1語と2語の連続，名詞はレンマ化）の出現回数を返す（ex013と同じ数え方）．

    Returns:
        dict[str, int]: 小文字の語 → 出現回数（CAPTION_MIN_COUNT 回以上のもの）．
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


@functools.lru_cache(maxsize=None)
def caption_vocabulary():
    """
    命名の語彙（キャプションに出現するレンマを持つ，物理的な実体のSynset）を返す．

    Returns:
        list[dict]: {"synset"，"word"（キャプションに出現する先頭のレンマ），"gloss"（定義文），
            "count"（その語のキャプションでの出現回数），"person"（person.n.01 の真の下位語か）} のリスト（Synset名の順）．
    """
    counts = caption_counts()
    vocab = []
    for s in sorted(wn.synset(ROOT).closure(lambda x: x.hyponyms()), key=lambda x: x.name()):
        lemmas = [lemma.replace("_", " ") for lemma in s.lemma_names()]
        word = next((lemma for lemma in lemmas if lemma.lower() in counts), None)
        if word is None:
            continue
        anc = ancestors(s.name())
        vocab.append({"synset": s.name(), "word": word, "gloss": s.definition(), "count": counts[word.lower()],
                      "person": "person.n.01" in anc and s.name() != "person.n.01"})
    return vocab


@functools.lru_cache(maxsize=None)
def lvis_vocabulary():
    """
    上限の測定に用いるLVISのカテゴリの語彙を返す．

    Returns:
        list[dict]: {"synset"，"word"，"gloss"} のリスト．
    """
    with open(LVIS_PATH) as f:
        categories = json.load(f)["categories"]
    vocab = []
    for c in sorted(categories, key=lambda c: c["id"]):
        try:
            s = wn.synset(c["synset"])
        except Exception:  # WordNet 3.0に存在しないSynset名
            continue
        word = re.sub(r"_\(.*\)$", "", c["name"]).replace("_", " ")
        vocab.append({"synset": s.name(), "word": word, "gloss": s.definition()})
    return vocab


def build_images(ctx, split, regions):
    """
    分割の画像ごとに，出力ボックスの候補（YOLO11xのボックスと，それらと重ならないOWLv2の領域），GT，両者のIoU，領域の特徴をまとめる．

    Args:
        ctx (dict): load_context の戻り値．
        split (str): 分割名．
        regions (dict): region_cache の戻り値．

    Returns:
        list[dict]: 画像ごとの {"image_id"，"boxes"（YOLO11xが先，続いて物体らしさの順のOWLv2の領域），"n_yolo"，
            "gts"，"iou"（ボックス×GT），"feats"（ボックス×D，半精度），"neg"，"not_exhaustive"，"pos"}．
    """
    from common.metrics import box_iou

    images = []
    for i in ctx["splits"][split]:
        reg = regions[str(i)]
        boxes = yolo_boxes(ctx, i)
        n_yolo = len(boxes)
        base = np.asarray([b["box"] for b in boxes], dtype=np.float32).reshape(-1, 4)
        cand = np.asarray(reg["boxes"], dtype=np.float32).reshape(-1, 4)
        overlap = box_iou(cand, base).max(axis=1) if n_yolo and len(cand) else np.zeros(len(cand))
        extra = [k for k in range(len(cand)) if overlap[k] < 0.5]
        boxes += [{"box": reg["boxes"][k], "score": reg["scores"][k], "source": "owlv2"} for k in extra]
        feats = torch.cat([reg["yolo_feats"], reg["feats"][extra]]) if len(boxes) else torch.zeros(0, reg["feats"].shape[-1])
        gts = ctx["lvis"]["gts"][i]
        b = np.asarray([x["box"] for x in boxes], dtype=np.float32).reshape(-1, 4)
        g = np.asarray([x["box"] for x in gts], dtype=np.float32).reshape(-1, 4)
        iou = box_iou(b, g) if len(b) and len(g) else np.zeros((len(b), len(g)))
        images.append({"image_id": str(i), "boxes": boxes, "n_yolo": n_yolo, "gts": gts, "iou": iou, "feats": feats,
                       "neg": ctx["lvis"]["neg"].get(i, set()), "not_exhaustive": ctx["lvis"]["not_exhaustive"].get(i, set()),
                       "pos": {x["category"] for x in gts}})
    return images
