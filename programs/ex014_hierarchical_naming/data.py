"""
ex014のデータ定義：COCO外の物体の命名の改善（order_018）．

評価のデータ，出力ボックスの候補，評価指標はex013と同じものを用いる（ex013の data，model，train を読み込む）．
- 命名の語彙: WordNetの physical_entity.n.01 以下のSynsetのうち，いずれかのレンマがCOCO train2017のキャプションに
  5回以上出現するもの（ex013の caption_terms）
- 上限の測定に用いる語彙: LVIS v1のカテゴリ（WordNet 3.0に存在するSynset）
"""
import functools
import importlib.util
import json
import os
import re
import sys

from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import DATASET_DIR  # noqa: E402

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX013 = "ex013_open_vocabulary_granular"
DATA_DIR = os.path.join(DATASET_DIR, "ex014_hierarchical_naming")
LVIS_PATH = os.path.join(DATASET_DIR, "lvis", "lvis_v1_val.json")
ROOT = "physical_entity.n.01"


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


@functools.lru_cache(maxsize=None)
def caption_vocabulary():
    """
    命名の語彙（キャプションに出現するレンマを持つ，物理的な実体のSynset）を返す．

    Returns:
        list[dict]: {"synset"，"word"（キャプションに出現するレンマを優先），"gloss"（定義文）} のリスト（Synset名の順）．
    """
    d13, m13, _ = ex013_modules()
    path = os.path.join(DATA_DIR, "caption_vocabulary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    vocab = []
    for s in sorted(wn.synset(ROOT).closure(lambda x: x.hyponyms()), key=lambda x: x.name()):
        if m13.is_visible(s):
            vocab.append({"synset": s.name(), "word": m13.word_of(s, True), "gloss": s.definition()})
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)
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
