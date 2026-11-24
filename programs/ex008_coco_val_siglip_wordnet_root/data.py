"""
ex008のデータ定義：SigLIPへの変更，YOLOとVLMのスコアの比較，WordNetの同じ深さの語彙による深さ0の評価に用いるデータ．

- 画像の分割（dev500，test500，lvis_dev，lvis_test）と検出結果のキャッシュはex007と同じものを用いる．
- 深さ0の候補語彙（WordNetの同じ深さの語彙）:
  COCOの各カテゴリのSynset c について，最長の上位語の経路で k 段上の祖先 a をとり，
  a からちょうど k 段下にあるSynset（c と同じ深さの「いとこ」）を候補とする．
  物理的な実体（physical_entity.n.01の下位語）に限り，COCOのカテゴリの上位語・下位語は粒度が異なるため除き，
  SemCorの出現頻度が min_count 以上の語に限る．
- COCO外の物体の評価インスタンス:
  LVISのアノテーションのうち，SynsetがCOCOのカテゴリのSynsetの上位語・下位語（自身を含む）でなく，
  レンマ名もCOCOのカテゴリ名と一致しない物体に対応し，
  COCOのGTとは対応しない検出ボックス．
"""
import functools
import importlib.util
import json
import os
import sys

import numpy as np
from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import gt_instances  # noqa: E402
from common.metrics import box_iou, match_boxes  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET, EXTRA_TO_SYNSET, synset_word  # noqa: E402

PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX007 = "ex007_coco_val_hierarchy_eval"
EX007_DIR = os.path.join(PROGRAM_DIR, EX007)
DATA_DIR = os.path.join(DATASET_DIR, "ex008_coco_val_siglip_wordnet_root")
PHYSICAL_ENTITY = "physical_entity.n.01"


def import_experiment_module(experiment, module):
    """
    他の実験ディレクトリのモジュール（data.py，model.py等）を，名前の衝突を避けて読み込む．

    Args:
        experiment (str): 実験ディレクトリ名（例: "ex006_coco_val500_yolo_clip_hybrid"）．
        module (str): モジュール名（例: "data"）．

    Returns:
        module: 読み込んだモジュール．
    """
    path = os.path.join(PROGRAM_DIR, experiment, f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"{experiment}_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.lru_cache(maxsize=None)
def load_ex007():
    """
    ex007の data，model，train モジュールを読み込む．
    ex007の train.py は data，model という名前でモジュールを読み込むため，読み込み中のみex007のディレクトリを
    sys.path の先頭に置き，終了後に sys.path と sys.modules をex008の状態に戻す．

    Returns:
        tuple[module, module, module]: ex007の data，model，train．
    """
    saved = {name: sys.modules.pop(name) for name in ("data", "model") if name in sys.modules}
    sys.path.insert(0, EX007_DIR)
    try:
        train = import_experiment_module(EX007, "train")
        data, model = sys.modules["data"], sys.modules["model"]
    finally:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != EX007_DIR]
        for name in ("data", "model"):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return data, model, train


def coco_synsets():
    """
    COCOの80クラスのSynsetの集合を返す．

    Returns:
        set[Synset]: Synsetの集合（80）．
    """
    return {wn.synset(s) for s in COCO_TO_SYNSET.values()}


@functools.lru_cache(maxsize=None)
def ancestors(synset_name):
    """
    Synsetの真の上位語（インスタンスの上位語を含まない）の集合を返す．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: 上位語のSynset名の集合．
    """
    return frozenset(s.name() for s in wn.synset(synset_name).closure(lambda x: x.hypernyms()))


def descendants_at(synset, k):
    """
    Synsetからちょうど k 段下にある下位語の集合を返す．

    Args:
        synset (Synset): 起点．
        k (int): 段数．

    Returns:
        set[Synset]: k 段下のSynsetの集合．
    """
    level = {synset}
    for _ in range(k):
        level = {h for s in level for h in s.hyponyms()}
    return level


def same_depth_synsets(coco_synset, k):
    """
    COCOのカテゴリのSynsetと同じ深さの「いとこ」（最長の上位語の経路で k 段上の祖先から k 段下のSynset）を返す．

    Args:
        coco_synset (Synset): COCOのカテゴリのSynset．
        k (int): 祖先までの段数．

    Returns:
        set[Synset]: 候補のSynsetの集合（coco_synset 自身を含む）．
    """
    path = max(coco_synset.hypernym_paths(), key=len)
    ancestor = path[max(0, len(path) - 1 - k)]
    return descendants_at(ancestor, k) | {coco_synset}


def build_wordnet_root_vocab(k, min_count=1):
    """
    WordNetのCOCOのカテゴリと同じ深さの語彙を，深さ0の候補語彙として作る（キャッシュする）．

    Args:
        k (int): 祖先までの段数．
        min_count (int): SemCorの出現頻度（全レンマの合計）の下限．

    Returns:
        list[dict]: {"word"（プロンプトに入れる語），"synset"，"coco"（COCOのカテゴリ名または None）} のリスト．
            COCOのカテゴリが先頭の80件で，以降はSynset名の順．表記が重複する語は先頭のみ残す．
    """
    path = os.path.join(DATA_DIR, f"wordnet_root_vocab_k{k}_min{min_count}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    coco = coco_synsets()
    coco_names = {s.name() for s in coco}
    candidates = set()
    for c in coco:
        candidates |= same_depth_synsets(c, k)
    vocab = [{"word": name, "synset": syn, "coco": name} for name, syn in COCO_TO_SYNSET.items()]
    words = {v["word"] for v in vocab}
    for s in sorted(candidates - coco, key=lambda x: x.name()):
        if sum(lemma.count() for lemma in s.lemmas()) < min_count:
            continue
        # 物理的な実体でない語（action，ability等）は物体の名前にならないため除く
        if PHYSICAL_ENTITY not in ancestors(s.name()):
            continue
        # COCOのカテゴリの上位語（粒度が粗い）と下位語（粒度が細かい）は除く
        if ancestors(s.name()) & coco_names or any(s.name() in ancestors(c.name()) for c in coco):
            continue
        word = synset_word(s)
        if word in words:
            continue
        words.add(word)
        vocab.append({"word": word, "synset": s.name(), "coco": None})
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f, indent=1)
    return vocab


def coco_extra_vocab():
    """
    ex006，ex007の切替に用いた102語（COCOの80クラスとCOCO外の22語）の語彙を返す．

    Returns:
        list[dict]: {"word"，"synset"，"coco"} のリスト（長さ102）．
    """
    return ([{"word": n, "synset": s, "coco": n} for n, s in COCO_TO_SYNSET.items()]
            + [{"word": n, "synset": s, "coco": None} for n, s in EXTRA_TO_SYNSET.items()])


def coco_vocab():
    """
    COCOの80クラスの語彙を返す．

    Returns:
        list[dict]: {"word"，"synset"，"coco"} のリスト（長さ80）．
    """
    return [{"word": n, "synset": s, "coco": n} for n, s in COCO_TO_SYNSET.items()]


@functools.lru_cache(maxsize=None)
def is_non_coco_synset(synset_name):
    """
    SynsetがCOCOのいずれのカテゴリにも属さない物体を表すかを判定する．
    COCOのカテゴリのSynsetおよびその上位語・下位語と，レンマ名がCOCOのカテゴリ名と一致するSynset
    （例：LVISの broccoli.n.01 とCOCOの broccoli.n.02 のように語義番号のみが異なるもの）はCOCOの物体とみなす．

    Args:
        synset_name (str): Synset名．

    Returns:
        bool: COCO外の物体のSynsetであるか（WordNetに存在しない名前は False）．
    """
    try:
        synset = wn.synset(synset_name)
    except Exception:  # WordNet 3.0に存在しないSynset名
        return False
    coco_names = set(COCO_TO_SYNSET.values())
    coco_words = {name.replace(" ", "_") for name in COCO_TO_SYNSET} | {wn.synset(s).lemma_names()[0] for s in coco_names}
    if synset.name() in coco_names or ancestors(synset.name()) & coco_names:
        return False
    if set(synset.lemma_names()) & coco_words:
        return False
    return not any(synset.name() in ancestors(c) for c in coco_names)


def build_noncoco_instances(coco, image_ids, detections, min_score, lvis, iou_threshold=0.5):
    """
    COCO外の物体（LVISのアノテーション）に対応し，COCOのGTとは対応しない検出ボックスを評価インスタンスとする．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        image_ids (list[int]): 画像IDのリスト．
        detections (dict): ex007の detect_detr / detect_yolo の戻り値．
        min_score (float): 検出スコアの閾値．
        lvis (dict[int, list[dict]]): ex007の load_lvis の戻り値．
        iou_threshold (float): 対応付けのIoU閾値．

    Returns:
        list[dict]: {"image_id"，"box"，"det_score"，"lvis_synset"，("yolo_word")} のリスト．
    """
    instances = []
    for image_id in image_ids:
        det = detections[str(image_id)]
        anns = [a for a in lvis.get(image_id, []) if is_non_coco_synset(a["synset"])]
        keep = [i for i, s in enumerate(det["scores"]) if s > min_score]
        if not anns or not keep:
            continue
        boxes = np.array([det["boxes"][i] for i in keep], dtype=np.float32).reshape(-1, 4)
        gt_boxes, _ = gt_instances(coco, image_id)
        if len(gt_boxes):
            # COCOのGTと対応するボックスはCOCOのカテゴリの評価に含まれるため除く
            on_coco = box_iou(boxes, np.asarray(gt_boxes, dtype=np.float32)).max(axis=1) >= iou_threshold
        else:
            on_coco = np.zeros(len(boxes), dtype=bool)
        lvis_boxes = np.array([a["box"] for a in anns], dtype=np.float32)
        for p, g, _ in match_boxes(boxes, lvis_boxes, iou_threshold):
            if on_coco[p]:
                continue
            k = keep[p]
            ins = {"image_id": image_id, "gt_index": -1 - g, "gt_name": None, "box": det["boxes"][k],
                   "det_score": det["scores"][k], "lvis_synset": anns[g]["synset"]}
            if "top5" in det:
                ins["yolo_word"] = det["top5"][k][0]
            instances.append(ins)
    return instances
