"""
ex020：データの読み込みと，細かい粒度の正解の作成（order_031）．

- 画像と深さ0の正解：ex017・ex019 と同じ COCO val2017 の dev 500枚，test 500枚（COCO の80クラス）．MSCOCO 以外は用いない
- 細かい粒度の正解：同じ画像の COCO のキャプション（captions_val2017，1枚5文）から作る．
  1. spaCy（en_core_web_sm）で名詞を取り出す．直前の語と合わせた2語の複合語（例："tennis player"，"hot dog"）が
     WordNet にあれば，それを優先する．
  2. 語を WordNet の第1語義（最も頻度の高い語義）の Synset に変換する（"white"，"hand" のような別の語義の混入を防ぐ）．
  3. 正解クラスの Synset の真の下位語になる Synset を，そのクラスのインスタンスの細かい正解とする（例：person → man.n.01）．
  キャプションはインスタンスと対応しないため，同じ画像の同じクラスのインスタンスは細かい正解の集合を共有する．
- 階層的評価の正解の集合 G：クラスの Synset から細かい正解の Synset までの WordNet の上位語の経路上の Synset（両端を含む）．
COCO の読み込み・分割・正解は ex017 の data.py，検出と特徴のキャッシュは ex019 を用いる（load_module で読み込む）．
"""
import functools
import importlib
import json
import os
import sys

import numpy as np
from nltk.corpus import wordnet as wn

EX_NAME = "ex020_coco_owl_alphaclip_granularity"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", EX_NAME)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", EX_NAME)
PROGRAM_DIRS = {"ex017": os.path.join(PROJECT_ROOT, "programs", "ex017_coco_granular_annotation"),
                "ex019": os.path.join(PROJECT_ROOT, "programs", "ex019_coco_owl_alphaclip_wordnet")}


@functools.lru_cache(maxsize=None)
def load_module(experiment):
    """
    他の実験の data，model，train モジュールを読み込む．
    ex020 自身の data，model，train と名前が衝突するため，一時的に sys.modules から外して読み込み，後で戻す．

    Args:
        experiment (str): "ex017" または "ex019"．

    Returns:
        tuple[module, module, module]: data，model，train．
    """
    names = ("data", "model", "train")
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, PROGRAM_DIRS[experiment])
    try:
        modules = tuple(importlib.import_module(n) for n in names)
    finally:
        sys.path.remove(PROGRAM_DIRS[experiment])
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return modules


def ex017_data():
    """
    Returns:
        module: ex017 の data モジュール．
    """
    return load_module("ex017")[0]


def class_synsets():
    """
    COCO の80クラス名から Synset 名への対応を返す．

    Returns:
        dict[str, str]: クラス名 → Synset 名（例："person" → "person.n.01"）．
    """
    return dict(ex017_data().COCO_TO_SYNSET)


def first_sense(word):
    """
    語の WordNet の名詞の第1語義を返す（語形の変化は WordNet の morphy で戻す）．

    Args:
        word (str): 語（複合語は "_" で連結．例："tennis_player"）．

    Returns:
        str | None: Synset 名．WordNet に無い場合は None．
    """
    synsets = wn.synsets(word, pos="n")
    return synsets[0].name() if synsets else None


def caption_noun_synsets(captions, nlp):
    """
    1枚の画像のキャプションの名詞を，WordNet の第1語義の Synset に変換する．

    Args:
        captions (list[str]): キャプション（長さ5程度）．
        nlp: spaCy の言語モデル．

    Returns:
        list[str]: Synset 名のユニークなリスト（出現順）．
    """
    out = []
    for doc in nlp.pipe(captions):
        tokens = list(doc)
        for k, t in enumerate(tokens):
            if t.pos_ not in ("NOUN", "PROPN") or not t.is_alpha:
                continue
            head = t.lemma_.lower()
            compound = f"{tokens[k - 1].text.lower()}_{head}" if k > 0 and tokens[k - 1].is_alpha else None
            s = (first_sense(compound) if compound else None) or first_sense(head)
            if s and s not in out:
                out.append(s)
    return out


def caption_nouns(coco, image_ids, tag):
    """
    画像ごとのキャプションの名詞の Synset を求める（DATASET_DIR にキャッシュする）．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．
        tag (str): キャッシュのファイル名の識別子（分割の名前）．

    Returns:
        dict[int, list[str]]: 画像ID → Synset 名のリスト．
    """
    path = os.path.join(DATASET_DIR, f"caption_nouns_{tag}.json")
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    import spacy
    nlp = spacy.load("en_core_web_sm")
    captions = {}
    for a in json.load(open(ex017_data().COCO_CAPTIONS_PATH))["annotations"]:
        captions.setdefault(a["image_id"], []).append(a["caption"])
    out = {i: caption_noun_synsets(captions.get(i, []), nlp) for i in image_ids}
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


@functools.lru_cache(maxsize=None)
def descendants(synset_name):
    """
    Synset の真の下位語の集合（推移閉包．インスタンスの下位語は含めない）．

    Args:
        synset_name (str): Synset 名．

    Returns:
        frozenset[str]: 下位語の Synset 名の集合．
    """
    return frozenset(s.name() for s in wn.synset(synset_name).closure(lambda x: x.hyponyms()))


def fine_labels(class_name, nouns):
    """
    キャプションの名詞のうち，正解クラスの真の下位語になるもの（細かい粒度の正解）を返す．

    Args:
        class_name (str): COCO のクラス名．
        nouns (list[str]): caption_nouns の1枚分．

    Returns:
        list[str]: 細かい正解の Synset 名のリスト．
    """
    below = descendants(class_synsets()[class_name])
    return [s for s in nouns if s in below]


@functools.lru_cache(maxsize=None)
def gt_paths(root, fine):
    """
    クラスの Synset から細かい正解の Synset までの上位語の経路（両端を含む）を全て返す．

    Args:
        root (str): クラスの Synset 名．
        fine (str): 細かい正解の Synset 名（root の下位語）．

    Returns:
        list[frozenset[str]]: 経路上の Synset 名の集合のリスト（重複は除く）．
    """
    paths = set()
    for p in wn.synset(fine).hypernym_paths():
        names = [s.name() for s in p]
        if root in names:
            paths.add(frozenset(names[names.index(root):]))
    return sorted(paths, key=len)


def instance_targets(class_name, fine):
    """
    1つのインスタンスの階層的評価の正解（細かい正解ごとの経路の集合）を返す．

    Args:
        class_name (str): COCO のクラス名．
        fine (list[str]): fine_labels の戻り値．

    Returns:
        list[tuple[str, frozenset[str]]]: (細かい正解の Synset 名, 経路の Synset 名の集合) のリスト．
    """
    root = class_synsets()[class_name]
    return [(f, p) for f in fine for p in gt_paths(root, f)]


@functools.lru_cache(maxsize=None)
def ordered_gt_path(root, fine):
    """
    クラスの Synset から細かい正解の Synset までの最短の上位語の経路を，上から順に返す．

    Args:
        root (str): クラスの Synset 名．
        fine (str): 細かい正解の Synset 名（root の下位語）．

    Returns:
        list[str]: Synset 名のリスト（先頭が root，末尾が fine）．
    """
    paths = []
    for p in wn.synset(fine).hypernym_paths():
        names = [s.name() for s in p]
        if root in names:
            paths.append(names[names.index(root):])
    return min(paths, key=len)


# ---------------------------------------------------------------------------
# order_032：VG.pdf の語彙（Visual Genome の頻度）と，枠と WordNet による木
# ---------------------------------------------------------------------------
VG_DIR = os.path.join(PROJECT_ROOT, "datasets", "visual_genome")   # objects.json（v1.2）と image_data.json
COCO_VAL_ANNOTATIONS = os.path.join(PROJECT_ROOT, "datasets", "coco", "annotations", "instances_val2017.json")
VG_MIN_COUNT = 10       # VG.pdf の n_min
VG_MIN_DEPTH = 5        # VG.pdf の式(2)の深さの条件
EVAL_MIN_DEPTH = 4      # VG.pdf の式(8)の深さの条件
# 最上位の枠（VG.pdf と同じく J = 40．VG.pdf に一覧が無いため，VG.pdf に挙がる clothing，furniture，animal，light，
# natural object を含めて定めた．VG の頻出語が仮想的な枠に入らないよう，各枠の Synset を加えた）．各枠は1つ以上の Synset の集合である．
FRAMES = {
    "person": ["person.n.01"], "animal": ["animal.n.01"], "plant": ["plant.n.02"], "food": ["food.n.01", "food.n.02"],
    "fruit and vegetable": ["edible_fruit.n.01", "vegetable.n.01"], "beverage": ["beverage.n.01"],
    "clothing": ["clothing.n.01"], "footwear": ["footwear.n.02"], "headwear": ["headdress.n.01"],
    "accessory": ["accessory.n.01", "jewelry.n.01"], "bag": ["bag.n.01", "bag.n.04"], "container": ["container.n.01"],
    "furniture": ["furniture.n.01"], "vehicle": ["vehicle.n.01", "craft.n.02", "conveyance.n.03"], "building": ["building.n.01"],
    "structure": ["structure.n.01"], "room and area": ["room.n.01", "area.n.05", "surface.n.01", "facility.n.01"], "body part": ["body_part.n.01"],
    "device": ["device.n.01"], "electronics": ["electronic_equipment.n.01", "electronic_device.n.01"],
    "machine": ["machine.n.01"], "kitchenware": ["kitchen_utensil.n.01", "cookware.n.01"],
    "tableware": ["tableware.n.01", "cutlery.n.02"], "tool": ["tool.n.01"], "sports equipment": ["sports_equipment.n.01"],
    "toy": ["toy.n.01", "plaything.n.01"], "light": ["source_of_illumination.n.01"], "sign": ["sign.n.02", "signboard.n.01", "written_communication.n.01", "symbol.n.01", "signal.n.01", "character.n.08", "numeral.n.01"],
    "covering": ["covering.n.02"], "decoration": ["decoration.n.01", "creation.n.02"], "natural object": ["natural_object.n.01"],
    "geological formation": ["geological_formation.n.01"], "body of water": ["body_of_water.n.01"],
    "land": ["land.n.04", "ground.n.01", "geographical_area.n.01"], "sky and weather": ["sky.n.01", "atmospheric_phenomenon.n.01", "physical_phenomenon.n.01"],
    "road and path": ["way.n.06", "road.n.01"], "material": ["material.n.01", "fabric.n.01"],
    "publication": ["paper.n.01", "publication.n.01"], "musical instrument": ["musical_instrument.n.01"],
    "equipment and other artifact": ["equipment.n.01", "implement.n.01", "instrumentality.n.03", "artifact.n.01"],
}
OTHER_FRAME = "other"   # どの枠にも含まれない語をまとめる仮想的な枠


def vg_counts():
    """
    Visual Genome の物体の注釈の Synset の出現回数と，Synset ごとの注釈者の名前の回数を求める（DATASET_DIR にキャッシュする）．
    COCO val2017 と重複する画像（image_data.json の coco_id で判定）の注釈は集計から除く（VG.pdf 4.1節）．

    Returns:
        dict: "counts"（Synset 名 → 回数），"names"（Synset 名 → 最も多い名前），"n_images"，"n_excluded"．
    """
    path = os.path.join(DATASET_DIR, "vg_counts.json")
    if os.path.exists(path):
        return json.load(open(path))
    import collections
    coco_val = {im["id"] for im in json.load(open(COCO_VAL_ANNOTATIONS))["images"]}
    excluded = {im["image_id"] for im in json.load(open(os.path.join(VG_DIR, "image_data.json"))) if im.get("coco_id") in coco_val}
    counts, names = collections.Counter(), collections.defaultdict(collections.Counter)
    objects = json.load(open(os.path.join(VG_DIR, "objects.json")))
    for im in objects:
        if im["image_id"] in excluded:
            continue
        for o in im["objects"]:
            for s in o.get("synsets", []):
                counts[s] += 1
                if o.get("names"):
                    names[s][o["names"][0].lower().strip()] += 1
    out = {"counts": dict(counts), "names": {s: c.most_common(1)[0][0] for s, c in names.items()},
           "n_images": len(objects), "n_excluded": len(excluded)}
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def vg_vocab(min_count=VG_MIN_COUNT, min_depth=VG_MIN_DEPTH):
    """
    VG の頻度に基づく語彙 V = {s | n(s) ≥ n_min, d(s) ≥ 5}（VG.pdf の式(2)）．プロンプトの語は VG で最も多い名前とする．

    Args:
        min_count (int): 最小の出現回数 n_min．
        min_depth (int): WordNet の根からの最短の深さの下限．

    Returns:
        dict[str, str]: Synset 名 → プロンプトの語．
    """
    vg = vg_counts()
    out = {}
    for s, n in vg["counts"].items():
        if n < min_count or ".n." not in s:
            continue
        try:
            if wn.synset(s).min_depth() >= min_depth:
                out[s] = vg["names"].get(s, wn.synset(s).lemma_names()[0].replace("_", " "))
        except Exception:   # 古い WordNet の Synset 名で現在の WordNet に無いもの
            continue
    return out


@functools.lru_cache(maxsize=None)
def hypernym_distances(synset_name):
    """
    Synset から各上位語（自身を含む）までの最短の段数．

    Args:
        synset_name (str): Synset 名．

    Returns:
        dict[str, int]: 上位語の Synset 名 → 段数．
    """
    out = {}
    for s, d in wn.synset(synset_name).hypernym_distances():
        out[s.name()] = min(d, out.get(s.name(), d))
    return out


def assign_frame(synset_name):
    """
    語を最上位の枠に割り当てる（VG.pdf の式(3)．上位語をたどって枠の Synset に至る段数が最小の枠）．

    Args:
        synset_name (str): Synset 名．

    Returns:
        str: 枠の名前（どの枠にも含まれない場合は OTHER_FRAME）．
    """
    dist = hypernym_distances(synset_name)
    best, best_d = OTHER_FRAME, np.inf
    for name, synsets in FRAMES.items():
        d = min((dist[f] for f in synsets if f in dist), default=np.inf)
        if d < best_d:
            best, best_d = name, d
    return best


def build_tree():
    """
    VG の語彙と COCO の80クラスの Synset を，枠と WordNet の上下関係で木にする（VG.pdf 4.2節の「枠＋WordNet」）．
    枠の中の各語は，上位語をたどって最初に出会う同じ枠の語彙の語を親とする（候補が複数なら根からの深さが最小で，
    VG の出現回数が最大のもの）．親が無い語は枠を親とする．結果は DATASET_DIR にキャッシュする．

    Returns:
        dict: "nodes"（ノード名のリスト．枠は "frame:名前"），"words"（ノード名 → プロンプトの語．枠は None），
              "synset"（ノード名 → 正誤判定に用いる Synset 名．枠は先頭の Synset，仮想的な枠は None），
              "parent"（ノード名 → 親のノード名．枠の親は "root"），"depth"（根からの段数．枠は1），"frame"．
    """
    path = os.path.join(DATASET_DIR, "vg_tree.json")
    if os.path.exists(path):
        return json.load(open(path))
    counts = vg_counts()["counts"]
    words = vg_vocab()
    words.update({s: c for c, s in class_synsets().items()})        # COCO のクラスは COCO の名前で必ず含める
    frame = {s: assign_frame(s) for s in words}
    parent = {}
    for s in words:
        cands = [(d, wn.synset(a).min_depth(), -counts.get(a, 0), a) for a, d in hypernym_distances(s).items()
                 if a != s and a in words and frame[a] == frame[s]]
        parent[s] = min(cands)[3] if cands else f"frame:{frame[s]}"
    frames = [f"frame:{f}" for f in [*FRAMES, OTHER_FRAME]]
    tree = {"nodes": frames + sorted(words), "words": {**dict.fromkeys(frames), **words},
            "synset": {**{f"frame:{k}": v[0] for k, v in FRAMES.items()}, f"frame:{OTHER_FRAME}": None, **{s: s for s in words}},
            "parent": {**dict.fromkeys(frames, "root"), **parent}, "frame": {**{f: f[6:] for f in frames}, **frame}}
    depth = {}

    def node_depth(n):
        """ノードの根からの段数を求める（枠は1）．"""
        if n not in depth:
            depth[n] = 1 if tree["parent"][n] == "root" else node_depth(tree["parent"][n]) + 1
        return depth[n]
    tree["depth"] = {n: node_depth(n) for n in tree["nodes"]}
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tree, f)
    return tree


@functools.lru_cache(maxsize=None)
def eval_set(synset_name):
    """
    階層的評価の集合：Synset 自身と全ての上位語のうち深さ4以上のもの（VG.pdf の式(8)）．
    COCO の80クラスの Synset は深さによらず含める（person.n.01 の深さは3のため）．

    Args:
        synset_name (str): Synset 名．

    Returns:
        frozenset[str]: Synset 名の集合．
    """
    coco = set(class_synsets().values())
    return frozenset(a for a in hypernym_distances(synset_name) if wn.synset(a).min_depth() >= EVAL_MIN_DEPTH or a in coco)
