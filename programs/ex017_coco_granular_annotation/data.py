"""
ex017：データの読み込みと語彙の定義（order_027）．

`main.pdf` と `進捗報告_2026B.pdf` の実装に用いるデータを扱う．他の実験のモジュールは読み込まない．
- COCO 2017 val の画像・物体の注釈・キャプション（`datasets/coco/`）
- 検証用・テスト用の画像の分割（seed 42 で1000枚をサンプリングし，500枚ずつに分ける）
- ボックスにちょうど合わせた切り抜き画像（論文3.3節．余白なし）を黒で正方形にパディングしたもの
- 語彙：COCOの80クラス，COCO外の22語（論文4.6節），WordNetの上位語（進捗報告の定性評価），
  キャプションの名詞・動詞（論文4.2節のタスクB．spaCy）
- WordNetの直接の下位語による候補語彙（論文の式(3)，進捗報告の「𝒲 ← w_j* の下位語からなる集合」）
"""
import json
import os
import shutil

import numpy as np
from nltk.corpus import wordnet as wn
from PIL import Image

EX_NAME = "ex017_coco_granular_annotation"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COCO_DIR = os.path.join(PROJECT_ROOT, "datasets", "coco")
COCO_IMAGE_DIR = os.path.join(COCO_DIR, "val2017")
COCO_INSTANCES_PATH = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
COCO_CAPTIONS_PATH = os.path.join(COCO_DIR, "annotations", "captions_val2017.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", EX_NAME)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", EX_NAME)
OPEN_VOCAB_DIR = os.path.join(DATASET_DIR, "open_vocab")
# COCO外の画像（Wikimedia Commons）の取得元．初回のみ OPEN_VOCAB_DIR に複製する
OPEN_VOCAB_SOURCE_DIR = os.path.join(PROJECT_ROOT, "datasets", "ex004_open_vocab_qualitative")
OPEN_VOCAB_IMAGES = ["penguin", "helicopter", "kangaroo", "camel", "lighthouse_1", "lighthouse_2"]

N_PER_SPLIT = 500     # 検証用・テスト用の画像数（論文4.1節の500枚）
SPLIT_SEED = 42       # サンプリングの乱数シード（論文4.1節）

# COCO 80クラス → WordNet Synset（定義文を確認して手動で対応付けた）
COCO_TO_SYNSET = {
    "person": "person.n.01", "bicycle": "bicycle.n.01", "car": "car.n.01", "motorcycle": "motorcycle.n.01",
    "airplane": "airplane.n.01", "bus": "bus.n.01", "train": "train.n.01", "truck": "truck.n.01",
    "boat": "boat.n.01", "traffic light": "traffic_light.n.01", "fire hydrant": "fireplug.n.01",
    "stop sign": "street_sign.n.01", "parking meter": "parking_meter.n.01", "bench": "bench.n.01",
    "bird": "bird.n.01", "cat": "cat.n.01", "dog": "dog.n.01", "horse": "horse.n.01", "sheep": "sheep.n.01",
    "cow": "cattle.n.01", "elephant": "elephant.n.01", "bear": "bear.n.01", "zebra": "zebra.n.01",
    "giraffe": "giraffe.n.01", "backpack": "backpack.n.01", "umbrella": "umbrella.n.01",
    "handbag": "bag.n.04", "tie": "necktie.n.01", "suitcase": "bag.n.06", "frisbee": "frisbee.n.01",
    "skis": "ski.n.01", "snowboard": "snowboard.n.01", "sports ball": "ball.n.01", "kite": "kite.n.03",
    "baseball bat": "baseball_bat.n.01", "baseball glove": "baseball_glove.n.01",
    "skateboard": "skateboard.n.01", "surfboard": "surfboard.n.01", "tennis racket": "tennis_racket.n.01",
    "bottle": "bottle.n.01", "wine glass": "wineglass.n.01", "cup": "cup.n.01", "fork": "fork.n.01",
    "knife": "knife.n.01", "spoon": "spoon.n.01", "bowl": "bowl.n.01", "banana": "banana.n.02",
    "apple": "apple.n.01", "sandwich": "sandwich.n.01", "orange": "orange.n.01", "broccoli": "broccoli.n.02",
    "carrot": "carrot.n.03", "hot dog": "hotdog.n.02", "pizza": "pizza.n.01", "donut": "doughnut.n.02",
    "cake": "cake.n.03", "chair": "chair.n.01", "couch": "sofa.n.01", "potted plant": "pot_plant.n.01",
    "bed": "bed.n.01", "dining table": "dining_table.n.01", "toilet": "toilet.n.02",
    "tv": "television_receiver.n.01", "laptop": "laptop.n.01", "mouse": "mouse.n.04",
    "remote": "remote_control.n.01", "keyboard": "computer_keyboard.n.01",
    "cell phone": "cellular_telephone.n.01", "microwave": "microwave.n.02", "oven": "oven.n.01",
    "toaster": "toaster.n.02", "sink": "sink.n.01", "refrigerator": "electric_refrigerator.n.01",
    "book": "book.n.02", "clock": "clock.n.01", "vase": "vase.n.01", "scissors": "scissors.n.01",
    "teddy bear": "teddy.n.01", "hair drier": "hand_blower.n.01", "toothbrush": "toothbrush.n.01",
}

# COCOに含まれない追加語（論文4.6節，22語）→ WordNet Synset
EXTRA_TO_SYNSET = {
    "penguin": "penguin.n.01", "helicopter": "helicopter.n.01", "kangaroo": "kangaroo.n.01",
    "camel": "camel.n.01", "lighthouse": "lighthouse.n.01", "dolphin": "dolphin.n.02",
    "whale": "whale.n.02", "shark": "shark.n.01", "eagle": "eagle.n.01", "owl": "owl.n.01",
    "parrot": "parrot.n.01", "flamingo": "flamingo.n.01", "tiger": "tiger.n.02", "lion": "lion.n.01",
    "monkey": "monkey.n.01", "gorilla": "gorilla.n.01", "panda": "giant_panda.n.01", "deer": "deer.n.01",
    "fox": "fox.n.01", "rabbit": "rabbit.n.01", "squirrel": "squirrel.n.01", "castle": "castle.n.02",
}

# 語彙レベルが最も高い単語の集合（進捗報告の定性評価）．WordNetの最上位カテゴリ（noun.Tops）のうち物体を表す6語
WORDNET_TOPS = {
    "person": "person.n.01", "animal": "animal.n.01", "plant": "plant.n.02", "food": "food.n.02",
    "artifact": "artifact.n.01", "natural object": "natural_object.n.01",
}


def load_json(path):
    """
    JSONファイルを読み込む．

    Args:
        path (str): ファイルのパス．

    Returns:
        dict | list: 読み込んだ内容．
    """
    with open(path) as f:
        return json.load(f)


def load_coco():
    """
    COCO 2017 val の物体の注釈を読み込み，検索用の辞書を作る．

    Returns:
        dict: "images"（image_id → 画像情報の辞書），"categories"（category_id → クラス名），
              "anns"（image_id → 注釈のリスト），"class_names"（80クラス名のリスト，category_id順）．
    """
    raw = load_json(COCO_INSTANCES_PATH)
    anns = {}
    for a in raw["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    categories = {c["id"]: c["name"] for c in raw["categories"]}
    return {"images": {im["id"]: im for im in raw["images"]}, "categories": categories, "anns": anns,
            "class_names": [categories[k] for k in sorted(categories)]}


def split_image_ids(coco, n_per_split=N_PER_SPLIT, seed=SPLIT_SEED):
    """
    物体の注釈を1件以上持つ画像から seed 固定で 2×n_per_split 枚をサンプリングし，検証用とテスト用に分ける．

    Args:
        coco (dict): load_coco の戻り値．
        n_per_split (int): 1つの分割の画像数．
        seed (int): 乱数シード．

    Returns:
        dict[str, list[int]]: "dev"（θ の選択に用いる）と "test"（報告に用いる）の image_id のリスト．
    """
    ids = sorted(i for i in coco["images"] if any(not a["iscrowd"] for a in coco["anns"].get(i, [])))
    chosen = np.random.default_rng(seed).choice(ids, size=2 * n_per_split, replace=False).tolist()
    return {"dev": chosen[:n_per_split], "test": chosen[n_per_split:]}


def load_image(coco, image_id):
    """
    COCOの画像をRGBで読み込む．

    Args:
        coco (dict): load_coco の戻り値．
        image_id (int): 画像ID．

    Returns:
        PIL.Image.Image: RGB画像（W×H）．
    """
    return Image.open(os.path.join(COCO_IMAGE_DIR, coco["images"][image_id]["file_name"])).convert("RGB")


def gt_instances(coco, image_id):
    """
    画像の物体の正解（iscrowd=0）をxyxy形式で返す．

    Args:
        coco (dict): load_coco の戻り値．
        image_id (int): 画像ID．

    Returns:
        tuple[np.ndarray, list[str]]: ボックス（G×4，xyxy，ピクセル）とクラス名（長さG）．
    """
    anns = [a for a in coco["anns"].get(image_id, []) if not a["iscrowd"]]
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)], dtype=np.float32)
    return boxes.reshape(-1, 4), [coco["categories"][a["category_id"]] for a in anns]


def crop_box(image, box):
    """
    ボックスにちょうど合わせて画像を切り抜く（論文3.3節．余白なし）．

    Args:
        image (PIL.Image.Image): 入力画像．
        box (array-like): xyxy形式のボックス（長さ4）．

    Returns:
        PIL.Image.Image: 切り抜き画像（幅・高さは1ピクセル以上）．
    """
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0 = max(0, min(x0, image.width - 1)), max(0, min(y0, image.height - 1))
    x1, y1 = max(x0 + 1, min(x1, image.width)), max(y0 + 1, min(y1, image.height))
    return image.crop((x0, y0, x1, y1))


def make_crop(image, box):
    """
    埋め込みモデル F に入力する切り抜き画像を作る．ボックスにちょうど合わせて切り抜き（周囲の物体は含めない），
    黒で正方形にパディングする．CLIP・SigLIP の前処理は短辺に合わせた中央の切り抜きを行うため，
    パディングしないと細長い物体の一部が失われる（予備的な確認で Top-1 正解率が 0.496 から 0.560 に向上した）．

    Args:
        image (PIL.Image.Image): 入力画像．
        box (array-like): xyxy形式のボックス（長さ4）．

    Returns:
        PIL.Image.Image: 正方形の切り抜き画像．
    """
    crop = crop_box(image, box)
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


def synset_word(synset):
    """
    Synsetを候補語として表す単語（先頭のレンマ名．アンダースコアは空白）に変換する．

    Args:
        synset (nltk.corpus.reader.wordnet.Synset): Synset．

    Returns:
        str: 候補語（例: "African elephant"）．
    """
    return synset.lemma_names()[0].replace("_", " ")


def hyponym_vocab(synset_name):
    """
    Synsetの直接の下位語（hyponyms．インスタンスの下位語は含めない）を候補語彙にする（論文の式(3)）．

    同じ単語になる複数のSynsetは，先に現れたものだけを残す．

    Args:
        synset_name (str | None): Synset名（例: "dog.n.01"）．None の場合は空の語彙を返す．

    Returns:
        list[tuple[str, str]]: (単語, Synset名) のリスト（長さは下位語の数）．
    """
    if synset_name is None:
        return []
    vocab, seen = [], set()
    for s in sorted(wn.synset(synset_name).hyponyms(), key=lambda x: x.name()):
        word = synset_word(s)
        if word not in seen:
            seen.add(word)
            vocab.append((word, s.name()))
    return vocab


def root_vocab(name):
    """
    第1段階（深さ0）の候補語彙を返す．

    Args:
        name (str): "coco80"（COCOの80クラス），"coco80_extra"（COCO 80 + COCO外の22語．論文4.6節），
                    "wordnet_tops"（WordNetの上位語6語．進捗報告の定性評価）のいずれか．

    Returns:
        list[tuple[str, str]]: (単語, Synset名) のリスト．
    """
    table = {"coco80": COCO_TO_SYNSET, "coco80_extra": COCO_TO_SYNSET | EXTRA_TO_SYNSET,
             "wordnet_tops": WORDNET_TOPS}[name]
    return list(table.items())


def caption_words(coco, image_ids):
    """
    画像ごとのキャプションから名詞・動詞のレンマを抽出する（論文4.2節のタスクB．spaCy en_core_web_sm）．

    結果は DATASET_DIR にキャッシュする．

    Args:
        coco (dict): load_coco の戻り値（画像IDの確認のみに用いる）．
        image_ids (list[int]): 画像IDのリスト．

    Returns:
        dict[int, list[str]]: image_id → その画像のキャプションに現れる名詞・動詞のユニークなレンマのリスト．
    """
    path = os.path.join(DATASET_DIR, f"caption_words_{len(image_ids)}_{image_ids[0]}.json")
    if os.path.exists(path):
        return {int(k): v for k, v in load_json(path).items()}
    import spacy
    nlp = spacy.load("en_core_web_sm")
    captions = {}
    for a in load_json(COCO_CAPTIONS_PATH)["annotations"]:
        if a["image_id"] in coco["images"]:
            captions.setdefault(a["image_id"], []).append(a["caption"])
    words = {}
    for image_id in image_ids:
        docs = nlp.pipe(captions.get(image_id, []))
        words[image_id] = sorted({t.lemma_.lower() for d in docs for t in d
                                  if t.pos_ in ("NOUN", "VERB") and t.is_alpha and not t.is_stop})
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(words, f)
    return words


def prepare_open_vocab_images():
    """
    COCO外の語を主要な被写体とする画像（Wikimedia Commons）を OPEN_VOCAB_DIR に用意し，読み込む（論文4.6節）．

    Returns:
        dict[str, PIL.Image.Image]: 画像名（例: "penguin"）→ RGB画像．
    """
    os.makedirs(OPEN_VOCAB_DIR, exist_ok=True)
    for name in OPEN_VOCAB_IMAGES + ["open_vocab_meta"]:
        ext = ".json" if name == "open_vocab_meta" else ".jpg"
        dst = os.path.join(OPEN_VOCAB_DIR, name + ext)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(OPEN_VOCAB_SOURCE_DIR, name + ext), dst)
    return {n: Image.open(os.path.join(OPEN_VOCAB_DIR, n + ".jpg")).convert("RGB") for n in OPEN_VOCAB_IMAGES}
