"""
ex001のデータ定義：画像単位の単語推定タスク（論文4.2節）．

- タスクA（閉域評価）: 候補語彙 V_A をCOCO 80クラスとし，画像中に存在するGTカテゴリ集合を正解とする．
- タスクB（広域評価）: 500枚のGTキャプションからspaCyで抽出した名詞・動詞のレンマの和集合を V_B とし，
  各画像のキャプション由来の語集合を正解とする．
"""
import json
import os
import sys

import spacy
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, load_coco, load_image, sample_image_ids  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402

DATA_DIR = os.path.join(DATASET_DIR, "ex001_coco_val500_image_word")


def caption_words(nlp, captions):
    """
    キャプションから名詞・動詞のレンマを抽出する．

    Args:
        nlp (spacy.Language): spaCyのパイプライン（en_core_web_sm）．
        captions (list[str]): 1画像のキャプションのリスト．

    Returns:
        set[str]: 小文字化したレンマの集合（ストップワード・非アルファベット語は除外）．
    """
    words = set()
    for doc in nlp.pipe(captions):
        words |= {t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "VERB") and t.is_alpha and not t.is_stop}
    return words


def build_task_data(n_images=500, seed=42):
    """
    タスクA・Bの候補語彙と画像ごとの正解語集合を作成し，datasets/以下にJSONで保存する．

    Args:
        n_images (int): サンプリングする画像数．
        seed (int): サンプリングの乱数シード．

    Returns:
        dict: {"image_ids": list[int], "A": {"vocab": list[str], "gt": dict}, "B": {...}}．
              gtは画像ID（str）→ 正解語のリスト．
    """
    path = os.path.join(DATA_DIR, f"tasks_n{n_images}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    coco_inst, coco_caps = load_coco("instances"), load_coco("captions")
    image_ids = sample_image_ids(coco_inst, n_images, seed)
    nlp = spacy.load("en_core_web_sm")

    gt_a, gt_b = {}, {}
    for image_id in tqdm(image_ids, desc="build tasks", leave=False):
        anns = coco_inst.loadAnns(coco_inst.getAnnIds(imgIds=image_id))
        gt_a[str(image_id)] = sorted({coco_inst.cats[a["category_id"]]["name"] for a in anns})
        captions = [a["caption"] for a in coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=image_id))]
        gt_b[str(image_id)] = sorted(caption_words(nlp, captions))

    data = {
        "image_ids": image_ids,
        "A": {"vocab": category_names(coco_inst), "gt": gt_a},
        "B": {"vocab": sorted(set().union(*map(set, gt_b.values()))), "gt": gt_b},
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def load_images(image_ids):
    """
    画像IDのリストに対応する画像を読み込む．

    Args:
        image_ids (list[int]): 画像IDのリスト（長さN）．

    Returns:
        list[PIL.Image.Image]: RGB画像のリスト（長さN）．
    """
    coco = load_coco("instances")
    return [load_image(coco, i) for i in tqdm(image_ids, desc="load images", leave=False)]
