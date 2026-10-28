"""COCO 2017 Validationデータセットの読み込みとサンプリング．"""
import contextlib
import io
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from .paths import COCO_CAPTIONS_PATH, COCO_IMAGE_DIR, COCO_INSTANCES_PATH


def load_coco(kind="instances"):
    """
    COCOのアノテーションを読み込む．

    Args:
        kind (str): "instances"（物体検出）または "captions"（キャプション）．

    Returns:
        pycocotools.coco.COCO: アノテーションを保持するCOCOオブジェクト．
    """
    path = COCO_INSTANCES_PATH if kind == "instances" else COCO_CAPTIONS_PATH
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(path)


def sample_image_ids(coco, n_images=500, seed=42):
    """
    COCO画像IDを再現可能な乱数でサンプリングする．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        n_images (int): サンプリングする画像数．
        seed (int): 乱数シード．

    Returns:
        list[int]: サンプリングされた画像IDのリスト（長さ n_images）．
    """
    image_ids = sorted(coco.getImgIds())
    return random.Random(seed).sample(image_ids, n_images)


def load_image(coco, image_id):
    """
    画像IDに対応するRGB画像を読み込む．

    Args:
        coco (COCO): COCOオブジェクト．
        image_id (int): 画像ID．

    Returns:
        PIL.Image.Image: RGB画像（H×W×3）．
    """
    file_name = coco.loadImgs(image_id)[0]["file_name"]
    return Image.open(f"{COCO_IMAGE_DIR}/{file_name}").convert("RGB")


def category_names(coco):
    """
    COCOの80オブジェクトクラス名をカテゴリID順に返す．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．

    Returns:
        list[str]: クラス名のリスト（長さ80）．
    """
    return [coco.cats[c]["name"] for c in sorted(coco.getCatIds())]


def gt_instances(coco, image_id):
    """
    画像中のGTインスタンス（iscrowd=0）を取得する．

    Args:
        coco (COCO): instancesのCOCOオブジェクト．
        image_id (int): 画像ID．

    Returns:
        tuple[np.ndarray, list[str]]: xyxy形式のボックス（K×4）とクラス名のリスト（長さK）．
    """
    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=False))
    boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                      for a in anns], dtype=np.float32).reshape(-1, 4)
    names = [coco.cats[a["category_id"]]["name"] for a in anns]
    return boxes, names


def crop_box(image, box):
    """
    バウンディングボックスにちょうど合わせて画像を切り出す（余白なし）．

    Args:
        image (PIL.Image.Image): 元画像．
        box (array-like): xyxy形式のボックス（長さ4）．

    Returns:
        PIL.Image.Image: クロップ画像．
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return image.crop((x0, y0, x1, y1))


def pad_to_square(image, fill=(0, 0, 0)):
    """
    クロップを単色で正方形にパディングする（VLMの前処理の中央切り出しで物体の一部が欠けるのを防ぐ）．
    ボックス外の画像内容は含めないため，ボックスにちょうど合わせたクロップという条件は保たれる．

    Args:
        image (PIL.Image.Image): クロップ画像（h×w）．
        fill (tuple[int, int, int]): パディング色．

    Returns:
        PIL.Image.Image: 正方形画像（max(h, w)×max(h, w)）．
    """
    w, h = image.size
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - w) // 2, (size - h) // 2))
    return canvas


def make_crop(image, box, crop_mode="pad"):
    """
    前処理方式に応じたクロップを作成する．

    Args:
        image (PIL.Image.Image): 元画像．
        box (array-like): xyxy形式のボックス（長さ4）．
        crop_mode (str): "center"（クロップをそのままVLMへ入力し，VLM側で中央切り出し）または
            "pad"（黒で正方形にパディング）．

    Returns:
        PIL.Image.Image: VLMへ入力するクロップ画像．
    """
    crop = crop_box(image, box)
    return pad_to_square(crop) if crop_mode == "pad" else crop
