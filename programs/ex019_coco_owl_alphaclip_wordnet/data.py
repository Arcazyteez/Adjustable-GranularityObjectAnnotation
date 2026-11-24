"""
ex019：データの読み込みと，埋め込みモデルへの入力画像の作成（order_029）．

- 画像と正解：ex017 と同じ COCO val2017 の dev 500枚（ボックス数と θ の選択）と test 500枚（報告）．MSCOCO 以外は用いない
- 古い提案手法（CLIP）の入力：ボックスにちょうど合わせた切り抜き画像を黒で正方形にパディングしたもの（ex017 の make_crop）
- 新しい提案手法（Alpha-CLIP）の入力：正方形の窓のRGB画像と，注目する領域を1とするαマスク（ボックスの内側を1とする矩形）
  - "crop"：ボックスにちょうど合わせた切り抜き画像を黒で正方形にパディングしたもの（αは切り抜き画像の部分が1）
  - "context{s}"：ボックスの中心を中心とし，ボックスの長辺の s 倍の正方形（画像外は黒．例："context1.5"，"context2"）
  - "full"：画像全体を黒で正方形にパディングしたもの（公式の参照表現理解の評価と同じく画像全体とマスクを与える）
  どの入力を用いるかは dev の正解ボックスでの Top-1 正解率で選ぶ（train.py の E1）．
COCO の読み込み・分割・正解・WordNet の語彙は ex017 の data.py をそのまま用いる（load_ex017 で読み込む）．
"""
import functools
import importlib
import os
import sys

import numpy as np
from PIL import Image

EX_NAME = "ex019_coco_owl_alphaclip_wordnet"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", EX_NAME)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", EX_NAME)
EX017_DIR = os.path.join(PROJECT_ROOT, "programs", "ex017_coco_granular_annotation")
EX017_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "ex017_coco_granular_annotation")
ALPHA_INPUTS = ["crop", "context1.5", "context2", "full"]   # Alpha-CLIP の入力の候補


@functools.lru_cache(maxsize=None)
def load_ex017():
    """
    ex017 の data，model，train モジュールを読み込む．
    ex019 自身の data，model，train と名前が衝突するため，一時的に sys.modules から外して読み込み，後で戻す．

    Returns:
        tuple[module, module, module]: ex017 の data，model，train．
    """
    names = ("data", "model", "train")
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, EX017_DIR)
    try:
        modules = tuple(importlib.import_module(n) for n in names)
    finally:
        sys.path.remove(EX017_DIR)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return modules


def load_coco():
    """
    COCO 2017 val の注釈を読み込む（ex017 の load_coco）．

    Returns:
        dict: "images"，"categories"，"anns"，"class_names"．
    """
    return load_ex017()[0].load_coco()


def load_splits(coco):
    """
    ex017 と同じ dev/test の分割（seed 42，各500枚）を返す．

    Args:
        coco (dict): load_coco の戻り値．

    Returns:
        dict[str, list[int]]: "dev"，"test" → 画像IDのリスト．
    """
    return load_ex017()[0].split_image_ids(coco)


def load_image(coco, image_id):
    """
    COCOの画像をRGBで読み込む．

    Args:
        coco (dict): load_coco の戻り値．
        image_id (int): 画像ID．

    Returns:
        PIL.Image.Image: RGB画像．
    """
    return load_ex017()[0].load_image(coco, image_id)


def gt_instances(coco, image_id):
    """
    画像の物体の正解（iscrowd=0）を返す．

    Args:
        coco (dict): load_coco の戻り値．
        image_id (int): 画像ID．

    Returns:
        tuple[np.ndarray, list[str]]: ボックス（G×4，xyxy）とクラス名（長さG）．
    """
    return load_ex017()[0].gt_instances(coco, image_id)


def make_crop(image, box):
    """
    CLIP に入力する切り抜き画像（黒で正方形にパディング）を作る（ex017 の make_crop）．

    Args:
        image (PIL.Image.Image): 入力画像．
        box (array-like): xyxy形式のボックス（長さ4）．

    Returns:
        PIL.Image.Image: 正方形の切り抜き画像．
    """
    return load_ex017()[0].make_crop(image, box)


def coco80_vocab():
    """
    深さ0の語彙（COCOの80クラスと対応するSynset）を返す．

    Returns:
        list[tuple[str, str]]: (単語, Synset名) のリスト（長さ80）．
    """
    return load_ex017()[0].root_vocab("coco80")


def square_window(image, box, mode):
    """
    Alpha-CLIP に入力する正方形の窓（画像の座標系）を決める．

    Args:
        image (PIL.Image.Image): 入力画像．
        box (array-like): xyxy形式のボックス（長さ4）．
        mode (str): "full"（画像全体）または "context{s}"（ボックスの長辺の s 倍の周囲）．

    Returns:
        tuple[float, float, float]: 窓の左上の x，y と一辺の長さ．
    """
    if mode == "full":
        return 0.0, 0.0, float(max(image.size))
    x0, y0, x1, y1 = (float(v) for v in box)
    side = max(x1 - x0, y1 - y0, 1.0) * float(mode.removeprefix("context"))
    return (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side


def make_alpha_input(image, box, mode, size=224):
    """
    Alpha-CLIP の入力（RGBの正方形画像とαマスク）を作る．窓の外側（画像外）は黒，αはボックスの内側を1とする．

    Args:
        image (PIL.Image.Image): 入力画像．
        box (array-like): xyxy形式のボックス（長さ4）．
        mode (str): ALPHA_INPUTS のいずれか．
        size (int): 出力の一辺のピクセル数．

    Returns:
        tuple[PIL.Image.Image, np.ndarray]: RGB画像（size×size）と αマスク（size×size，float32，0または1）．
    """
    if mode == "crop":
        crop = load_ex017()[0].crop_box(image, box)
        side = max(crop.size)
        canvas = Image.new("RGB", (side, side))
        offset = ((side - crop.width) // 2, (side - crop.height) // 2)
        canvas.paste(crop, offset)
        alpha = np.zeros((side, side), np.float32)
        alpha[offset[1]:offset[1] + crop.height, offset[0]:offset[0] + crop.width] = 1.0
        alpha = np.asarray(Image.fromarray(alpha).resize((size, size), Image.Resampling.NEAREST), np.float32)
        return canvas.resize((size, size), Image.Resampling.BICUBIC), alpha
    left, top, side = square_window(image, box, mode)
    scale = size / side
    rgb = image.transform((size, size), Image.Transform.AFFINE, (1 / scale, 0, left, 0, 1 / scale, top),
                          resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    x0, y0, x1, y1 = ((float(v) - o) * scale for v, o in zip(box, (left, top, left, top)))
    alpha = np.zeros((size, size), np.float32)
    c0, r0 = int(np.clip(np.floor(x0), 0, size - 1)), int(np.clip(np.floor(y0), 0, size - 1))
    c1, r1 = int(np.clip(np.ceil(x1), c0 + 1, size)), int(np.clip(np.ceil(y1), r0 + 1, size))
    alpha[r0:r1, c0:c1] = 1.0
    return rgb, alpha
