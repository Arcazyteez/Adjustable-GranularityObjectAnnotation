"""Alpha-CLIP の入力（crop，context1.5，context2，full）の実例を図にする．

ex019 の data.make_alpha_input（実験で用いた関数そのもの）で，COCO val2017 の test の正解ボックスから
RGB の正方形画像（224×224）と αマスクを作り，並べて表示する．新しい実験は行わない．

出力：outputs/summary_new_method/fig_alpha_inputs.png

実行：.venv_pytorch/bin/python programs/summary_new_method/show_alpha_inputs.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
EX019 = ROOT / "programs" / "ex019_coco_owl_alphaclip_wordnet"
SAVE = ROOT / "outputs" / "summary_new_method"
MODES = ["crop", "context1.5", "context2", "full"]
# 人物が持ち物を持つ例と，小さい物体の例（test の正解ボックスから自動で選ぶ）
TARGETS = [("person", "tennis racket"), ("person", "surfboard"), ("cup", None)]


def import_ex019_data():
    """ex019 の data モジュールを読み込む．

    Args:
        なし．

    Returns:
        module: ex019 の data．
    """
    sys.path.insert(0, str(EX019))
    cwd = os.getcwd()
    os.chdir(EX019)
    try:
        import data
    finally:
        os.chdir(cwd)
    return data


def box_area(box):
    """ボックスの面積を返す．

    Args:
        box (np.ndarray): xyxy 形式のボックス，形状 (4,)．

    Returns:
        float: 面積．
    """
    return float((box[2] - box[0]) * (box[3] - box[1]))


def overlaps(a, b):
    """2つのボックスが重なるかを返す．

    Args:
        a (np.ndarray): xyxy 形式のボックス，形状 (4,)．
        b (np.ndarray): xyxy 形式のボックス，形状 (4,)．

    Returns:
        bool: 重なれば True．
    """
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def find_example(data, coco, image_ids, target, partner):
    """target のクラスの正解ボックスを1つ選ぶ．partner があれば，そのクラスのボックスと重なるものに限る．

    Args:
        data (module): ex019 の data．
        coco (dict): COCO の注釈．
        image_ids (list[int]): 探す画像 ID．
        target (str): 対象のクラス名．
        partner (str | None): 重なるべきクラス名．

    Returns:
        tuple[int, np.ndarray]: 画像 ID とボックス（xyxy，形状 (4,)）．
    """
    best = None
    for image_id in image_ids:
        boxes, names = data.gt_instances(coco, image_id)
        others = [b for b, n in zip(boxes, names) if n == partner]
        for box, name in zip(boxes, names):
            if name != target or (partner and not any(overlaps(box, o) for o in others)):
                continue
            area = box_area(box)
            if partner is None and not 1500 < area < 6000:
                continue
            if best is None or (partner and area > best[2]) or (partner is None and area > best[2]):
                best = (image_id, np.asarray(box, float), area)
    return best[0], best[1]


def show_pair(ax_rgb, ax_alpha, rgb, alpha, title):
    """RGB 画像と αマスクを並べて描く．

    Args:
        ax_rgb (matplotlib.axes.Axes): RGB の描画先．
        ax_alpha (matplotlib.axes.Axes): α の描画先．
        rgb (PIL.Image.Image): 224×224 の RGB 画像．
        alpha (np.ndarray): αマスク，形状 (224, 224)．
        title (str): 題名．

    Returns:
        None
    """
    ax_rgb.imshow(rgb)
    ax_rgb.set_title(f"{title}\nRGB", fontsize=10)
    ax_alpha.imshow(alpha, cmap="gray", vmin=0, vmax=1)
    ax_alpha.set_title(f"{title}\nα（白=1）", fontsize=10)


def main():
    """図を作って保存する．

    Args:
        なし．

    Returns:
        None
    """
    from matplotlib import font_manager
    for name in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    data = import_ex019_data()
    coco = data.load_coco()
    test_ids = data.load_splits(coco)["test"]
    SAVE.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TARGETS), 1 + 2 * len(MODES), figsize=(22, 2.9 * len(TARGETS)))
    for row, (target, partner) in enumerate(TARGETS):
        image_id, box = find_example(data, coco, test_ids, target, partner)
        image = data.load_image(coco, image_id)
        axes[row, 0].imshow(image)
        axes[row, 0].add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec="red", lw=2))
        axes[row, 0].set_title(f"元画像（{target}，id {image_id}）", fontsize=10)
        for k, mode in enumerate(MODES):
            rgb, alpha = data.make_alpha_input(image, box, mode)
            show_pair(axes[row, 1 + 2 * k], axes[row, 2 + 2 * k], rgb, alpha, mode)
        print(target, image_id, box.round(1).tolist())
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(SAVE / "fig_alpha_inputs.png", dpi=110)


if __name__ == "__main__":
    main()
