"""
ex004の実行スクリプト：COCO外の語を含む実画像へのアノテーションと，COCO画像に対する階層的アノテーションの
定性評価（論文4.6節，4.7節，図4，図5）．

学習は行わない．各画像でDETRを1回だけ実行し，検出された全領域のクロップに対して
第1段階の候補語彙でゼロショット分類した後，WordNetの下位語をたどる粒度可変アノテーションを行う．

出力: outputs/ex004_open_vocab_qualitative/CLIP-ViT-B32/{画像セット}_detr_threshold{閾値}_crop_{前処理}/{seed}/
    config.json, annotations.json, {画像名}.png, grid.png
"""
import itertools
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import make_crop  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET, EXTRA_TO_SYNSET, hierarchical_annotation  # noqa: E402
from data import load_coco_qualitative_images, load_open_vocab_images  # noqa: E402
from model import load_model  # noqa: E402

# ===== 実験条件（order_001） =====
EXPERIMENT = "ex004_open_vocab_qualitative"
VLM_NAME = "CLIP-ViT-B32"
SEEDS = [42]
IMAGE_SETS = {"open_vocab": load_open_vocab_images, "coco": load_coco_qualitative_images}
DETR_THRESHOLDS = [0.5]  # 論文4.6節の「検出スコア0.5以上」
CROP_MODES = ["pad"]
WORD_TO_SYNSET = COCO_TO_SYNSET | EXTRA_TO_SYNSET


def annotate_image(detector, vlm, image, vocab, crop_mode):
    """
    1枚の画像に対し，検出 → 第1段階の分類 → WordNet階層のアノテーションを行う．

    Args:
        detector (DETRDetector): 検出器．
        vlm (CLIPModel): VLM．
        image (PIL.Image.Image): 入力画像．
        vocab (list[str]): 第1段階の候補語彙．
        crop_mode (str): クロップの前処理．

    Returns:
        list[dict]: 領域ごとの {"box", "det_score", "chain"}（chainは深さごとの推定単語と確信度）．
    """
    boxes, scores = detector.detect(image)
    if len(boxes) == 0:
        return []
    crops = [make_crop(image, box, crop_mode) for box in boxes]
    probs = vlm.score(crops, vocab)
    regions = []
    for box, det_score, crop, prob in zip(boxes, scores, crops, probs):
        word = vocab[int(prob.argmax())]
        chain = hierarchical_annotation(vlm, crop, word, WORD_TO_SYNSET[word], float(prob.max()))
        regions.append({"box": box.tolist(), "det_score": float(det_score), "chain": chain})
    return regions


def draw_annotations(ax, image, regions, title):
    """
    画像に検出ボックスと各段階の推定単語・確信度を描画する．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        image (PIL.Image.Image): 入力画像．
        regions (list[dict]): annotate_image の戻り値．
        title (str): タイトル．

    Returns:
        None
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    colors = plt.cm.tab10.colors
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region["box"]
        color = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
        text = "\n".join(("" if s["depth"] == 0 else "> ") + f"{s['word']} ({s['conf']:.2f})" for s in region["chain"])
        ax.text(x0, y0, text, fontsize=6, color="white", va="top", bbox=dict(facecolor=color, alpha=0.75, pad=1))


def save_figures(target_dir, samples, results):
    """
    画像ごとの図と一覧図を保存する．

    Args:
        target_dir (str): 保存先ディレクトリ．
        samples (list[dict]): 画像のリスト．
        results (dict): 画像名 → 領域のリスト．

    Returns:
        None
    """
    for sample in samples:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_annotations(ax, sample["image"], results[sample["name"]], f"{sample['name']} ({len(results[sample['name']])} boxes)")
        fig.savefig(os.path.join(target_dir, f"{sample['name']}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    n_cols = 3
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, sample in zip(axes.flat, samples):
        draw_annotations(ax, sample["image"], results[sample["name"]], sample["name"])
    fig.tight_layout()
    fig.savefig(os.path.join(target_dir, "grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """全条件の定性評価を実行する（完了済みの条件はスキップ）．"""
    for seed, set_name, threshold, crop_mode in itertools.product(SEEDS, IMAGE_SETS, DETR_THRESHOLDS, CROP_MODES):
        target_dir = os.path.join(OUTPUT_DIR, EXPERIMENT, VLM_NAME,
                                  f"{set_name}_detr_threshold{threshold}_crop_{crop_mode}", str(seed))
        if os.path.exists(os.path.join(target_dir, "annotations.json")):
            continue
        samples, vocab = IMAGE_SETS[set_name]()
        detector, vlm = load_model(VLM_NAME, threshold, seed)
        results = {s["name"]: annotate_image(detector, vlm, s["image"], vocab, crop_mode) for s in samples}
        os.makedirs(target_dir, exist_ok=True)
        save_figures(target_dir, samples, results)
        with open(os.path.join(target_dir, "annotations.json"), "w") as f:
            json.dump({s["name"]: {"meta": s["meta"], "regions": results[s["name"]]} for s in samples}, f, indent=2,
                      ensure_ascii=False)
        with open(os.path.join(target_dir, "config.json"), "w") as f:
            json.dump({"vlm": VLM_NAME, "detr_threshold": threshold, "crop_mode": crop_mode, "seed": seed,
                       "image_set": set_name, "vocab_size": len(vocab), "vocab": vocab}, f, indent=4)
        for name, regions in results.items():
            print(name, [" > ".join(f"{s['word']}({s['conf']:.2f})" for s in r["chain"]) for r in regions])
        del detector, vlm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
