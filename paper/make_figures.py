"""
論文用の図を outputs/ の実験結果から作成し，paper/figures/ に保存する．

実行: .venv_pytorch/bin/python paper/make_figures.py
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER_DIR)
sys.path.insert(0, os.path.join(ROOT, "programs"))
from common.coco import crop_box, load_coco, load_image, pad_to_square  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
EX001 = os.path.join(OUT, "ex001_coco_val500_image_word")
EX003 = os.path.join(OUT, "ex003_coco_val500_pipeline")
EX004 = os.path.join(OUT, "ex004_open_vocab_qualitative", "CLIP-ViT-B32")
EX005 = os.path.join(OUT, "ex005_coco_val500_yolo_comparison")
EX006 = os.path.join(OUT, "ex006_coco_val500_yolo_clip_hybrid")
EX007 = os.path.join(OUT, "ex007_coco_val_hierarchy_eval")
VLMS = ["ResNet50-IN1k", "CLIP-RN50", "CLIP-ViT-B32"]
VLM_LABELS = ["ResNet50-IN1k", "CLIP-RN50", "CLIP-ViT-B/32"]
COLORS = ["#8c8c8c", "#4C72B0", "#DD8452"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})


def savefig(fig, name):
    """
    図をPDFとPNGで保存する．

    Args:
        fig (matplotlib.figure.Figure): 保存する図．
        name (str): ファイル名（拡張子なし）．

    Returns:
        None
    """
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_vlm_comparison():
    """
    ex001の4指標（タスクA・BのP@1，R@5）を手法間で比較する棒グラフ（誤差棒はブートストラップ95%CI）を作成する．

    Returns:
        None
    """
    panels = [("A", "P@1", "(a) Task A, P@1"), ("A", "R@5", "(b) Task A, R@5"),
              ("B", "P@1", "(c) Task B, P@1"), ("B", "R@5", "(d) Task B, R@5")]
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.4))
    for ax, (task, metric, title) in zip(axes.flat, panels):
        ms = [json.load(open(os.path.join(EX001, m, task, "42", "metrics.json")))[metric] for m in VLMS]
        means = np.array([m["mean"] for m in ms])
        err = np.array([[m["mean"] - m["ci_low"], m["ci_high"] - m["mean"]] for m in ms]).T
        ax.bar(range(3), means, yerr=err, capsize=2, color=COLORS)
        ax.set_xticks(range(3), ["ResNet50\nIN1k", "CLIP\nRN50", "CLIP\nViT-B/32"], fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    fig.tight_layout()
    savefig(fig, "fig_vlm_comparison")


def fig_pipeline_accuracy():
    """
    ex003のTop-1/Top-5正解率をDETRの閾値に対して描画する（VLMとクロップ前処理ごと）．

    Returns:
        None
    """
    thresholds = [0.5, 0.7, 0.9, 0.97]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for ax, metric, title in zip(axes, ["top1_accuracy", "top5_accuracy"], ["(a) Top-1", "(b) Top-5"]):
        for vlm, label, color in zip(VLMS, VLM_LABELS, COLORS):
            for crop, style in [("center", "--"), ("pad", "-")]:
                values = [json.load(open(os.path.join(EX003, vlm, f"detr_threshold{t}_crop_{crop}", "42", "metrics.json")))[metric]
                          for t in thresholds]
                ax.plot(range(4), values, style, marker="o", ms=3, color=color, label=f"{label} ({crop})")
        ax.set_xticks(range(4), [str(t) for t in thresholds])
        ax.set_xlabel("DETR score threshold")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), fontsize=5.5, frameon=False)
    fig.tight_layout()
    savefig(fig, "fig_pipeline_accuracy")


def fig_depth_confidence(condition="detr_threshold0.97_crop_pad"):
    """
    WordNet階層の深さごとの平均確信度（棒，左軸）とインスタンス数（折れ線，右軸，対数）を描画する．

    Args:
        condition (str): ex003の条件名．

    Returns:
        None
    """
    stats = pd.read_csv(os.path.join(EX003, "CLIP-ViT-B32", condition, "42", "depth_stats.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(stats["depth"], stats["mean_conf"], yerr=stats["std_conf"], capsize=2, color="#4C72B0", alpha=0.85,
           label="mean confidence")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("WordNet depth $d$ (0: COCO 80 classes)")
    ax.set_ylabel("mean confidence")
    ax2 = ax.twinx()
    ax2.plot(stats["depth"], stats["n_instances"], "o-", color="#C44E52", ms=3, label="#instances")
    ax2.set_yscale("log")
    ax2.set_ylabel("#instances")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig_depth_confidence")


def fig_crop_preprocess(examples=((426253, "bottle"), (137727, "train"))):
    """
    クロップの前処理（中央切り出し，正方形パディング）でVLMへの入力がどう変わるかを示す図を作成する．

    Args:
        examples (tuple): (画像ID, GTクラス名) の組．ex003の閾値0.97の条件から選ぶ．

    Returns:
        None
    """
    coco = load_coco("instances")
    records = {crop: json.load(open(os.path.join(EX003, "CLIP-ViT-B32", f"detr_threshold0.97_crop_{crop}", "42", "instances.json")))
               for crop in ("center", "pad")}
    fig, axes = plt.subplots(len(examples), 3, figsize=(3.4, 1.25 * len(examples)))
    for row, (image_id, gt_name) in zip(axes, examples):
        idx = next(i for i, r in enumerate(records["center"]) if r["image_id"] == image_id and r["gt_name"] == gt_name)
        crop = crop_box(load_image(coco, image_id), records["center"][idx]["box"])
        w, h = crop.size
        s = min(w, h)
        center = crop.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        panels = [(crop, f"crop (GT: {gt_name})"),
                  (center, f"center: {records['center'][idx]['top5'][0]} ({records['center'][idx]['top5_conf'][0]:.2f})"),
                  (pad_to_square(crop), f"pad: {records['pad'][idx]['top5'][0]} ({records['pad'][idx]['top5_conf'][0]:.2f})")]
        for ax, (im, title) in zip(row, panels):
            ax.imshow(im)
            ax.set_title(title, fontsize=6)
            ax.axis("off")
    fig.tight_layout()
    savefig(fig, "fig_crop_preprocess")


def draw_regions(ax, image, regions, title, min_det_score=0.0):
    """
    画像にボックスと各深さの推定単語・確信度を描画する．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        image (PIL.Image.Image): 画像．
        regions (list[dict]): ex004の annotations.json の領域のリスト．
        title (str): タイトル．
        min_det_score (float): 描画する領域の検出スコアの下限．

    Returns:
        None
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=7)
    ax.axis("off")
    palette = plt.cm.tab10.colors
    for i, region in enumerate(r for r in regions if r["det_score"] >= min_det_score):
        x0, y0, x1, y1 = region["box"]
        color = palette[i % len(palette)]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.2))
        text = "\n".join(("" if s["depth"] == 0 else "> ") + f"{s['word']} ({s['conf']:.2f})" for s in region["chain"])
        ax.text(x0 + 2, y0 + 2, text, fontsize=4.5, color="white", va="top", bbox=dict(facecolor=color, alpha=0.8, pad=0.8, lw=0))


def fig_qualitative():
    """
    COCO外の語を含む実画像（ex004 open_vocab）とCOCO画像（ex004 coco）の階層的アノテーション結果の図を作成する．

    Returns:
        None
    """
    data_dir = os.path.join(ROOT, "datasets", "ex004_open_vocab_qualitative")
    ann = json.load(open(os.path.join(EX004, "open_vocab_detr_threshold0.5_crop_pad", "42", "annotations.json")))
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    for ax, name in zip(axes.flat, ["penguin", "helicopter", "camel", "kangaroo", "lighthouse_1", "lighthouse_2"]):
        image = Image.open(os.path.join(data_dir, f"{name}.jpg")).convert("RGB")
        draw_regions(ax, image, ann[name]["regions"], f"{name} ({len(ann[name]['regions'])} boxes)")
    fig.tight_layout()
    savefig(fig, "fig_open_vocab")

    coco = load_coco("instances")
    ann = json.load(open(os.path.join(EX004, "coco_detr_threshold0.5_crop_pad", "42", "annotations.json")))
    names = ["coco_elephant_556765", "coco_giraffe_507042", "coco_bear_460927", "coco_cat_219578"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    for ax, name in zip(axes, names):
        draw_regions(ax, load_image(coco, ann[name]["meta"]["image_id"]), ann[name]["regions"],
                     f"COCO {ann[name]['meta']['image_id']}")
    fig.tight_layout()
    savefig(fig, "fig_coco_qualitative")


def fig_yolo_open_vocab(condition="open_vocab_conf0.25"):
    """
    COCO外の語を含む実画像に対するYOLO11x（COCOの80クラス）の検出結果の図を作成する（order_003）．

    Args:
        condition (str): ex005の条件名．

    Returns:
        None
    """
    data_dir = os.path.join(ROOT, "datasets", "ex004_open_vocab_qualitative")
    ann = json.load(open(os.path.join(EX005, "yolo11x", condition, "42", "annotations.json")))
    names = ["penguin", "helicopter", "camel", "kangaroo", "lighthouse_1", "lighthouse_2"]
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.6))
    for ax, name in zip(axes, names):
        ax.imshow(Image.open(os.path.join(data_dir, f"{name}.jpg")).convert("RGB"))
        ax.set_title(f"{name} ({len(ann[name]['boxes'])} boxes)", fontsize=6)
        ax.axis("off")
        # 確信度の高い検出のラベルが前面に来るよう，確信度の低い順に描画する
        detections = list(enumerate(zip(ann[name]["boxes"], ann[name]["top5"], ann[name]["top5_conf"])))
        for i, (box, top5, conf5) in reversed(detections):
            color = plt.cm.tab10.colors[i % 10]
            ax.add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color, linewidth=1))
            ax.text(box[0] + 2, box[1] + 2, f"{top5[0]} ({conf5[0]:.2f})", fontsize=4, color="white", va="top",
                    bbox=dict(facecolor=color, alpha=0.8, pad=0.6, lw=0))
    fig.tight_layout()
    savefig(fig, "fig_yolo_open_vocab")


def fig_yolo_comparison():
    """
    YOLOと提案手法の，GTの正しい命名率（横軸）と対応インスタンスのTop-1正解率（縦軸）の関係を描画する（order_003）．

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    confs = [0.25, 0.5, 0.7]
    styles = [("yolo11n", "YOLO11n", "o", "#9ecae1"), ("yolo11m", "YOLO11m", "v", "#4292c6"),
              ("yolo11x", "YOLO11x", "s", "#08519c"), ("yolov8x", "YOLOv8x", "^", "#6a51a3"),
              ("yolo11x+CLIP-ViT-B32", "YOLO11x box + CLIP", "D", "#DD8452")]
    for method, label, marker, color in styles:
        ms = [json.load(open(os.path.join(EX005, method, f"coco500_conf{c}", "42", "metrics.json"))) for c in confs]
        ax.plot([m["annotation_recall"] for m in ms], [m["top1_accuracy"] for m in ms], marker=marker, ms=3.5,
                color=color, label=label)
    for theta in [0.5, 0.7, 0.9, 0.97]:
        m = json.load(open(os.path.join(EX003, "CLIP-ViT-B32", f"detr_threshold{theta}_crop_pad", "42", "metrics.json")))
        ax.plot(m["top1_accuracy"] * m["n_instances"] / m["n_gt"], m["top1_accuracy"], "k*", ms=6,
                label="proposed (DETR + CLIP)" if theta == 0.5 else None)
    ax.set_xlabel("correctly annotated GT / all GT")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig_yolo_comparison")


def fig_hybrid_qualitative(mode="hybrid_switch_c0.5", conf=0.25):
    """
    YOLOとCLIPのハイブリッド構成（ex006）による階層的アノテーションの図を作成する（order_006）．
    上段はCOCO外の語を含む実画像，下段は図7と同じCOCO画像である．深さ0の決定元を「YOLO:」「CLIP:」で示す．

    Args:
        mode (str): ex006の構成名．
        conf (float): YOLOの検出スコアの閾値．

    Returns:
        None
    """
    data_dir = os.path.join(ROOT, "datasets", "ex004_open_vocab_qualitative")
    open_ann = json.load(open(os.path.join(EX006, mode, f"open_vocab_conf{conf}", "42", "annotations.json")))
    coco_ann = json.load(open(os.path.join(EX006, mode, f"coco_qualitative_conf{conf}", "42", "annotations.json")))
    coco = load_coco("instances")
    # ペンギンの画像はボックスが14件あり単語が重なるため，検出スコアの上位3件のみ単語を表示する
    panels = [(Image.open(os.path.join(data_dir, f"{n}.jpg")).convert("RGB"), open_ann[n]["regions"], n, 3 if n == "penguin" else None)
              for n in ["penguin", "helicopter", "camel", "kangaroo"]]
    panels += [(load_image(coco, coco_ann[n]["meta"]["image_id"]), coco_ann[n]["regions"], f"COCO {coco_ann[n]['meta']['image_id']}", None)
               for n in ["coco_elephant_556765", "coco_giraffe_507042", "coco_bear_460927", "coco_cat_219578"]]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6))
    for ax, (image, regions, title, max_labels) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
        labeled = sorted(range(len(regions)), key=lambda j: -regions[j]["det_score"])[:max_labels]
        for i, region in enumerate(regions):
            x0, y0, x1, y1 = region["box"]
            color = plt.cm.tab10.colors[i % 10]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.2))
            if i not in labeled:
                continue
            head = "CLIP: " if region["root_source"] == "clip" else "YOLO: "
            text = "\n".join((head if s["depth"] == 0 else "> ") + f"{s['word']} ({s['conf']:.2f})" for s in region["chain"])
            ax.text(x0 + 2, y0 + 2, text, fontsize=4.5, color="white", va="top", bbox=dict(facecolor=color, alpha=0.8, pad=0.8, lw=0))
    fig.tight_layout()
    savefig(fig, "fig_hybrid_qualitative")


def fig_stopping(split="lvis_test", conditions=(("proposed", 0.5, "Proposed (θ=0.5)"), ("hybrid", 0.25, "Hybrid (YOLO 0.25)"))):
    """
    累積確信度による細分化の停止規則について，LVISのテスト用分割における階層的適合率と再現率の関係を描画する（ex007）．

    Args:
        split (str): 分割名．
        conditions (tuple): (手法, 閾値, 凡例) の組．

    Returns:
        None
    """
    stopping = pd.read_csv(os.path.join(EX007, "table_stopping.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for (method, threshold, label), color in zip(conditions, COLORS[1:]):
        curve = pd.read_csv(os.path.join(EX007, method, f"{split}_th{threshold}", "42", "stopping_curve.csv"))
        tau, depth = curve[curve["rule"] == "tau"], curve[curve["rule"] == "depth"]
        ax.plot(tau["lvis_hR"], tau["lvis_hP"], "-o", color=color, ms=3, label=f"{label}: τ")
        ax.plot(depth["lvis_hR"], depth["lvis_hP"], "s", color=color, mfc="none", ms=5, label=f"{label}: depth limit")
        selected = stopping[(stopping["method"] == method) & (stopping["threshold"] == threshold) &
                            (stopping["split"] == split)]["selected_tau"].iloc[0]
        point = tau[np.isclose(tau["value"], selected)].iloc[0]
        ax.plot(point["lvis_hR"], point["lvis_hP"], "*", color="red", ms=10)
        ax.annotate(f"τ={selected:g}", (point["lvis_hR"], point["lvis_hP"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("Hierarchical recall hR")
    ax.set_ylabel("Hierarchical precision hP")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, "fig_stopping")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_vlm_comparison()
    fig_pipeline_accuracy()
    fig_depth_confidence()
    fig_crop_preprocess()
    fig_qualitative()
    fig_yolo_open_vocab()
    fig_yolo_comparison()
    fig_hybrid_qualitative()
    fig_stopping()
    print(sorted(os.listdir(FIG_DIR)))
