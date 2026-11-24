"""
論文（paper_2026B）の図を outputs/ex012〜ex015 の結果から作成し，paper_2026B/figures/ に保存する．

実行: .venv_pytorch/bin/python paper_2026B/make_figures.py
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER_DIR)
sys.path.insert(0, os.path.join(ROOT, "programs"))
from common.coco import load_coco, load_image  # noqa: E402

EX012 = os.path.join(ROOT, "outputs", "ex012_wordnet_granular_annotation")
EX013 = os.path.join(ROOT, "outputs", "ex013_open_vocabulary_granular")
EX015 = os.path.join(ROOT, "outputs", "ex015_lvis_open_vocabulary_annotation")
EX015A = os.path.join(ROOT, "outputs", "ex015a_annotation_score")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
METHODS = [("proposed_yolo_CLIP", "YOLO11x + CLIP"), ("proposed_yolo_SigLIP", "YOLO11x + SigLIP"),
           ("proposed_detr_CLIP", "DETR + CLIP"), ("proposed_detr_SigLIP", "DETR + SigLIP")]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6})


def savefig(fig, name):
    """
    図をPDFとPNGで保存する．

    Args:
        fig (matplotlib.figure.Figure): 図．
        name (str): ファイル名（拡張子なし）．

    Returns:
        None
    """
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_theta():
    """
    検証用の画像における，θ によって残った予測の割合（横軸）と，(a) ラベルの一致率，(b) F値（縦軸）の関係．
    類似度の範囲は F によって異なるため，横軸は θ ではなく残った予測の割合とする．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
    for (method, label), color in zip(METHODS, COLORS):
        curve = pd.read_csv(os.path.join(EX012, method, "dev", "42", "theta_curve.csv"))
        kept = curve["n_pred"] / curve["n_pred"].iloc[0]
        axes[0].plot(kept, curve["label_accuracy"], "-o", ms=1.8, lw=1, color=color, label=label)
        axes[1].plot(kept, curve["f1"], "-o", ms=1.8, lw=1, color=color, label=label)
    for ax, title in zip(axes, ["(a) Label accuracy", "(b) F1"]):
        ax.set_xlabel("Fraction of predictions kept")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    savefig(fig, "fig_theta")


def fig_qualitative(image_ids=("22192", "256941", "229997", "210299"),
                    rows=(("qualitative_yolo_CLIP", "YOLO11x + CLIP"), ("qualitative_yolo_SigLIP", "YOLO11x + SigLIP"))):
    """
    初期の語彙をWordNetの上位語とした定性評価の結果（上段：CLIP，下段：SigLIP）．
    各画像の幅を縦横比に合わせて配置し，ラベルは画像の範囲内で切り取る．

    Args:
        image_ids (tuple[str]): 表示する画像ID．
        rows (tuple): (結果のディレクトリ名, 行の見出し) の組．

    Returns:
        None
    """
    coco = load_coco("instances")
    images = {i: load_image(coco, int(i)) for i in image_ids}
    ratios = [images[i].size[0] / images[i].size[1] for i in image_ids]
    width, gap, left = 7.0, 0.04, 0.22
    height = (width - left - gap * (len(image_ids) - 1)) / sum(ratios)
    fig = plt.figure(figsize=(width, height * len(rows) + gap * (len(rows) - 1)))
    total_h = height * len(rows) + gap * (len(rows) - 1)
    for r, (name, title) in enumerate(rows):
        with open(os.path.join(EX012, name, "test", "42", "annotations.json")) as f:
            results = json.load(f)["results"]
        x = left
        y = total_h - (r + 1) * height - r * gap
        for image_id, ratio in zip(image_ids, ratios):
            ax = fig.add_axes([x / width, y / total_h, ratio * height / width, height / total_h])
            x += ratio * height + gap
            ax.imshow(images[image_id])
            ax.set_xticks([])
            ax.set_yticks([])
            for k, item in enumerate(results[image_id][:4]):
                x0, y0, x1, y1 = item["box"]
                color = COLORS[k % len(COLORS)]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1.0))
                text = " > ".join(w["word"] for w in item["chain"]) or "(rejected)"
                ax.text(x0, y0, text, fontsize=4.6, color="white", va="top", clip_on=True,
                        bbox=dict(facecolor=color, alpha=0.8, pad=0.6, lw=0))
        fig.text(0.02, (y + height / 2) / total_h, title, rotation=90, va="center", ha="center")
    savefig(fig, "fig_qualitative")


def fig_open(configs=(("ex012", "ex012"), ("A_root", "+A1"), ("A_add", "+A2"), ("B_prune", "+B"), ("BD_stop", "+D"),
                      ("BDC_look", "+C"), ("BDCE_all", "+E"))):
    """
    LVISのテスト用の画像における，構成（横軸）ごとのCOCO外の物体の検出・命名の割合とCOCOの物体の命名の割合（縦軸）（ex013）．

    Args:
        configs (tuple): (構成名, 横軸のラベル) の組．

    Returns:
        None
    """
    t = pd.read_csv(os.path.join(EX013, "table_results.csv")).set_index("config")
    x = range(len(configs))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for k, (col, label, color) in enumerate([("noncoco_detected", "Non-COCO: detected", COLORS[0]),
                                             ("noncoco_named", "Non-COCO: named", COLORS[1]),
                                             ("coco_named", "COCO: named", COLORS[2])]):
        ax.bar([i + (k - 1) * 0.27 for i in x], [t.loc[c, col] for c, _ in configs], 0.27, color=color, label=label)
    ax.set_xticks(list(x), [label for _, label in configs])
    ax.set_ylabel("Fraction of LVIS objects")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    savefig(fig, "fig_open")


def fig_stages(stages=(("S1", "flat", "flat"), ("S1", "deepest", "deepest"), ("S1", "greedy", "greedy"),
                       ("S2", "ens_noperson", "+text"), ("S4", "k20", "K=20"), ("S4", "k50", "K=50"), ("S4", "k100", "K=100"))):
    """
    ex015の段階ごとの，テスト用の画像におけるCOCO外の物体の検出・命名・細分化の割合（OWLv2はLVISの人手の注釈を用いない重み）．

    Args:
        stages (tuple): (段階, 構成名, 横軸のラベル) の組．

    Returns:
        None
    """
    t = pd.read_csv(os.path.join(EX015, "table_results.csv"))
    t = t[t["model"] == "st"].set_index(["stage", "name"])
    x = range(len(stages))
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for k, (col, label, color) in enumerate([("noncoco_detected", "detected", COLORS[0]), ("noncoco_named", "named", COLORS[1]),
                                             ("noncoco_fine", "fine", COLORS[2])]):
        ax.bar([i + (k - 1) * 0.27 for i in x], [t.loc[(a, b), col] for a, b, _ in stages], 0.27, color=color, label=label)
    ax.set_xticks(list(x), [label for _, _, label in stages], fontsize=5.5)
    ax.set_ylabel("Fraction of non-COCO objects")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    savefig(fig, "fig_stages")


def fig_tradeoff(models=(("st", "OWLv2 B/16 (self-training only)"), ("ens", "OWLv2 B/16 (with LVIS-base fine-tuning)"),
                         ("st_large", "OWLv2 L/14 (self-training only)"))):
    """
    加える領域の数 K（棄却なし）と物体らしさの閾値 o（K=100）を変えた場合の，加えた領域の適合率（横軸）とCOCO外の物体の命名の割合（縦軸）．

    Args:
        models (tuple): (OWLv2の重みの名前, 凡例) の組．

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for (key, label), color in zip(models, COLORS):
        path = os.path.join(EX015, f"tradeoff_{key}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        k = d[d["objectness"] == 0.0].sort_values("k_add")
        o = d[d["k_add"] == 100].sort_values("objectness")
        ax.plot(k["test_added_precision"], k["test_noncoco_named"], "-o", color=color, ms=3, label=label)
        ax.plot(o["test_added_precision"], o["test_noncoco_named"], "--s", color=color, ms=2.5, alpha=0.7)
        for _, r in k.iterrows():
            ax.annotate(f"K={int(r['k_add'])}", (r["test_added_precision"], r["test_noncoco_named"]), fontsize=5,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Precision of added regions")
    ax.set_ylabel("Non-COCO objects named")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=5)
    savefig(fig, "fig_tradeoff")


def fig_qualitative_open(image_ids=("64462", "283113", "304396"), rows=(("st", "self-training only"), ("ens", "with LVIS fine-tuning"))):
    """
    ex015aの定性評価（青：YOLO11xのボックスとクラス，赤：校正したスコアが高い加えた領域の上位8件と選んだ語．数値はアノテーションのスコア）．

    Args:
        image_ids (tuple[str]): 表示する画像ID．
        rows (tuple): (OWLv2の重みの名前, 行の見出し) の組．

    Returns:
        None
    """
    coco = load_coco("instances")
    images = {i: load_image(coco, int(i)) for i in image_ids}
    ratios = [images[i].size[0] / images[i].size[1] for i in image_ids]
    width, gap, left = 7.0, 0.04, 0.22
    height = (width - left - gap * (len(image_ids) - 1)) / sum(ratios)
    total_h = height * len(rows) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(width, total_h))
    for r, (key, title) in enumerate(rows):
        with open(os.path.join(EX015A, "qualitative", f"{key}.json")) as f:
            results = json.load(f)
        x = left
        y = total_h - (r + 1) * height - r * gap
        for image_id, ratio in zip(image_ids, ratios):
            ax = fig.add_axes([x / width, y / total_h, ratio * height / width, height / total_h])
            x += ratio * height + gap
            ax.imshow(images[image_id])
            ax.set_xticks([])
            ax.set_yticks([])
            for item in results[image_id]:
                x0, y0, x1, y1 = item["box"]
                color = COLORS[0] if item["source"] == "yolo" else COLORS[3]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=0.8))
                ax.text(x0, y0, f"{item['word']} {item['score']:.2f}", fontsize=4.6, color="white", va="top", clip_on=True,
                        bbox=dict(facecolor=color, alpha=0.8, pad=0.6, lw=0))
        fig.text(0.02, (y + height / 2) / total_h, title, rotation=90, va="center", ha="center")
    savefig(fig, "fig_qualitative_open")


def fig_score(model="st", scores=(("det", "detection score"), ("det_x_name_added", "det x q (added)"), ("calibrated", "calibrated"))):
    """
    ex015aのスコアの評価：スコアの順に出力を加えた場合の hR と hP_strict（左），出力のスコアの校正（中央），
    経路上の各単語のスコアの校正（右）．

    Args:
        model (str): OWLv2の重みの名前．
        scores (tuple): (スコアの名前, 凡例) の組．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for (kind, label), color in zip(scores, COLORS):
        c = pd.read_csv(os.path.join(EX015A, kind, model, "42", "curve_test.csv"))
        c = c[c["hR"] > 0]  # 出力が1件もGTと共通のSynsetを持たない先頭の点は除く
        axes[0].plot(c["hR"], c["strict_hP"], color=color, label=label)
        rel = json.load(open(os.path.join(EX015A, kind, model, "42", "reliability.json")))
        if kind != "det_x_name_added":
            axes[1].plot([r["score"] for r in rel], [r["accuracy"] for r in rel], "-o", ms=2.5, color=color, label=label)
    chain = json.load(open(os.path.join(EX015A, f"chain_reliability_{model}.json")))
    for key, label, color in [("table_q", "q", COLORS[1]), ("table_calibrated", "calibrated", COLORS[2])]:
        axes[2].plot([r["score"] for r in chain[key]], [r["accuracy"] for r in chain[key]], "-o", ms=2.5, color=color, label=label)
    axes[0].set_xlabel("hR")
    axes[0].set_ylabel("hP (strict)")
    axes[0].set_title("(a) ranking by score")
    for ax, title in [(axes[1], "(b) annotation score"), (axes[2], "(c) score of each word")]:
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("score")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=5)
    fig.tight_layout()
    savefig(fig, "fig_score")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_theta()
    fig_qualitative()
    fig_stages()
    fig_tradeoff()
    fig_qualitative_open()
    fig_score()
    print(sorted(os.listdir(FIG_DIR)))
