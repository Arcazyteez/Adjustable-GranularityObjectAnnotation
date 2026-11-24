"""
新しい論文（paper_v2）の図を outputs/ のex009，ex010の結果から作成し，paper_v2/figures/ に保存する．

実行: .venv_pytorch/bin/python paper_v2/make_figures.py
"""
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER_DIR)
OUT = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
EX009 = os.path.join(OUT, "ex009_coco_siglip_wordnet_annotation")
EX010 = os.path.join(OUT, "ex010_lvis_open_world_detection")
EX011 = os.path.join(OUT, "ex011_open_world_naming")
DETECTOR_LABELS = {"yolo_low": "YOLO11x", "detr_low": "DETR", "owlv2": "OWLv2", "gdino": "Grounding DINO"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6})


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


def fig_lvis_stopping(conditions=(("yolo", "none", 0.25, "YOLO11x"), ("proposed", "sigmoid", 0.5, "Proposed"),
                                  ("hybrid", "sigmoid", 0.25, "Hybrid"))):
    """
    LVISのテスト用分割における，累積確信度の停止規則の階層的再現率 hR と階層的適合率 hP の関係（ex009）．

    Args:
        conditions (tuple): (手法, 確信度, 閾値, 凡例) の組．

    Returns:
        None
    """
    stop = pd.read_csv(os.path.join(EX009, "table_stopping.csv"))
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for (method, norm, thr, label), color in zip(conditions, ["#555555", "#4C72B0", "#DD8452"]):
        curve = pd.read_csv(os.path.join(EX009, method, norm, f"lvis_test_th{thr}", "42", "stopping_curve.csv"))
        tau = curve[curve["rule"] == "tau"]
        if method == "yolo":
            ax.plot(tau["hR"].iloc[0], tau["hP"].iloc[0], "D", color=color, ms=5, label=label)
            continue
        ax.plot(tau["hR"], tau["hP"], "-o", color=color, ms=2.5, lw=1, label=f"{label}: τ")
        depth = curve[curve["rule"] == "depth"]
        ax.plot(depth["hR"], depth["hP"], "s", color=color, mfc="none", ms=4, label=f"{label}: depth limit")
        selected = stop[(stop["method"] == method) & (stop["normalization"] == norm) & (stop["threshold"] == thr)
                        & (stop["stopping"] == "tau")].iloc[0]
        ax.plot(selected["hR"], selected["hP"], "*", color="red", ms=8)
    ax.set_xlabel("Hierarchical recall hR")
    ax.set_ylabel("Hierarchical precision hP")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    savefig(fig, "fig_lvis_stopping")


def fig_recall():
    """
    1画像あたりのボックス数（横軸）に対するCOCOの物体とCOCO外の物体の再現率（縦軸）（ex010）．

    Returns:
        None
    """
    table = pd.read_csv(os.path.join(EX010, "table_recall.csv")).set_index("detector")
    ks = [1, 5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1), sharey=True)
    for ax, kind, title in zip(axes, ["coco", "noncoco"], ["(a) COCO objects", "(b) Non-COCO objects"]):
        for (name, label), color in zip(DETECTOR_LABELS.items(), ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
            ax.plot([table.loc[name, f"boxes@{k}"] for k in ks], [table.loc[name, f"{kind}_recall@{k}"] for k in ks],
                    "-o", ms=2, lw=1, color=color, label=label)
        for name, marker, label in [("yolo0.25", "D", "YOLO11x (0.25)"), ("detr0.5", "^", "DETR (0.5)")]:
            ax.plot(table.loc[name, "boxes@all"], table.loc[name, f"{kind}_recall@all"], marker, color="black", ms=4,
                    label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Boxes per image")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall (IoU ≥ 0.5)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    savefig(fig, "fig_recall")


def fig_qualitative(names=("camel", "kangaroo", "lighthouse_1", "lighthouse_2"), detector="owlv2", n_extra=3):
    """
    ex004の実画像に対する，YOLO11xにOWLv2の候補を加えた構成の結果（ex010）．
    青はYOLO11xのボックス（ハイブリッド手法と同じ），赤は加えた候補のうち検出スコアの上位 n_extra 件である．

    Args:
        names (tuple[str]): 表示する画像の名前（2×2に配置する）．
        detector (str): 加える候補の検出器．
        n_extra (int): 表示する加えた候補の数．

    Returns:
        None
    """
    with open(os.path.join(EX010, "qualitative", "annotations.json")) as f:
        results = json.load(f)["results"]
    images = {os.path.splitext(os.path.basename(p))[0]: Image.open(p).convert("RGB")
              for p in glob.glob(os.path.join(ROOT, "datasets", "ex004_open_vocab_qualitative", "*.jpg"))}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    for ax, name in zip(axes.flat, names):
        regions = results[f"{detector}/{name}"]
        shown = [r for r in regions if r["source"] == "yolo"] + [r for r in regions if r["source"] != "yolo"][:n_extra]
        ax.imshow(images[name])
        ax.axis("off")
        for r in shown:
            x0, y0, x1, y1 = r["box"]
            color = "#4C72B0" if r["source"] == "yolo" else "#C44E52"
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1.2))
            head = "YOLO: " if r["source"] == "yolo" else "VLM: "
            ax.text(x0, y0, head + " > ".join(c["word"] for c in r["chain"][:3]), fontsize=5.5, color="white",
                    va="top", bbox=dict(facecolor=color, alpha=0.8, pad=0.8, lw=0))
    fig.tight_layout()
    savefig(fig, "fig_qualitative")


def fig_budget(configs=(("ex010_baseline", "Baseline (top-K)"), ("p2_reject", "Rejection"),
                        ("p1_topdown", "Rejection + top-down"), ("p4_topdown_so400m", "+ SO400M"),
                        ("p34_topdown_all", "+ SO400M, crop 1.5, hier. prompt"))):
    """
    加えた候補の数（横軸）に対する，COCO外の物体の階層的再現率と加えた候補の命名の適合率（縦軸）（ex011）．

    Args:
        configs (tuple): (構成名, 凡例) の組．

    Returns:
        None
    """
    t = pd.read_csv(os.path.join(EX011, "table_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
    for (config, label), color in zip(configs, ["#555555", "#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
        rows = t[(t["config"] == config) & t["selection"].str.startswith("B")].sort_values("extra_per_image")
        axes[0].plot(rows["extra_per_image"], rows["noncoco_hR"], "-o", ms=2.5, lw=1, color=color, label=label)
        axes[1].plot(rows["extra_per_image"], rows["extra_precision"], "-o", ms=2.5, lw=1, color=color, label=label)
    for ax, title in zip(axes, ["(a) Non-COCO hR", "(b) Precision of added boxes"]):
        ax.set_xlabel("Added boxes per image")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    savefig(fig, "fig_budget")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_lvis_stopping()
    fig_recall()
    fig_qualitative()
    if os.path.exists(os.path.join(EX011, "table_results.csv")):
        fig_budget()
    print(sorted(os.listdir(FIG_DIR)))
