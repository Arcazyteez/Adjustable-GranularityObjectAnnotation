"""新しい提案手法（OWLv2 + Alpha-CLIP + WordNet）の ex018〜ex021 の結果をまとめた図を作る．

order_035 で作成した．新しい実験は行わず，outputs/ex018〜ex021 の CSV から
「OWL と YOLO の比較」と「CLIP と Alpha-CLIP の比較」の図を作る．

出力：outputs/summary_new_method/fig_owl_yolo.png，fig_clip_alphaclip.png，table_summary.csv

実行：.venv_*/bin/python programs/summary_new_method/make_summary.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SAVE = OUT / "summary_new_method"

EX018 = OUT / "ex018_owl_region_check" / "table_ex018.csv"
EX019_RECALL = OUT / "ex019_coco_owl_alphaclip_wordnet" / "E2_pipeline" / "recall_person_test.csv"
EX019_E1 = OUT / "ex019_coco_owl_alphaclip_wordnet" / "E1_naming" / "summary.csv"
EX019_E2 = OUT / "ex019_coco_owl_alphaclip_wordnet" / "E2_pipeline" / "summary_test.csv"
EX020 = OUT / "ex020_coco_owl_alphaclip_granularity" / "summary_test.csv"
EX020_STEP = OUT / "ex020_coco_owl_alphaclip_granularity" / "E3_step" / "summary_test.csv"
EX021 = OUT / "ex021_coco_owl_rejection" / "E1_rejection" / "summary_test.csv"

BEST_NEW = "New_OWLv2@50_crop_nms_contain_mass_joint_masslabel"
CURRENT_NEW = "New_OWLv2@20_context1.5_none_joint"
OLD = "Old_DETR_CLIP-B32_none_vlm"
YOLO = "YOLO11x_closed"
ALPHA = "AlphaCLIP-ViT-B16_context1.5"
CLIP = "CLIP-ViT-B32"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def annotate_bars(ax, bars, fmt="{:.2f}"):
    """棒グラフの各棒の上に値を書く．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        bars (BarContainer): ax.bar の戻り値．
        fmt (str): 値の書式．

    Returns:
        None
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, fmt.format(height),
                ha="center", va="bottom", fontsize=9)


def grouped_bars(ax, groups, series, values, colors, ylabel, title):
    """グループ化した棒グラフを描く．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        groups (list[str]): 横軸のグループ名，長さ G．
        series (list[str]): 系列名，長さ S．
        values (np.ndarray): 値，形状 (S, G)．
        colors (list[str]): 系列の色，長さ S．
        ylabel (str): 縦軸の名前．
        title (str): 題名．

    Returns:
        None
    """
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for i, (name, color) in enumerate(zip(series, colors)):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, values[i], width, label=name, color=color)
        annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, float(np.nanmax(values)) + 0.25)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=len(series) if len(series) < 3 else 1)


def ex021_row(method, criterion="f1"):
    """ex021 の test の主要な比較の1行を取り出す．

    Args:
        method (str): 手法名（summary_test.csv の method 列）．
        criterion (str): θ の選び方（"f1" または "hf"）．

    Returns:
        pd.Series: 該当する行．
    """
    df = pd.read_csv(EX021)
    return df[(df["method"] == method) & (df["criterion"] == criterion)].iloc[0]


def plot_owl_yolo():
    """OWL と YOLO の比較の図（3つのパネル）を作る．

    Args:
        なし．

    Returns:
        pd.DataFrame: 図に用いた主要な数値，形状 (行数, 列数)．
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) ex018：LVIS での YOLO と OWLv2 objectness の領域の被覆
    df18 = pd.read_csv(EX018)
    df18 = df18[(df18["condition"] == "OWLv2_objectness") & df18["point"].isin(["budget10", "budget20", "budget50"])]
    vals = np.array([df18["coco_owl_given_yolo_hit"].to_numpy(), df18["noncoco_owl_given_yolo_miss"].to_numpy()])
    grouped_bars(axes[0], ["10個/枚", "20個/枚", "50個/枚"],
                 ["COCOの物体：P(OWL|YOLO検出)", "COCO外の物体：P(OWL|YOLO見逃し)"],
                 vals, ["#4C72B0", "#DD8452"], "割合",
                 "(a) LVIS：OWLv2の物体らしさの被覆（ex018）")
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].set_ylim(0, 1.35)

    # (b) ex019：COCO のクラス非依存の再現率（人物・人物以外）
    df19 = pd.read_csv(EX019_RECALL).set_index("detector")
    order = ["YOLO11x", "DETR", "OWLv2@20", "OWLv2@50"]
    vals = np.array([df19.loc[order, "recall_person"].to_numpy(), df19.loc[order, "recall_other"].to_numpy()])
    labels = [f"{d}\n({df19.loc[d, 'boxes_per_image']:.1f}個/枚)" for d in order]
    grouped_bars(axes[1], labels, ["人物", "人物以外"], vals, ["#55A868", "#8172B3"],
                 "再現率（IoU 0.5）", "(b) COCO：検出器の再現率（ex019）")
    axes[1].tick_params(axis="x", labelsize=9)

    # (c) ex021：検出から命名までの F値・hF・人物の再現率
    rows = {"YOLO11x\n単体": ex021_row(YOLO), "New\n(最良)": ex021_row(BEST_NEW),
            "Old": ex021_row(OLD), "New\n(ex019)": ex021_row(CURRENT_NEW)}
    vals = np.array([[r["f1"] for r in rows.values()], [r["sub_hF"] for r in rows.values()],
                     [r["sub_person_recall"] for r in rows.values()]])
    grouped_bars(axes[2], list(rows), ["COCOのF値", "細かい粒度のhF", "人物の再現率"], vals,
                 ["#4C72B0", "#C44E52", "#55A868"], "値", "(c) COCO：検出から命名まで（ex021，test）")
    fig.tight_layout()
    fig.savefig(SAVE / "fig_owl_yolo.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame([{"method": k.replace("\n", ""), "f1": r["f1"], "top1": r["top1_acc"],
                          "precision": r["precision"], "recall": r["recall"], "sub_hF": r["sub_hF"],
                          "person_recall": r["sub_person_recall"], "kept_per_image": r["kept_per_image"]}
                         for k, r in rows.items()])


def ex019_top1(detector, embedder):
    """ex019 の E2 の θ あり・joint の Top-1 と F値を取り出す（予算は dev で選んだもの）．

    Args:
        detector (str): "DETR" または "OWLv2"．
        embedder (str): F の名前．

    Returns:
        tuple[float, float]: (Top-1, F値)．
    """
    df = pd.read_csv(EX019_E2)
    row = df[(df["method"] == f"{detector}+{embedder}+joint") & (df["rejection"] == "theta")].iloc[0]
    return row["top1_acc"], row["f1"]


def plot_clip_alphaclip():
    """CLIP と Alpha-CLIP の比較の図（3つのパネル）を作る．

    Args:
        なし．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) ex019：正解ボックスと検出ボックスでの深さ0の Top-1
    e1 = pd.read_csv(EX019_E1)
    e1 = e1[e1["split"] == "test"].set_index("embedder")
    groups = ["正解ボックス", "DETRのボックス", "OWLv2のボックス"]
    vals = np.array([[e1.loc[CLIP, "top1_acc"], ex019_top1("DETR", CLIP)[0], ex019_top1("OWLv2", CLIP)[0]],
                     [e1.loc[ALPHA, "top1_acc"], ex019_top1("DETR", ALPHA)[0], ex019_top1("OWLv2", ALPHA)[0]]])
    grouped_bars(axes[0], groups, ["CLIP-ViT-B/32", "Alpha-CLIP（context1.5）"], vals,
                 ["#8C8C8C", "#C44E52"], "Top-1（COCOの80クラス）", "(a) 深さ0の命名（ex019，test）")

    # (b) ex020：正解ボックスでの細かい粒度の hF（命名の手順ごと）
    e20 = pd.read_csv(EX020)
    e20 = e20[e20["experiment"] == "E1_gt"].set_index(["method", "procedure"])
    procs = ["coarse", "wn_hyponym", "vg_tree", "vg_subtree"]
    vals = np.array([[e20.loc[(CLIP, p), "hF"] for p in procs], [e20.loc[(ALPHA, p), "hF"] for p in procs]])
    grouped_bars(axes[1], ["COCOのみ", "WordNetの\n下位語", "VGの木", "部分木の\n確率の合計"],
                 ["CLIP-ViT-B/32", "Alpha-CLIP（context1.5）"], vals, ["#8C8C8C", "#C44E52"],
                 "hF", "(b) 正解ボックスの細かい粒度（ex020，test）")

    # (c) ex020：正解の経路を1段ずつ与えた各段の正解率
    st = pd.read_csv(EX020_STEP).set_index(["embedder", "tree", "group", "step_c"])
    keys = [("WordNet", "person", 1), ("VG", "person", 1), ("WordNet", "person", 2),
            ("WordNet", "person", 3), ("WordNet", "non-person", 1), ("VG", "non-person", 1)]
    names = ["人物1段目\nWordNet", "人物1段目\nVG", "人物2段目\nWordNet", "人物3段目\nWordNet",
             "非人物1段目\nWordNet", "非人物1段目\nVG"]
    vals = np.array([[st.loc[(e, *k), "top1"] for k in keys] for e in [CLIP, ALPHA]])
    grouped_bars(axes[2], names, ["CLIP-ViT-B/32", "Alpha-CLIP（context1.5）"], vals,
                 ["#8C8C8C", "#C44E52"], "Top-1", "(c) 細分化の各段の正解率（ex020，正解ボックス）")
    axes[2].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(SAVE / "fig_clip_alphaclip.png", dpi=150)
    plt.close(fig)


def set_japanese_font():
    """日本語のフォントを設定する（見つからなければ既定のまま）．

    Args:
        なし．

    Returns:
        None
    """
    from matplotlib import font_manager
    for name in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Noto Sans JP"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return


def main():
    """図と表を作って保存する．

    Args:
        なし．

    Returns:
        None
    """
    SAVE.mkdir(parents=True, exist_ok=True)
    set_japanese_font()
    table = plot_owl_yolo()
    plot_clip_alphaclip()
    table.to_csv(SAVE / "table_summary.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
