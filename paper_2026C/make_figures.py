"""
論文（paper_2026C）の図を outputs/ex019〜ex021 の結果から作成し，paper_2026C/figures/ に保存する．

order_039：検出器と分類器の比較の図 fig_detector（ex010 の LVIS の再現率と ex021 の E3_detector の F値）を追加した．

実行: .venv_pytorch/bin/python paper_2026C/make_figures.py
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER_DIR)
EX020 = os.path.join(ROOT, "outputs", "ex020_coco_owl_alphaclip_granularity")
EX021 = os.path.join(ROOT, "outputs", "ex021_coco_owl_rejection")
EX010 = os.path.join(ROOT, "outputs", "ex010_lvis_open_world_detection")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
E3_DETECTORS = [("YOLO11x", "YOLO11x"), ("DETR", "DETR"), ("OWLv2@20", "OWLv2@20"), ("OWLv2@50", "OWLv2@50")]
E3_EMBEDDERS = [("CLIP-B32", "CLIP-B/32"), ("Alpha-crop", "Alpha-CLIP (crop)"), ("Alpha-context1.5", "Alpha-CLIP (context)")]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
SCORE_TYPES = ["joint", "objectness", "vlm", "mass", "mass_joint"]
SCORE_LABELS = ["joint", "obj.", "VLM", "mass", "mass$\\times$obj."]
DEDUPS = [("none", "none"), ("nms_contain", "NMS + containment")]
BEST = "New_OWLv2@50_crop_nms_contain_mass_joint_masslabel"
CURRENT = "New_OWLv2@20_context1.5_none_joint"
OLD = "Old_DETR_CLIP-B32_none_vlm"
YOLO = "YOLO11x_closed"
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


def summary():
    """
    ex021 の test の結果の表（θ は dev の F値で選んだもの）を読み込む．

    Returns:
        pandas.DataFrame: 手法名を索引とする表．
    """
    df = pd.read_csv(os.path.join(EX021, "E1_rejection", "summary_test.csv"))
    return df[df.criterion == "f1"].set_index("method")


def fig_rejection():
    """
    (a) 棄却のスコアごとの test の F値（OWLv2@50，crop，mass ラベル，除去の有無），
    (b) 全ての条件の COCO の適合率と人物の再現率．

    Returns:
        None
    """
    df = summary()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    ax = axes[0]
    for k, (dedup, label) in enumerate(DEDUPS):
        vals = [df.loc[f"New_OWLv2@50_crop_{dedup}_{s}_masslabel", "f1"] for s in SCORE_TYPES]
        ax.bar(np.arange(len(SCORE_TYPES)) + (k - 0.5) * 0.38, vals, 0.38, color=COLORS[k], label=f"New, {label}")
    for ref, ls, label in ((OLD, "--", "Old"), (YOLO, ":", "YOLO11x")):
        ax.axhline(df.loc[ref, "f1"], ls=ls, color="k", lw=0.8, label=f"{label} ({df.loc[ref, 'f1']:.3f})")
    ax.set_xticks(np.arange(len(SCORE_TYPES)), SCORE_LABELS)
    ax.set_ylabel("COCO F1 (test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Rejection score (OWLv2@50, crop, mass label)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    new = df[df.index.str.startswith("New")]
    for score, color in zip(SCORE_TYPES, COLORS):
        sub = new[new.score_type == score]
        ax.scatter(sub.precision, sub.sub_person_recall, s=6, color=color, alpha=0.7, label=score.replace("_", "$\\times$") if score == "mass_joint" else score)
    for name, marker, label in ((BEST, "*", "Best New"), (CURRENT, "X", "Joint (prev.)"), (OLD, "s", "Old"), (YOLO, "^", "YOLO11x")):
        ax.scatter([df.loc[name, "precision"]], [df.loc[name, "sub_person_recall"]], marker=marker, s=40, color="k",
                   facecolor="none" if name != BEST else "k", label=label)
    ax.set_xlabel("COCO precision (test)")
    ax.set_ylabel("Person recall (test)")
    ax.set_title("(b) All 120 configurations")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig_rejection")


def fig_detector():
    """
    (a)(b) LVIS（ex010）の1枚あたりのボックス数とクラス非依存の再現率（COCO の物体，COCO 外の物体），
    (c) 提案手法の検出器 D と分類器 F を替えた test の F値（ex021 の E3_detector，提案の判定）．

    Returns:
        None
    """
    rec = pd.read_csv(os.path.join(EX010, "table_recall.csv")).set_index("detector")
    ks = [1, 5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    names = [("yolo_low", "YOLO11x", COLORS[0]), ("detr_low", "DETR", COLORS[1]), ("owlv2", "OWLv2", COLORS[3])]
    for ax, kind, title in ((axes[0], "coco", "(a) LVIS, COCO objects"), (axes[1], "noncoco", "(b) LVIS, non-COCO objects")):
        for key, label, color in names:
            ax.plot([rec.loc[key, f"boxes@{k}"] for k in ks], [rec.loc[key, f"{kind}_recall@{k}"] for k in ks],
                    "-o", ms=2.5, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Boxes per image")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall (IoU $\\geq$ 0.5)")
    axes[0].legend(frameon=False, loc="upper left")
    e3 = pd.read_csv(os.path.join(EX021, "E3_detector", "summary_test.csv")).set_index("method")
    ax = axes[2]
    width = 0.8 / len(E3_EMBEDDERS)
    for k, (emb, label) in enumerate(E3_EMBEDDERS):
        vals = [e3.loc[f"{d}_{emb}_proposed", "f1"] for d, _ in E3_DETECTORS]
        bars = ax.bar(np.arange(len(E3_DETECTORS)) + (k - 1) * width, vals, width, color=COLORS[[1, 0, 2][k]], label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=4, rotation=90)
    ax.set_xticks(np.arange(len(E3_DETECTORS)), [l.replace("@", "\n@") for _, l in E3_DETECTORS])
    ax.set_ylabel("COCO F1 (test)")
    ax.set_ylim(0, 0.9)
    ax.set_title("(c) Detector $\\times$ classifier")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig_detector")


def fig_step():
    """
    正解ボックスで正解の経路を1段ずつ与えた場合の各段の正解率（人物，WordNet の木と WordNet∩VG の木）．

    Returns:
        None
    """
    s = pd.read_csv(os.path.join(EX020, "E3_step", "summary_test.csv"))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for k, (emb, label) in enumerate([("CLIP-ViT-B32", "CLIP-B/32"), ("AlphaCLIP-ViT-B16_context1.5", "Alpha-CLIP")]):
        for tree, ls in (("WordNet", "--"), ("VG", "-")):
            g = s[(s.embedder == emb) & (s.tree == tree) & (s.group == "person") & (s.step_c <= 3)]
            tree_name = "WordNet$\\cap$VG" if tree == "VG" else tree  # order_043：VG の木は WordNet∩VG の木
            ax.plot(g.step_c, g.top1, ls, marker="o", ms=3, color=COLORS[k], label=f"{label}, {tree_name} tree")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Step on the correct path (person)")
    ax.set_ylabel("Top-1 of the correct child")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right", fontsize=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig_step")


def load_module_file(name, path):
    """
    ファイルからモジュールを読み込む（実験の model.py は同じ名前のため，別名で読み込む）．

    Args:
        name (str): モジュールの別名．
        path (str): ファイルのパス．

    Returns:
        module: モジュール．
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kept(record, cfg, theta, rej):
    """
    ex021 の条件で残すボックスのインデックスと表示の語を返す．

    Args:
        record (dict): ex021 のソースの1枚分．
        cfg (dict): "score_type"，"dedup"，"label"．
        theta (float): θ．
        rej (module): ex021 の model モジュール．

    Returns:
        list[tuple[np.ndarray, str]]: (ボックス, 語) のリスト．
    """
    top = record["top_mass"] if cfg["label"] == "mass" else record["top"]
    s = rej.rejection_score(cfg["score_type"], record["conf"], record["det_scores"], record["mass"]).astype(np.float64)
    keep = rej.dedup_mask(record["boxes"], [t[0] for t in top], s, cfg["dedup"])
    s = np.where(keep, s, -np.inf)
    return [(record["boxes"][k], record["out_sub"][k][1]) for k in np.where(s >= theta)[0]]


def e3_config(method):
    """
    ex021 の E3_detector の手法名から，残すボックスの条件と θ を返す．

    Args:
        method (str): E3_detector の手法名（例："YOLO11x_Alpha-crop_proposed"）．

    Returns:
        tuple[dict, float]: ("score_type"，"dedup"，"label" の辞書, θ)．
    """
    e3 = pd.read_csv(os.path.join(EX021, "E3_detector", "summary_test.csv")).set_index("method")
    cfg = {"score_type": "mass_joint", "dedup": "nms_contain", "label": "mass"}
    return cfg, float(e3.loc[method, "theta"])


def two_level_label(record, k, word):
    """
    細かい語と深さ0の語（COCO のクラス）の2つの粒度の表示を作る．

    Args:
        record (dict): ex021 のソースの1枚分．
        k (int): ボックスのインデックス．
        word (str): 部分木の確率の合計で選んだ語．

    Returns:
        str: 同じなら語のみ，異なれば "細かい語 (深さ0の語)"．
    """
    coarse = record["top_mass"][k][0]
    return word if word == coarse else f"{word} ({coarse})"


def kept_two_level(record, cfg, theta, rej):
    """
    残すボックスと2つの粒度の表示を返す．

    Args:
        record (dict): ex021 のソースの1枚分．
        cfg (dict): "score_type"，"dedup"，"label"．
        theta (float): θ．
        rej (module): ex021 の model モジュール．

    Returns:
        list[tuple[np.ndarray, str]]: (ボックス, 表示) のリスト．
    """
    s = rej.rejection_score(cfg["score_type"], record["conf"], record["det_scores"], record["mass"]).astype(np.float64)
    keep = rej.dedup_mask(record["boxes"], [t[0] for t in record["top_mass"]], s, cfg["dedup"])
    s = np.where(keep, s, -np.inf)
    return [(record["boxes"][k], two_level_label(record, k, record["out_sub"][k][1])) for k in np.where(s >= theta)[0]]


def fig_qualitative(image_ids=(0, 1, 4), n_images=6, name="fig_qualitative", width=7.0, row_h=1.55, label_size=4.5, title_size=6):
    """
    test の画像に対する，検出器を替えたモデルの出力（(a) OWLv2，(b) YOLO11x のボックス，(c) DETR のボックス，
    分類器はいずれも Alpha-CLIP（crop））と (d) YOLO11x 単体の出力．
    (a)〜(c) は細かい語と深さ0の語の2つの粒度を表示する（YOLO11x 単体はクラス名）．

    Args:
        image_ids (tuple[int]): 比較する画像の行の番号（test の画像IDの順に等間隔に選んだ n_images 枚のうち）．
        n_images (int): 等間隔に選ぶ画像の枚数．
        name (str): 保存するファイル名（拡張子なし）．
        width (float): 図の幅（インチ）．
        row_h (float): 1行の高さ（インチ）．
        label_size (float): ラベルの文字の大きさ．
        title_size (float): 列の見出しの文字の大きさ．

    Returns:
        None
    """
    import torch
    from PIL import Image
    rej = load_module_file("ex021_model", os.path.join(ROOT, "programs", "ex021_coco_owl_rejection", "model.py"))
    df = summary()
    src_dir = os.path.join(ROOT, "datasets", "ex021_coco_owl_rejection")
    files = {"owl": ("source_OWLv250_AlphaCLIP-ViT-B16_crop_test.pt", "OWLv2@50_Alpha-crop_proposed"),
             "yolo": ("source_YOLO11x_AlphaCLIP-ViT-B16_crop_test.pt", "YOLO11x_Alpha-crop_proposed"),
             "detr": ("source_DETR_AlphaCLIP-ViT-B16_crop_test.pt", "DETR_Alpha-crop_proposed")}
    sources = {k: {r["image_id"]: r for r in torch.load(os.path.join(src_dir, f), weights_only=False)} for k, (f, _) in files.items()}
    configs = {k: e3_config(m) for k, (_, m) in files.items()}
    yolo = torch.load(os.path.join(ROOT, "datasets", "ex017_coco_granular_annotation", "detections_YOLO11x_test.pt"), weights_only=False)
    ids = sorted(sources["owl"])
    ids = [ids[::len(ids) // n_images][:n_images][k] for k in image_ids]
    coco_dir = os.path.join(ROOT, "datasets", "coco", "val2017")
    titles = ["(a) OWLv2 + Alpha-CLIP", "(b) YOLO11x boxes + Alpha-CLIP", "(c) DETR boxes + Alpha-CLIP", "(d) YOLO11x (closed set)"]
    fig, axes = plt.subplots(len(ids), 4, figsize=(width, row_h * len(ids)))
    for r, i in enumerate(ids):
        image = Image.open(os.path.join(coco_dir, f"{i:012d}.jpg")).convert("RGB")
        outs = [kept_two_level(sources[k][i], *configs[k], rej) for k in ("owl", "yolo", "detr")]
        y = yolo[i]
        th = df.loc[YOLO, "theta"]
        outs.append([(b, l) for b, sc, l in zip(np.asarray(y["boxes"]), np.asarray(y["scores"]), y["labels"]) if sc >= th])
        for c, out in enumerate(outs):
            ax = axes[r, c]
            ax.imshow(image)
            ax.axis("off")
            for box, label in out:
                x0, y0, x1, y1 = box
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", lw=0.8))
                if label:
                    ax.text(x0, y0, label, fontsize=label_size, va="top" if y0 < 0.06 * image.height else "bottom", bbox={"facecolor": "cyan", "alpha": 0.75, "pad": 0.4, "lw": 0})
            if r == 0:
                ax.set_title(titles[c], fontsize=title_size)
    fig.tight_layout(pad=0.2)
    savefig(fig, name)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_detector()
    fig_step()
    fig_qualitative()
    fig_qualitative(image_ids=(0, 4), name="fig_qualitative_slide", width=12.0, row_h=2.7, label_size=10, title_size=12)
    print("saved to", FIG_DIR)
