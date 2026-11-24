"""
paper_2026C の内容の研究発表スライド（slides_2026C.pptx）を作成するスクリプト（order_034）．

.ai/ai-pptx-kit/templates/research.pptx を展開し，slides/build_slides.py の部品（OOXMLの組み立て，LaTeXからOMMLへの変換）を用いて
各スライドを書き込む．スライド用の図は，配置する大きさで文字が20ポイントになるように outputs/ex019〜ex021 の結果から描き直す．
LibreOfficeは使用しない．
order_039：検出器（OWLv2，YOLO11x，DETR）と分類器（Alpha-CLIP，CLIP）の比較を加えて作り直した．
order_040：棄却のスコアの比較を除き，モデル（検出器，分類器，語彙の木による粒度）を中心とする構成に作り直した．
order_041：前稿のモデル（DETR＋CLIP）の紹介と，現在のモデルへの更新点と理由の3枚を加えた．
order_042：表を含む全てのスライドに，表の記号の簡単な説明（legend）を加えた．
order_043：「VGの語彙の木」の表記を「WordNet∩VGの木」に直した．
order_047：発表の台本を各スライドの発表者ノートに入れた（NOTES，add_notes）．
order_046：発表の台本（order_045）に合わせ，専門用語を言い換えて全てのスライドの文と表を簡単にした．
order_044：研究結果を簡単に伝えるため，τ を変えた例とトレードオフの2枚を加え，検出器の比較を2枚に縮めた（16枚）．
表の数値は outputs/ex021_coco_owl_rejection/E3_detector と outputs/ex022_yolo_detr_region_check から読む．

実行: PPTX_SKILL_SCRIPTS=<pptxスキルのscripts> .venv_pytorch/bin/python slides/build_slides_2026C.py
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_slides as bs  # noqa: E402
from build_slides import RED, Slide, bullet, math_para, math_run, note, para, run  # noqa: E402

ROOT = bs.ROOT
OUTPUT = os.path.join(ROOT, "slides_2026C.pptx")
EX020 = os.path.join(ROOT, "outputs", "ex020_coco_owl_alphaclip_granularity")
EX021 = os.path.join(ROOT, "outputs", "ex021_coco_owl_rejection")
EX022 = os.path.join(ROOT, "outputs", "ex022_yolo_detr_region_check")
EX010 = os.path.join(ROOT, "outputs", "ex010_lvis_open_world_detection")
PAPER_FIG = os.path.join(ROOT, "paper_2026C", "figures")
MEDIA_DIR = os.path.join(ROOT, "slides", "media_2026C")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
BEST = "New_OWLv2@50_crop_nms_contain_mass_joint_masslabel"
CURRENT = "New_OWLv2@20_context1.5_none_joint"
OLD = "Old_DETR_CLIP-B32_none_vlm"
YOLO = "YOLO11x_closed"

REFS = {
    1: "[1] G. Jocher and J. Qiu, “Ultralytics YOLO11,” https://github.com/ultralytics/ultralytics, 2024.",
    2: "[2] M. Minderer, A. Gritsenko, and N. Houlsby, “Scaling Open-Vocabulary Object Detection,” Proc. NeurIPS, 2023.",
    3: "[3] Z. Sun, et al., “Alpha-CLIP: A CLIP Model Focusing on Wherever You Want,” Proc. IEEE/CVF CVPR, 2024.",
    4: "[4] R. Krishna, et al., “Visual Genome,” International Journal of Computer Vision, vol.123, pp.32-73, 2017.",
    5: "[5] G. A. Miller, “WordNet: A Lexical Database for English,” Communications of the ACM, vol.38, no.11, pp.39-41, 1995.",
    6: "[6] T.-Y. Lin, et al., “Microsoft COCO: Common Objects in Context,” Proc. ECCV, pp.740-755, 2014.",
    7: "[7] N. Carion, et al., “End-to-End Object Detection with Transformers,” Proc. ECCV, 2020.",
    8: "[8] A. Radford, et al., “Learning Transferable Visual Models From Natural Language Supervision,” Proc. ICML, 2021.",
    9: "[9] A. Gupta, P. Dollár, and R. Girshick, “LVIS: A Dataset for Large Vocabulary Instance Segmentation,” Proc. IEEE/CVF CVPR, 2019.",
}


# ===================== スライド用の図 =====================

def _style():
    """スライド用の図の文字の大きさ（配置した大きさで20ポイント）を設定する．"""
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20,
                         "legend.fontsize": 18, "font.family": "DejaVu Sans"})


def summary():
    """
    ex021 の test の結果の表（θ は dev の F値で選んだもの）．

    Returns:
        pd.DataFrame: 手法名を索引とする表．
    """
    df = pd.read_csv(os.path.join(EX021, "E1_rejection", "summary_test.csv"))
    return df[df.criterion == "f1"].set_index("method")


def e3_summary():
    """
    ex021 の E3_detector の test の結果（検出器 × 分類器 × 棄却の判定）に，LVIS の COCO 外の物体の再現率を加えた表．

    Returns:
        pd.DataFrame: 手法名を索引とする表．
    """
    df = pd.read_csv(os.path.join(EX021, "E3_detector", "summary_test.csv")).set_index("method")
    lvis = pd.read_csv(os.path.join(EX021, "E3_detector", "lvis_test.csv")).set_index("method")
    return df.join(lvis[["noncoco_recall"]])


def detector_table():
    """
    検出器のみの比較の表（ex022 の COCO の値と，E3_detector の LVIS の COCO 外の物体の再現率）．

    Returns:
        pd.DataFrame: 検出器を索引とする表．
    """
    det = pd.read_csv(os.path.join(EX022, "table_detector_all.csv")).set_index("detector")
    lvis = pd.read_csv(os.path.join(EX021, "E3_detector", "lvis_test.csv")).set_index("method")
    det["noncoco_recall"] = [lvis.loc[f"{d}_all_boxes", "noncoco_recall"] for d in det.index]
    return det


def fig_lvis(path, w, h):
    """
    LVIS（ex010）の1枚あたりのボックス数と COCO 外の物体の再現率（YOLO11x，DETR，OWLv2）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    rec = pd.read_csv(os.path.join(EX010, "table_recall.csv")).set_index("detector")
    ks = [1, 5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(w, h))
    for key, label, color in (("yolo_low", "YOLO11x", COLORS[0]), ("detr_low", "DETR", COLORS[1]), ("owlv2", "OWLv2", COLORS[3])):
        ax.plot([rec.loc[key, f"boxes@{k}"] for k in ks], [rec.loc[key, f"noncoco_recall@{k}"] for k in ks], "-o", ms=8, lw=3,
                color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Boxes per image")
    ax.set_ylabel("Recall of non-COCO objects")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_scores(path, w, h):
    """
    棄却のスコアごとの test の F値（OWLv2@50，crop，mass ラベル，包含の除去あり）と，Old・YOLO11x の F値．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    df = summary()
    scores = ["joint", "objectness", "vlm", "mass", "mass_joint"]
    labels = ["q·o", "o", "q", "m", "m·o"]
    vals = [df.loc[f"New_OWLv2@50_crop_nms_contain_{s}_masslabel", "f1"] for s in scores]
    fig, ax = plt.subplots(figsize=(w, h))
    ax.bar(range(len(scores)), vals, 0.6, color=[COLORS[0]] * 4 + [COLORS[3]])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    ax.axhline(df.loc[OLD, "f1"], ls="--", color="k", lw=1.5, label=f"Old {df.loc[OLD, 'f1']:.3f}")
    ax.set_xticks(range(len(scores)), labels)
    ax.set_xlabel("Rejection score")
    ax.set_ylabel("COCO F1 (test)")
    ax.set_ylim(0, 0.5)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_step(path, w, h):
    """
    人物の GT ボックスで正しい経路を1段ずつ与えた場合の各段の正解率（Alpha-CLIP，WordNet の木と WordNet∩VG の木）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    s = pd.read_csv(os.path.join(EX020, "E3_step", "summary_test.csv"))
    fig, ax = plt.subplots(figsize=(w, h))
    for tree, color, label in (("WordNet", COLORS[0], "WordNet (398)"), ("VG", COLORS[3], "WordNet$\\cap$VG (136)")):
        g = s[(s.embedder == "AlphaCLIP-ViT-B16_context1.5") & (s.tree == tree) & (s.group == "person") & (s.step_c <= 2)]
        ax.plot(g.step_c, g.top1, "-o", ms=10, lw=3, color=color, label=label)
        for x, y in zip(g.step_c, g.top1):
            ax.text(x + 0.05, y + (0.05 if tree == "VG" else -0.08), f"{y:.3f}", color=color)
    ax.set_xticks([1, 2])
    ax.set_xlim(0.8, 2.4)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Step below person")
    ax.set_ylabel("Top-1 of correct child")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def crop_rows(src, dst, keeps=((0.0, 0.6),)):
    """
    論文の比較画像（3行）から指定した範囲の行を切り出し，縦に連結する（スライドでは画像を大きく表示するため）．

    Args:
        src (str): 論文の図のパス．
        dst (str): 保存先．
        keeps (tuple[tuple[float, float]]): 残す縦の範囲（割合）のリスト．

    Returns:
        str: 保存先．
    """
    from PIL import Image
    image = Image.open(src).convert("RGB")
    parts = [image.crop((0, int(a * image.height), image.width, int(b * image.height))) for a, b in keeps]
    out = Image.new("RGB", (image.width, sum(p.height for p in parts)), "white")
    y = 0
    for part in parts:
        out.paste(part, (0, y))
        y += part.height
    out.save(dst)
    return dst


#: τ の例として表示する (画像ID, 正解のインデックス)（order_044．slides/tau_examples_2026C.py の出力から選んだ）．
TAU_PICKS = [(450559, None), (492992, None), (149622, None)]
TAU_ROWS = [0.5, 0.3, 0.1, 0.05, 0.02]


def fig_tau_examples(path, w, h, picks=TAU_PICKS, taus=TAU_ROWS):
    """
    正解の箱で τ を変えたときの出力の語（order_044）．上段に箱を描いた画像の切り抜き，下段に τ ごとの語を示す．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．
        picks (list[tuple[int, int | None]]): (画像ID, 正解のインデックス)．インデックスが None なら最初の例．
        taus (list[float]): 表示する τ（降順）．

    Returns:
        str: 保存先．
    """
    import json

    from PIL import Image
    data = json.load(open(os.path.join(MEDIA_DIR, "tau_examples.json")))
    plt.rcParams["font.family"] = ["DejaVu Sans", "IPAexGothic"]  # order_046：「答えなし」を日本語で表示する
    fig = plt.figure(figsize=(w, h))
    n = len(picks)
    col_w = 1.0 / n
    for c, (image_id, g) in enumerate(picks):
        ex = next(e for e in data["examples"] if e["image_id"] == image_id and (g is None or e["gt_index"] == g))
        image = Image.open(os.path.join(ROOT, "datasets", "coco", "val2017", f"{image_id:012d}.jpg")).convert("RGB")
        x0, y0, x1, y1 = ex["box"]
        m = 0.25 * max(x1 - x0, y1 - y0)
        cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
        crop = image.crop((cx0, cy0, min(image.width, x1 + m), min(image.height, y1 + m)))
        ax = fig.add_axes([c * col_w + 0.01, 0.41, col_w - 0.02, 0.58])
        ax.imshow(crop)
        ax.add_patch(plt.Rectangle((x0 - cx0, y0 - cy0), x1 - x0, y1 - y0, fill=False, color="cyan", lw=3))
        ax.axis("off")
        words = {o["tau"]: o["word"] or "（答えなし）" for o in ex["outputs"]}
        finest = words[taus[-1]]
        for r, t in enumerate(taus):
            y = 0.35 - 0.078 * r
            word = words[t]
            color = "#C00000" if word == finest else ("gray" if word == "（答えなし）" else "#1F4E79")
            fig.text(c * col_w + 0.05, y, f"τ = {t:g}", fontsize=17, va="center", ha="left")
            fig.text(c * col_w + 0.14, y, word, fontsize=19, va="center", ha="left", color=color, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_tau_tradeoff(path, w, h):
    """
    正解の箱で τ を変えたときの出力の内訳（ex020 E1_gt，Alpha-CLIP context1.5，T = 1，test）（order_044）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    g = pd.read_csv(os.path.join(EX020, "E1_gt", "AlphaCLIP-ViT-B16_context1.5+vg_subtree", "grid_test.csv"))
    g = g[g.params.str.startswith("T=1,")].copy()
    g["tau"] = g.params.str.extract(r"tau=([0-9.]+)")[0].astype(float)
    g = g[g.tau.isin([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])].sort_values("tau")
    parts = [("finer_acc", "細かくて正しい", "#C00000"), ("rate_coarse_correct", "粗いが正しい（personなど）", "#4C72B0"),
             ("rate_wrong", "間違い", "#A0A0A0"), ("rate_none", "答えなし", "#E6E6E6")]
    plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]  # order_046：ラベルを日本語にした
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(g))
    bottom = np.zeros(len(g))
    for col, label, color in parts:
        v = g[col].to_numpy()
        ax.bar(x, v, bottom=bottom, color=color, label=label, width=0.7)
        if col == "finer_acc":
            for xi, vi in zip(x, v):
                ax.text(xi, vi / 2, f"{vi:.2f}", ha="center", va="center", color="white", fontsize=15)
        bottom += v
    ax.set_xticks(x, [f"{t:g}" for t in g.tau])
    ax.set_xlabel("つまみ τ")
    ax.set_ylabel("物の割合")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _style()
    return path


# ===================== order_050 の説明用の図 =====================

OWL_EXAMPLE = 381639                       # OWLv2 の説明の例（コーン，ドレス，人形が80種類以外，order_054）
VG_EXAMPLE = 32081                                     # VG と COCO val2017 で重複する写真（テニスの選手）
VG_SHOW = ["tennis player", "tennis racket", "tennis hat", "tennis shoe", "court", "tennis net"]
NAMING_EXAMPLE = (69356, 1)                            # 前回と今回の決め方を比べる物体（果物売り）
NAMING_EXTRA = [(80659, 5), (577932, 4)]               # 実際の例（騎手，自転車に乗る人）
CLASSIFIER_EXAMPLES = [(426203, 1), (447465, 3), (579655, 0)]   # CLIP は誤り，Alpha-CLIP は正しい物体


def _jp():
    """図の文字を日本語のフォントにする（英数字も同じフォントで表示する）．"""
    plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]


def _coco_image(image_id):
    """
    COCO val2017 の画像を読み込む．

    Args:
        image_id (int): 画像ID．

    Returns:
        PIL.Image.Image: RGB の画像．
    """
    from PIL import Image
    return Image.open(os.path.join(ROOT, "datasets", "coco", "val2017", f"{image_id:012d}.jpg")).convert("RGB")


def _square_window(image, box, scale):
    """
    ボックスの長辺の scale 倍の正方形の窓で画像を切り出す（画像の外は黒）．

    Args:
        image (PIL.Image.Image): 画像．
        box (list[float]): xyxy のボックス．
        scale (float): 窓の一辺とボックスの長辺の比（1 は切り抜き）．

    Returns:
        tuple[PIL.Image.Image, tuple[float, float]]: 窓の画像と，窓の左上の画像上の座標．
    """
    from PIL import Image
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = cx - side / 2, cy - side / 2
    canvas = Image.new("RGB", (int(side), int(side)))
    canvas.paste(image.crop((int(left), int(top), int(left + side), int(top + side))), (0, 0))
    return canvas, (left, top)


def fig_vg_example(path, w, h):
    """
    同じ写真の COCO の注釈（80種類）と Visual Genome の注釈（自由な名前）を並べる（order_050）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    import json
    _jp()
    vg = json.load(open(os.path.join(MEDIA_DIR, "vg_32081.json")))
    image = _coco_image(VG_EXAMPLE)
    boxes, labels = C19_gt(VG_EXAMPLE)
    sx, sy = image.width / vg["w"], image.height / vg["h"]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    for ax, title in zip(axes, ["COCO：決められた80種類", "Visual Genome：注釈者が自由に名前を付ける"]):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=18)
    for (x0, y0, x1, y1), name in zip(boxes, labels):
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", lw=3))
        axes[0].text(x0, y0, name, fontsize=15, va="bottom", bbox={"facecolor": "cyan", "alpha": 0.85, "pad": 1, "lw": 0})
    shown = set()
    for o in vg["objects"]:
        name = o["names"][0]
        if name not in VG_SHOW or name in shown:
            continue
        shown.add(name)
        x, y, bw, bh = o["box"]
        axes[1].add_patch(plt.Rectangle((x * sx, y * sy), bw * sx, bh * sy, fill=False, color="yellow", lw=2.5))
        inside = name in ("court", "tennis net")
        axes[1].text(x * sx + (8 if inside else 0), y * sy + (bh * sy - 8 if name == "court" else (8 if inside else 0)), name, fontsize=14,
                     va="bottom" if name != "tennis net" else "top",
                     bbox={"facecolor": "yellow", "alpha": 0.85, "pad": 1, "lw": 0})
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    _style()
    return path


def fig_owl_example(path, w, h, image_id=None):
    """
    同じ写真で，DETR と OWLv2 が LVIS の COCO 外の物体を囲めたかを並べる（order_054）．

    左は DETR（閾値0.5）の箱（水色）と COCO 外の物体（囲めなかった物は白い点線），
    右は OWLv2（物体らしさの上位50個）が囲めた COCO 外の物体（橙）．例は slides/owl_example_2026C.py の出力から選ぶ．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．
        image_id (int | None): 画像ID（None は OWL_EXAMPLE）．

    Returns:
        str: 保存先．
    """
    import json
    _jp()
    image_id = image_id or OWL_EXAMPLE
    r = {e["image_id"]: e for e in json.load(open(os.path.join(MEDIA_DIR, "owl_examples.json")))}[image_id]
    image = _coco_image(image_id)
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    n = r["n_noncoco"]
    titles = [f"DETR：{n - r['n_owl_only']}個 / {n}個", f"OWLv2：{r['n_owl_only']}個 / {n}個"]
    for ax, title in zip(axes, titles):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=20)
    for x0, y0, x1, y1 in r["detr_boxes"]:
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", lw=2))
    for x0, y0, x1, y1 in r["owl_only_boxes"]:
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=2, ls="--"))
        axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="orange", lw=3))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    _style()
    return path


def C19_gt(image_id):
    """
    COCO の正解（ボックスとクラス名）を返す．

    Args:
        image_id (int): 画像ID．

    Returns:
        tuple[np.ndarray, list[str]]: ボックス（G×4，xyxy）とクラス名．
    """
    sys.path.insert(0, os.path.join(ROOT, "programs", "ex020_coco_owl_alphaclip_granularity"))
    import train as E20  # noqa: E402
    if not hasattr(C19_gt, "coco"):
        C19_gt.coco = E20.E19_DATA.load_coco()
    return E20.E19_DATA.gt_instances(C19_gt.coco, image_id)


def _example(kind, image_id, g):
    """
    explain_examples.json から1つの例を取り出す．

    Args:
        kind (str): "naming" または "classifier"．
        image_id (int): 画像ID．
        g (int): 正解のインデックス．

    Returns:
        dict: 例．
    """
    import json
    if not hasattr(_example, "data"):
        _example.data = json.load(open(os.path.join(MEDIA_DIR, "explain_examples.json")))
    return next(e for e in _example.data[kind] if e["image_id"] == image_id and e["gt_index"] == g)


def fig_crop(path, image_id, g, scale=1.6, size=4.0):
    """
    正解の箱の周りを切り出し，箱を描いた画像を保存する（order_050）．

    Args:
        path (str): 保存先．
        image_id (int): 画像ID．
        g (int): 正解のインデックス．
        scale (float): 窓の一辺とボックスの長辺の比．
        size (float): 図の一辺（インチ）．

    Returns:
        str: 保存先．
    """
    e = _example("naming", image_id, g)
    image = _coco_image(image_id)
    x0, y0, x1, y1 = e["box"]
    side = max(x1 - x0, y1 - y0) * scale / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = max(0, cx - side), max(0, cy - side)
    window = image.crop((int(left), int(top), int(min(image.width, cx + side)), int(min(image.height, cy + side))))
    fig, ax = plt.subplots(figsize=(size, size * window.height / window.width))
    ax.imshow(window)
    x0, y0, x1, y1 = e["box"]
    ax.add_patch(plt.Rectangle((x0 - left, y0 - top), x1 - x0, y1 - y0, fill=False, color="cyan", lw=4))
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_naming_examples(path, w, h, examples=NAMING_EXTRA):
    """
    実際の物体に対する前回（1段ずつ）と今回（枝ごとの合計）の出力の経路を並べる（order_050）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．
        examples (list[tuple[int, int]]): (画像ID, 正解のインデックス)．

    Returns:
        str: 保存先．
    """
    _jp()
    fig = plt.figure(figsize=(w, h))
    n = len(examples)
    for r, (image_id, g) in enumerate(examples):
        e = _example("naming", image_id, g)
        top = 1 - r / n
        window, (left, tp) = _square_window(_coco_image(image_id), e["box"], 1.6)
        ax = fig.add_axes([0.0, top - 1 / n + 0.02, 0.2, 1 / n - 0.04])
        ax.imshow(window)
        x0, y0, x1, y1 = e["box"]
        ax.add_patch(plt.Rectangle((x0 - left, y0 - tp), x1 - x0, y1 - y0, fill=False, color="cyan", lw=3))
        ax.axis("off")
        old = [f'{o["word"]} {o["score"] * 100:.0f}%' for o in e["old"][:3]]
        new = [f'{p["word"]} {p["mass"] * 100:.0f}%' for p in e["new"]]
        for k, (label, words, ok, color) in enumerate([("前回", old, e["old"][min(2, len(e["old"]) - 1)]["correct"], "#7F7F7F"),
                                                        ("今回", new, e["new"][-1]["correct"], "#1F4E79")]):
            y = top - (0.32 + 0.36 * k) / n
            fig.text(0.22, y, label, fontsize=17, va="center", fontweight="bold", color=color)
            fig.text(0.29, y, "  →  ".join(words), fontsize=16, va="center", color=color)
            fig.text(0.97, y, "○" if ok else "×", fontsize=22, va="center", ha="right", color="#C00000" if ok else "#7F7F7F",
                     fontweight="bold")
        fig.text(0.22, top - 0.08 / n, f'正解：{e["fine"].split(".")[0].replace("_", " ")}', fontsize=14, va="center")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _style()
    return path


def fig_classifier_examples(path, w, h, examples=CLASSIFIER_EXAMPLES):
    """
    CLIP（切り抜き）と Alpha-CLIP（周り1.5倍＋マスク）が見る画像と，上位3件の予測を並べる（order_050）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．
        examples (list[tuple[int, int]]): (画像ID, 正解のインデックス)．

    Returns:
        str: 保存先．
    """
    import numpy as np
    from PIL import Image
    _jp()
    fig, axes = plt.subplots(2, len(examples) * 2, figsize=(w, h), gridspec_kw={"height_ratios": [3, 1.3]})
    for c, (image_id, g) in enumerate(examples):
        e = _example("classifier", image_id, g)
        image = _coco_image(image_id)
        x0, y0, x1, y1 = e["box"]
        crop, _ = _square_window(image, e["box"], 1.0)
        win, (left, top) = _square_window(image, e["box"], 1.5)
        arr = np.asarray(win).astype(float)
        mask = np.zeros(arr.shape[:2], bool)
        mask[int(y0 - top):int(y1 - top), int(x0 - left):int(x1 - left)] = True
        arr[~mask] = arr[~mask] * 0.45
        shown = Image.fromarray(arr.astype(np.uint8))
        for k, (img, name, key) in enumerate([(crop, "CLIP：切り抜き", "CLIP-ViT-B32"),
                                              (shown, "Alpha-CLIP：周り＋マスク", "AlphaCLIP-ViT-B16_context1.5")]):
            ax = axes[0, 2 * c + k]
            ax.imshow(img)
            if k == 1:
                ax.add_patch(plt.Rectangle((x0 - left, y0 - top), x1 - x0, y1 - y0, fill=False, color="white", lw=2))
            ax.set_title(name, fontsize=13)
            ax.axis("off")
            tx = axes[1, 2 * c + k]
            tx.axis("off")
            for r, (word, p) in enumerate(e[key]):
                ok = word == e["class"]
                tx.text(0.02, 0.85 - 0.33 * r, f"{word}  {p * 100:.0f}%", fontsize=13, transform=tx.transAxes,
                        color="#C00000" if ok else "black", fontweight="bold" if r == 0 else "normal")
        fig.text((2 * c + 1) / (2 * len(examples)), 0.005, f'正解：{e["class"]}', ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _style()
    return path


def fig_alpha_input(path, w, h, example=CLASSIFIER_EXAMPLES[0]):
    """
    Alpha-CLIP の入力（RGB の画像と，注目する領域を白で示すαチャネル）を並べる（order_050）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．
        example (tuple[int, int]): (画像ID, 正解のインデックス)．

    Returns:
        str: 保存先．
    """
    import numpy as np
    _jp()
    e = _example("classifier", *example)
    win, (left, top) = _square_window(_coco_image(example[0]), e["box"], 1.5)
    x0, y0, x1, y1 = e["box"]
    alpha = np.zeros((win.height, win.width))
    alpha[int(y0 - top):int(y1 - top), int(x0 - left):int(x1 - left)] = 1
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    axes[0].imshow(win)
    axes[0].set_title("RGB（3チャネル）", fontsize=16)
    axes[1].imshow(alpha, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("α（4つ目：白＝主役）", fontsize=16)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _style()
    return path


def make_media():
    """
    スライド用の図を作成する．

    Returns:
        dict: 図の名前 → パス．
    """
    os.makedirs(MEDIA_DIR, exist_ok=True)
    _style()
    return {"tau_examples": fig_tau_examples(os.path.join(MEDIA_DIR, "tau_examples.png"), 12.4, 5.3),
            "tau_tradeoff": fig_tau_tradeoff(os.path.join(MEDIA_DIR, "tau_tradeoff.png"), 7.6, 5.5),
            "qualitative": os.path.join(PAPER_FIG, "fig_qualitative_slide.png"),
            "owl_example": fig_owl_example(os.path.join(MEDIA_DIR, "owl_example.png"), 7.2, 4.6),
            "vg_example": fig_vg_example(os.path.join(MEDIA_DIR, "vg_example.png"), 12.4, 4.9),
            "naming_crop": fig_crop(os.path.join(MEDIA_DIR, "naming_crop.png"), *NAMING_EXAMPLE),
            "naming_examples": fig_naming_examples(os.path.join(MEDIA_DIR, "naming_examples.png"), 12.4, 4.6),
            "classifier_examples": fig_classifier_examples(os.path.join(MEDIA_DIR, "classifier_examples.png"), 12.4, 4.6),
            "alpha_input": fig_alpha_input(os.path.join(MEDIA_DIR, "alpha_input.png"), 5.6, 3.0)}


# ===================== スライドの内容 =====================

def equation(s, y, h, latex, number, size=2800):
    """
    数式と右側の式番号を配置する．

    Args:
        s (Slide): スライド．
        y, h (float): 位置と高さ（インチ）．
        latex (str): 数式．
        number (int): 式番号．
        size (int): 文字サイズ．

    Returns:
        None
    """
    s.text(0.8, y, 11.0, h, [math_para(latex, size)], anchor="ctr")
    s.text(11.9, y, 1.2, h, [para(run(f"({number})", 2800), align="r")], anchor="ctr")


#: 表で用いる記号の簡単な説明（order_042）．論文の表 tab_metrics と式(hier)の定義に合わせる．
DEFS = {
    "N": "1枚あたりの残した箱の数",
    "N_det": "1枚あたりの箱の数",
    "Top-1": "正しく囲んだ物体のうち，ラベルがCOCOのクラスと一致した割合",
    "PRF": "P：残した箱のうち正しい物体を正しいラベルで囲んだ割合，R：GTの物体のうち囲めた割合，F：PとRの調和平均",
    "F": "COCOの80クラスで評価したF値（適合率と再現率の調和平均）",
    "hF": "細かい正解（例：tennis player）に対するF値．木の上位語の重なりで測るため，粗い語（person）でも部分点",
    "Acc": "命名の正解率：正解と同じ語，より細かい正しい語，正しい上位語のいずれかを出した割合",
    "Fine": "COCOのクラスより細かい語を正しく出した割合（例：personではなくtennis player）",
    "R_N_det": "COCO外の物体（LVISの注釈，2687件）を検出器の箱で囲めた割合",
    "R_N_kept": "COCO外の物体（LVISの注釈，2687件）を，残した箱で囲めた割合",
    "R_all": "COCOの物体のうち，IoU 0.5以上の箱で囲めた割合（ラベルは問わない）",
}


def legend(s, y, items, x=0.3, w=12.7, size=1600):
    """
    表の下に，表で用いる記号の簡単な説明（表の見方）を灰色の枠で追加する（order_042）．

    Args:
        s (Slide): 追加先のスライド．
        y (float): 枠の上端（インチ）．
        items (list[tuple[str, str]]): (記号, 説明) の組．$ で囲んだ記号はLaTeXの数式として扱う．記号が空なら前の行の続き．
        x, w (float): 枠の左端と幅（インチ）．
        size (int): 文字サイズ（1/100ポイント）．

    Returns:
        float: 枠の下端（インチ）．
    """
    paras = []
    for sym, desc in items:
        if not sym:
            paras.append(para(run("　" + desc, size), size=size))
            continue
        head = math_run(sym.strip("$"), size) if sym.startswith("$") else run(sym, size, True)
        paras.append(para(head, run("：" + desc, size), size=size))
    h = 0.27 * len(items) * size / 1600 + 0.22
    s.box(x, y, w, h, paras, fill="F2F2F2", line="none", geom="rect")
    return y + h


def title_slide():
    """
    タイトルのスライド．

    Returns:
        Slide: スライド．
    """
    s = Slide("", layout=1)
    s.shapes.append(
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="タイトル 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="ctrTitle"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="125506" y="772319"/><a:ext cx="11940525" cy="2656681"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/>'
        '<a:t>学び直さずに，名前の細かさを変えられる</a:t></a:r><a:br><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/></a:br>'
        '<a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/><a:t>物体アノテーション：OWLv2，Alpha-CLIPと語彙の木</a:t></a:r></a:p></p:txBody></p:sp>')
    s.shapes.append(
        '<p:sp><p:nvSpPr><p:cNvPr id="4" name="字幕 3"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="subTitle" idx="1"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="125506" y="4339531"/><a:ext cx="11940524" cy="1210860"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>所属機関名</a:t></a:r></a:p>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>○著者名</a:t></a:r></a:p></p:txBody></p:sp>')
    return s


def pipeline_row(s, y, steps, fill):
    """
    処理の流れの箱を横に並べる．

    Args:
        s (Slide): スライド．
        y (float): 上端（インチ）．
        steps (list[str]): 箱の文字．
        fill (str): 塗りの色．

    Returns:
        None
    """
    w, gap = 2.75, 0.45
    for i, t in enumerate(steps):
        x = 0.5 + i * (w + gap)
        s.box(x, y, w, 0.9, [para(run(t, 2400), align="ctr")], fill=fill, line="accent1", anchor="ctr")
        if i < len(steps) - 1:
            s.arrow(x + w + 0.03, y + 0.45, x + w + gap - 0.03, y + 0.45)


def build(media):
    """
    全スライドを定義する（order_039：検出器と分類器の比較を加えた構成）．

    Args:
        media (dict): 図の名前 → パス．

    Returns:
        list[Slide]: スライド．
    """
    df = summary()
    e3 = e3_summary()
    det = detector_table()
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    slides = [title_slide()]

    # order_046：発表の台本（order_045）に合わせ，専門用語を言い換えて文と表を簡単にした．
    pct = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    old = df.loc[OLD]
    new = e3.loc["OWLv2@50_Alpha-crop_proposed"]
    dets = [("YOLO11x", "YOLO11x"), ("DETR", "DETR"), ("OWLv2@50", "OWLv2")]

    # 2. 背景
    s = Slide("背景：物を見つけるAIは，80種類の名前しか言えない")
    s.text(0.3, 1.0, 12.7, 2.9, [
        bullet("YOLO[1]やDETR[7]は，学習した80種類（COCO[6]）の名前だけを出す"),
        note("人は全部person，鳥は全部bird"),
        bullet("細かい名前（skateboarder，owl）には，学び直しが必要"),
        bullet("目的：学び直さずに，名前の細かさを変えたい", color=RED),
    ])
    s.box(1.3, 4.15, 10.7, 1.5, [
        para(run("person", 2800, True), run("　→　", 2800), run("skater", 2800, True), run("　→　", 2800),
             run("skateboarder", 2800, True), align="ctr"),
        para(run("自信がなければ，粗い名前で止める", 2400), align="ctr"),
    ], fill="DDEBF7", line="accent1", anchor="ctr")
    s.refs(1, 6, 7)
    slides.append(s)

    # 3. 前の研究
    s = Slide("前の研究：DETR＋CLIP＋辞書（WordNet）")
    pipeline_row(s, 1.2, ["DETRで見つける", "箱で切り抜く", "CLIPで名前", "辞書で細かく"], "E2EFDA")
    s.text(0.3, 2.5, 12.7, 4.0, [
        bullet("問題1：80種類以外の物を見落とす"),
        bullet("問題2：切り抜くと，周りの様子が分からない"),
        bullet("問題3：辞書のpersonの下は398語"),
        note("正しい語をほとんど選べない（正解率0.6%）", color=RED),
    ])
    s.refs(5, 7, 8)
    slides.append(s)

    # 4. 取り替えた部品
    s = Slide("今回：4つの部品を取り替えた")
    s.table(0.3, 1.0, [2.5, 2.4, 3.0, 4.8], [
        ["部品", "前の研究", "今回", "理由"],
        ["見つける係", "DETR", "OWLv2[2]", f"80種類以外も見つける（{pct(det.loc['DETR', 'noncoco_recall'])}→{pct(det.loc['OWLv2@50', 'noncoco_recall'])}）"],
        ["名前を当てる係", "CLIP", "Alpha-CLIP[3]", "見る場所をマスクで指定できる"],
        ["辞書", "WordNet全部", "WordNet∩VG[4]", "写真によく出る語だけ（398→136語）"],
        ["細かさの決め方", "1段ずつ下る", "枝ごとに足す", "総合点 0.27 → 0.64"],
    ], size=1800, row_h=0.6, highlight=[(1, 3), (2, 3), (3, 3), (4, 3)])
    s.text(0.3, 4.2, 12.7, 1.6, [
        bullet("どの部品も，学び直さずに使う", size=2400),
        note("総合点（hF）：細かい正解にどれだけ近いか．粗い名前でも部分点", size=1800),
        note("VG：Visual Genome．写真に写った物の名前を集めたデータ", size=1800),
    ])
    s.refs(2, 3, 4)
    slides.append(s)

    # 5. 取り替えた効果
    s = Slide("効果：80種類以外も見つかり，細かい名前が倍に")
    s.table(1.3, 1.05, [6.0, 2.2, 2.2], [
        ["指標（写真500枚）", "前の研究", "今回"],
        ["80種類以外の物を見つけた割合", pct(det.loc["DETR", "noncoco_recall"]), pct(det.loc["OWLv2@50", "noncoco_recall"])],
        ["細かくて正しい名前の割合", pct(old.sub_finer_acc), pct(new.sub_finer_acc)],
        ["80種類の名前の点数（F値）", f2(old.f1), f2(new.f1)],
        ["総合点（hF）", f2(old.sub_hF), f2(new.sub_hF)],
    ], size=2000, row_h=0.6, highlight=[(1, 2), (2, 2), (3, 2)])
    s.text(0.3, 4.3, 12.7, 2.2, [
        bullet("見つける力と，細かい名前は良くなった", size=2400),
        bullet("ただし，総合点（hF）はほぼ同じ", size=2400, color=RED),
    ])
    legend(s, 5.75, [("F値", "正しい物を，80種類の正しい名前で囲めたかの点数（0〜1）．余計な箱や見逃しが多いと下がる"),
                     ("総合点（hF）", "細かい正解（例：skateboarder）にどれだけ近いか．粗い名前（person）でも部分点")], size=1500)
    slides.append(s)

    # 6. 提案モデル
    s = Slide("提案モデル：見つける→名前の確率→枝ごとに足す")
    pipeline_row(s, 1.2, ["OWLv2で見つける", "Alpha-CLIP", "枝ごとに足す", "つまみτで選ぶ"], "FFF2CC")
    s.text(0.3, 2.5, 12.7, 3.6, [
        bullet("見つける係：OWLv2[2]（何でも見つける）"),
        bullet("名前を当てる係：Alpha-CLIP[3]（見る場所を指定できる）"),
        bullet("辞書：写真によく出る3263語の木（WordNet∩VG[4]）"),
        bullet("どれも学び直さない", color=RED),
    ])
    s.refs(2, 3, 4)
    slides.append(s)

    # 6a. OWLv2 とは（order_054）
    s = Slide("OWLv2とは：名前を決めずに，物らしい所を囲む")
    steps = ["写真を小さなマスに分ける", "マスごとに箱と物らしさ", "物らしさの高い50個を使う"]
    for k, t in enumerate(steps):
        x = 0.4 + k * 4.3
        s.box(x, 1.0, 3.8, 0.75, [para(run(t, 1800, True), align="ctr")], fill="FFF2CC", line="accent1", anchor="ctr")
        if k < 2:
            s.arrow(x + 3.83, 1.375, x + 4.27, 1.375)
    w, h = s.image(media["owl_example"], 0.4, 1.9, h=3.75)
    s.text(0.3, 1.9 + h, 12.7, 0.45, [para(run("数字：80種類以外の物（コーン，ドレス，人形）18個のうち囲めた数．水色：DETRの箱，白い点線：DETRが見逃した物", 1400))])
    s.text(0.4 + w + 0.2, 1.95, 12.9 - (0.4 + w + 0.2), 3.9, [
        bullet("Googleの物体検出AI（2023年）", size=1800, space_before=0),
        bullet("Webの大量の写真と説明文で学習", size=1800, space_before=300),
        note("→ 80種類に限らず見つける", size=1700),
        bullet("本来は言葉（catなど）で探す", size=1800, space_before=300),
        note("今回は言葉を使わず，物らしさだけで囲む", size=1700),
        bullet("名前はAlpha-CLIPが付ける", size=1800, space_before=300),
    ])
    s.text(0.3, 6.15, 12.7, 0.6, [para(run("この写真のコーン，ドレス，人形（80種類以外）は，OWLv2だけが囲めた", 2200, True, RED))])
    s.refs(2)
    slides.append(s)

    # 6b. Visual Genome とは（order_050）
    s = Slide("VGとは：写真の物に自由に名前を付けたデータ")
    w, h = s.image(media["vg_example"], 1.2, 0.9, w=10.9)
    s.text(0.3, 0.9 + h, 12.7, 6.9 - 0.9 - h, [
        bullet("約10万枚の写真に，物の名前を約290万件．名前はWordNetの語に対応付け済み", size=2000, space_before=0),
        bullet("目的：写真で実際によく使われる名前を知る → 辞書を3263語に絞る", size=2000, space_before=200),
        note("10回以上出てくる語だけを使う．名前は注釈者が一番多く使った言い方（例：tennis player）", size=1700),
    ])
    slides.append(s)

    # 7. 細かさの決め方（式の代わりに木の例で示す．order_051：左に前回の下位語の探し方を加えた）
    s = Slide("前回は1段ずつ1語を選び，今回は枝ごとに足して選ぶ")
    gray, blue = ("F2F2F2", "A6A6A6"), ("DDEBF7", "accent1")
    # order_053：赤い枠＝前回のモデルが選んだ語．「1段目・2段目」を①②の言い方に変え，②の候補を枠で囲んだ．
    s.text(0.3, 0.95, 6.1, 0.5, [para(run("前回：上から1段ずつ，一番高い語を選ぶ", 2000, True), align="ctr")])
    s.box(1.4, 1.5, 3.9, 0.8, [para(run("① 80種類の名前から選ぶ", 1500), align="ctr"), para(run("person", 2000, True), align="ctr")],
          fill="FFFFFF", line="C00000", anchor="ctr", line_w=38100)
    s.arrow(3.35, 2.3, 3.35, 2.6)
    s.box(0.3, 2.6, 6.1, 1.45, [para(run("② personのすぐ下の語から選ぶ", 1500), align="ctr")], fill="FFFFFF", line=gray[1], anchor="t")
    s.box(0.45, 3.1, 1.8, 0.75, [para(run("skater　7%", 1800), align="ctr")], fill=gray[0], line=gray[1], anchor="ctr")
    s.box(2.45, 3.1, 1.9, 0.75, [para(run("gatherer　9%", 1800, True, RED), align="ctr")], fill="FFFFFF", line="C00000", anchor="ctr",
          line_w=38100)
    s.box(4.55, 3.1, 1.7, 0.75, [para(run("ほかの語", 1800), align="ctr")], fill=gray[0], line=gray[1], anchor="ctr")
    s.arrow(1.35, 3.85, 1.35, 4.3)
    s.box(0.3, 4.3, 2.1, 0.65, [para(run("skateboarder 8%", 1500), align="ctr")], fill=gray[0], line=gray[1], anchor="ctr")
    s.text(2.55, 4.2, 3.9, 1.0, [
        para(run("skateboarderはskaterの下なので，", 1500)),
        para(run("②では比べない", 1500)),
        para(run("→ 9%のgathererを選んで間違い", 1500, True, RED)),
    ])
    s.text(6.9, 0.95, 6.1, 0.5, [para(run("今回：枝ごとに確率を足して選ぶ", 2000, True), align="ctr")])
    s.box(6.9, 1.55, 3.1, 0.75, [para(run("person　合計50%", 1900, True), align="ctr")], fill=blue[0], line=blue[1], anchor="ctr")
    s.box(6.9, 2.95, 3.1, 0.75, [para(run("skater　合計15%", 1900, True), align="ctr")], fill=blue[0], line=blue[1], anchor="ctr")
    s.box(6.9, 4.2, 3.1, 0.75, [para(run("skateboarder　8%", 1900, True), align="ctr")], fill=blue[0], line=blue[1], anchor="ctr")
    s.arrow(8.45, 2.3, 8.45, 2.95)
    s.arrow(8.45, 3.7, 8.45, 4.2)
    s.box(10.3, 1.55, 2.7, 3.4, [
        para(run("合計がτ以上で", 1700, True), align="ctr"),
        para(run("一番細かい名前", 1700, True), align="ctr"),
        para(run("τ = 0.3 → person", 1800), align="ctr", space_before=700),
        para(run("τ = 0.1 → skater", 1800), align="ctr", space_before=500),
        para(run("τ = 0.05 →", 1800, True, RED), align="ctr", space_before=500),
        para(run("skateboarder", 1800, True, RED), align="ctr"),
    ], fill="FFFFFF", line="accent1", anchor="t")
    s.text(6.9, 4.95, 6.1, 0.8, [para(run("skaterの枝15% ＝ skater 7% ＋ skateboarder 8%", 1600)),
                                para(run("personの枝50% ＝ skaterの枝15% ＋ ほかの人の語35%", 1600))])
    s.text(0.3, 5.8, 12.7, 1.3, [
        bullet("前回：下の語の確率を数えない．候補が多いと当てずっぽうになり，間違えると戻れない", size=2000, space_before=0),
        bullet("今回：τ＝細かさのつまみ．高いと粗い名前，低いと細かい名前．学び直さない", size=2000, color=RED, space_before=200),
        note("赤い枠：前回のモデルが選んだ語．数値は説明のための例（左右で同じ確率）", size=1500),
    ])
    slides.append(s)

    # 7b. 前回と今回の決め方の違い（同じ物体の実例，order_050）
    s = Slide("前回と今回の決め方の違い：同じ物体で比べる")
    iw, ih = s.image(media["naming_crop"], 0.6, 1.0, h=3.3)
    s.text(0.2, 1.0 + ih + 0.02, 3.6, 0.5, [para(run("正解：seller（売り手）", 1800, True), align="ctr")])
    s.text(3.9, 0.95, 4.4, 0.5, [para(run("前回：1段ずつ選ぶ", 2000, True), align="ctr")])
    s.box(3.9, 1.55, 4.4, 1.0, [para(run("1段目：80語から選ぶ", 1800), align="ctr"), para(run("person　29%", 2200, True), align="ctr")],
          fill="F2F2F2", line="A6A6A6", anchor="ctr")
    s.arrow(6.1, 2.55, 6.1, 3.0)
    s.box(3.9, 3.0, 4.4, 1.0, [para(run("2段目：personの下398語から選ぶ", 1800), align="ctr"),
                               para(run("gatherer　5%　×", 2200, True), align="ctr")], fill="F2F2F2", line="A6A6A6", anchor="ctr")
    s.text(3.9, 4.05, 4.4, 0.9, [para(run("398語に確率が散らばり，", 1600), align="ctr"), para(run("5%でも1位になってしまう", 1600), align="ctr")])
    s.text(8.7, 0.95, 4.4, 0.5, [para(run("今回：枝ごとに足して選ぶ", 2000, True), align="ctr")])
    s.box(8.7, 1.55, 4.4, 1.0, [para(run("personの枝の合計", 1800), align="ctr"), para(run("76%", 2200, True), align="ctr")],
          fill="DDEBF7", line="accent1", anchor="ctr")
    s.arrow(10.9, 2.55, 10.9, 3.0)
    s.box(8.7, 3.0, 4.4, 1.0, [para(run("vendor（売り手）の枝の合計", 1800), align="ctr"),
                               para(run("69%　○", 2200, True, RED), align="ctr")], fill="DDEBF7", line="accent1", anchor="ctr")
    s.text(8.7, 4.05, 4.4, 0.9, [para(run("似た語の確率がまとまり，", 1600), align="ctr"),
                                 para(run("τ以上で一番細かいvendorを選ぶ", 1600), align="ctr")])
    s.text(0.3, 5.15, 12.7, 1.7, [
        bullet("前回：各段で1語ずつ決める．候補が多いと当てずっぽうになり，間違えると戻れない", size=2000, space_before=0),
        bullet("今回：全ての語の確率を枝ごとに足してから，自信のある一番深い語を選ぶ", size=2000, space_before=200, color=RED),
        note("名前を当てる係はどちらもAlpha-CLIP（決め方だけを比べた）", size=1600),
    ])
    slides.append(s)

    # 7c. 実際の例（order_050）
    s = Slide("前回は最初の枝を間違えると戻れない")
    w, h = s.image(media["naming_examples"], 0.45, 1.0, w=12.4)
    s.text(0.3, 1.0 + h + 0.05, 12.7, 1.2, [
        para(run("位置は正解の箱，名前を当てる係はどちらもAlpha-CLIP．数値は前回は各段の確率，今回は枝の合計", 1600)),
        para(run("前回は箱に一緒に写った馬や自転車を選び，その下の語へ進んでしまう", 2200, True, RED)),
    ])
    slides.append(s)

    # 8. 実験条件
    s = Slide("実験：COCOの写真500枚で試した")
    s.text(0.3, 1.0, 12.7, 5.6, [
        bullet("写真：COCO 2017（調整に500枚，評価に500枚）", size=2600),
        bullet("細かい正解：写真の説明文に出る言葉（912件）", size=2600),
        note("例：skateboarder，tennis player．85%は人", size=2200),
        bullet("80種類以外の物の正解：LVIS[9]の注釈（2687件）", size=2600),
        bullet("つまみ：τ = 0.05（調整用の写真で決めた）", size=2600),
    ])
    s.refs(9)
    slides.append(s)

    # 9. τ の例
    s = Slide("つまみτを下げると，同じ物の名前が細かくなる")
    w, h = s.image(media["tau_examples"], 0.45, 0.95, w=12.4)
    s.text(0.3, 0.95 + h + 0.05, 12.7, 1.0, [
        para(run("物の位置は正解を使用．赤：一番細かい名前．きれいに細かくなる例を選んだ", 1800)),
        para(run("つまみ1つで，学び直さずに細かさが変わる", 2400, True, RED)),
    ])
    slides.append(s)

    # 10. 細かさと正しさ
    s = Slide("細かくすると，間違いも増える")
    s.image(media["tau_tradeoff"], 0.2, 1.0, w=7.6)
    s.text(7.9, 1.05, 5.2, 5.6, [
        bullet("τを下げる", size=2400),
        note("細かい名前も，間違いも増える", size=2000),
        bullet("τを上げる", size=2400),
        note("粗い名前や答えなしが増える", size=2000),
        bullet("細かくて正しい名前は最大15%", size=2400, color=RED),
        note("ここが今後の課題", size=2000),
    ])
    slides.append(s)

    # 11. 枝ごとに足す効果
    s = Slide("枝ごとに足すと，総合点が0.27から0.64に上がる")
    s.table(1.3, 1.05, [6.0, 2.2, 2.2], [
        ["名前の決め方（位置は正解）", "総合点（hF）", "正解率"],
        ["80種類の名前だけ", "0.27", "31%"],
        ["辞書を1段ずつ下る（前の研究）", "0.27", "30%"],
        ["枝ごとに足す（今回）", "0.64", "78%"],
    ], size=2000, row_h=0.6, highlight=[(3, 1), (3, 2)])
    s.text(0.3, 3.7, 12.7, 2.9, [
        bullet("自信がないときは，無理に細かくせずpersonで止まる", size=2400),
        note("物の位置は正解を使い，名前の付け方だけを比べた", size=1800),
        note("正解率：正しい名前（粗くても可）を出した割合", size=1800),
    ])
    slides.append(s)

    # 12. 見つける係
    rows = [["見つける係", "80種類の物", "80種類以外の物"]]
    for key, label in dets:
        rows.append([label, pct(det.loc[key, "recall"]), pct(det.loc[key, "noncoco_recall"])])
    s = Slide("80種類以外の物を見つけられるのは，OWLv2だけ")
    s.table(1.8, 1.05, [3.4, 2.8, 3.2], rows, size=2000, row_h=0.6, highlight=[(3, 1), (3, 2)])
    s.text(0.3, 3.7, 12.7, 2.9, [
        bullet("見つけられない物には，名前を付けられない", size=2400),
        bullet("→ 見つける係にOWLv2を選んだ", size=2400, color=RED),
        note("表の値：正解の物のうち，箱で囲めた割合（80種類以外はLVISの注釈）", size=1800),
        note("ただしOWLv2は余計な箱も多い（87%は物に当たらない）", size=1800),
    ])
    slides.append(s)

    # 13. 弱点
    rows = [["見つける係（名前はAlpha-CLIP）", "80種類の点数（F値）", "細かくて正しい"]]
    for key, label in dets:
        r = e3.loc[f"{key}_Alpha-crop_proposed"]
        rows.append([label, f2(r.f1), pct(r.sub_finer_acc)])
    rows.append(["前の研究（DETR＋CLIP）", f2(old.f1), pct(old.sub_finer_acc)])
    s = Slide("弱点：80種類だけの点数では，YOLO・DETRの方が高い")
    s.table(1.3, 1.05, [5.4, 2.8, 2.4], rows, size=2000, row_h=0.6, highlight=[(1, 1)])
    s.text(0.3, 4.2, 12.7, 2.5, [
        bullet("OWLv2は，余計な物も多く囲んでしまう", size=2400),
        bullet("余計な箱を捨てるルールで，80種類以外の物も捨てる", size=2400, color=RED),
        note(f"80種類以外の物：見つけた{pct(det.loc['OWLv2@50', 'noncoco_recall'])} → 残したのは{pct(new.noncoco_recall)}．捨てるルールの改良は今後の課題",
             size=1800, color=RED),
    ])
    legend(s, 6.0, [("F値", "正しい物を，80種類の正しい名前で囲めたかの点数（0〜1）．余計な箱や見逃しが多いと下がる")], size=1500)
    slides.append(s)

    # 14. 名前を当てる係
    rows = [["見つける係", "CLIP", "Alpha-CLIP"]]
    for key, label in dets:
        rows.append([label, f2(e3.loc[f"{key}_CLIP-B32_proposed", "f1"]), f2(e3.loc[f"{key}_Alpha-crop_proposed", "f1"])])
    fine = [e3.loc[f"{d}_{emb}_proposed", "sub_finer_acc"] for emb in ("CLIP-B32", "Alpha-crop") for d, _ in dets]
    s = Slide("名前を当てる係は，Alpha-CLIPがCLIPより良い")
    s.table(2.3, 1.05, [3.4, 2.4, 2.6], rows, size=2000, row_h=0.6, highlight=[(1, 2), (2, 2), (3, 2)])
    s.text(0.3, 3.7, 12.7, 1.5, [
        bullet("どの見つける係と組み合わせても，Alpha-CLIPが高い", size=2400),
        bullet(f"細かくて正しい名前の割合も約2倍（{pct(min(fine[:3]))}前後 → {pct(min(fine[3:]))}〜{pct(max(fine[3:]))}）", size=2400),
    ])
    legend(s, 5.2, [("表の値はF値", "正しい物を，80種類の正しい名前で囲めたかの点数（0〜1，高いほど良い）"),
                    ("計算の例", "正解の物10個．箱を8個出し，6個が正しい物を正しい名前で囲んだ"),
                    ("", "→ 箱の正しさ 6/8 = 0.75，見逃さなさ 6/10 = 0.6 → F値は2つの平均（調和平均）で 0.67")], size=1600)
    slides.append(s)

    # 14b. CLIP と Alpha-CLIP の例（order_050）
    s = Slide("Alpha-CLIPは周りの様子から，物の名前を見分ける")
    w, h = s.image(media["classifier_examples"], 0.45, 0.95, w=12.4)
    s.text(0.3, 0.95 + h + 0.05, 12.7, 1.3, [
        para(run("位置は正解の箱．COCOの80種類から選んだ上位3件（赤：正解）", 1600)),
        para(run("板の周りが道路ならskateboard，雪ならskis．手と顔の間ならcell phone", 2200, True, RED)),
    ])
    slides.append(s)

    # 14c. 人と人以外での差（order_050）
    s = Slide("差が大きいのは人以外の物．人では悪くなる")
    s.table(0.6, 1.05, [5.6, 2.3, 2.1, 2.1], [
        ["名前を当てる係（位置は正解）", "人以外の物", "人", "全体"],
        ["CLIP（切り抜き）", "51%", "47%", "50%"],
        ["Alpha-CLIP（切り抜き＋マスク）", "57%", "43%", "53%"],
        ["Alpha-CLIP（周り1.5倍＋マスク）", "66%", "25%", "53%"],
    ], size=2000, row_h=0.6, highlight=[(3, 1), (1, 2)])
    s.text(0.3, 3.65, 12.7, 3.2, [
        bullet("人以外の物：周りを見ると，CLIPより15ポイント高い", size=2200),
        bullet("人：周りの持ち物の名前（frisbee，surfboardなど）を付けてしまう", size=2200),
        bullet("今回のモデルは「切り抜き＋マスク」を使った（箱の多いOWLv2で人の箱が多いため）", size=2000, color=RED),
        note("表の値：80種類で1位の予測が正しい割合（人以外2368件，人1090件）．同じ大きさのCLIPは全体48%", size=1600),
    ])
    slides.append(s)

    # 14d. Alpha-CLIP の仕組み（order_050）
    s = Slide("Alpha-CLIPが周りを見ながら注目できる理由")
    s.image(media["alpha_input"], 0.2, 1.0, w=5.0)
    s.text(5.3, 1.0, 7.9, 3.1, [
        bullet("CLIP：入力はRGBの3チャネルだけ", size=2000, space_before=0),
        note("どの物を聞いているか伝えられない → 切り抜くので，周りが消える", size=1700),
        bullet("Alpha-CLIP：4つ目の入力α（白＝主役）を追加", size=2000),
        note("周りの画素も残したまま，「ここが主役」と伝えられる", size=1700),
    ])
    s.text(0.3, 4.2, 12.7, 2.7, [
        bullet("αを読む層を加え，「マスク付き写真と説明文」の組で学習し直した[3]", size=2000),
        note("→ 白い部分を主役として説明しつつ，周りを手がかりに使うことを学んだ", size=1800),
        bullet("CLIPの知識（語の意味）はそのまま受け継ぐので，任意の語を使える", size=2000),
        bullet("ただし周りを見すぎると，人に持ち物の名前を付ける（前のスライド）", size=2000, color=RED),
    ])
    s.refs(3)
    slides.append(s)

    # 15. 実際の出力
    s = Slide("細かい名前と粗い名前を，同時に出せる")
    w, h = s.image(media["qualitative"], 0.6, 0.95, w=12.1)
    s.text(0.35, 0.95 + h, 12.6, 0.5, [
        para(run("(a)〜(c) 今回のモデル（見つける係を替えた3通り），(d) YOLO11xだけ．括弧内は粗い名前", 1800))])
    slides.append(s)

    # 16. まとめ
    s = Slide("まとめ")
    s.text(0.3, 1.0, 12.7, 5.6, [
        bullet("学び直さずに，つまみτだけで名前の細かさを変えられる", size=2600),
        note("例：person → skater → skateboarder", size=2200),
        bullet("枝ごとに確率を足すと，総合点が0.27 → 0.64", size=2600),
        bullet("80種類以外の物を見つけられるのは，OWLv2だけ", size=2600),
        bullet("課題1：細かくて正しい名前は，まだ最大15%", size=2600, color=RED),
        bullet("課題2：余計な箱を捨てるとき，80種類以外の物も捨てる", size=2600, color=RED),
    ])
    slides.append(s)
    return slides


# ===================== 発表者ノート（order_047） =====================

# 8枚目の今回の決め方の台本（order_051．学科横断発表（第二回）でも使う）
TREE_NOTE = (
    "数値は説明のための例です．\n"
    "辞書は木の形になっていて，personの下にskater，その下にskateboarderがあります．\n"
    "AIはそれぞれの言葉に，この言葉である確率を出します．\n"
    "ポイントは，枝ごとに確率を足し合わせることです．\n"
    "たとえばskateboarderが8%，skater自身が7%なら，skaterの枝の合計は15%です．その他の人の言葉が合わせて35%なら，personの枝の合計は50%になります．\n"
    "そして，合計がτ以上の中で，いちばん細かい言葉を答えにします．\n"
    "τが0.3ならperson，0.1ならskater，0.05ならskateboarderです．\n"
    "つまりτが細かさのつまみです．高いと粗い名前，低いと細かい名前になります．\n"
    "つまみを回すだけなので，学び直す必要はありません．"
)

#: 各スライドの発表者ノート（発表の台本，order_045 を order_046 のスライドに合わせて修正）．1要素が1枚，改行で段落を分ける．
NOTES = [
    # 1. 題名
    "今日は，写真に写った物を，好きな細かさで名前付けする方法についてお話しします．\n"
    "たとえば同じ人を「person」とも「skateboarder」とも呼べるようにします．\n"
    "しかも，AIを学び直させる必要がありません．",
    # 2. 背景
    "物を見つけるAI，たとえばYOLOやDETRは，学習した80種類の名前しか言えません．\n"
    "人は全部person，鳥は全部birdになります．\n"
    "skateboarderやowlと言わせたければ，データを集めて学び直させる必要があり，手間がかかります．\n"
    "そこで私たちは，学び直さずに，名前の細かさを変えたいと考えました．\n"
    "下の枠のように，自信があれば細かい名前まで進み，自信がなければ粗い名前で止めるのが理想です．",
    # 3. 前の研究
    "前の研究では，3段で名前を付けていました．\n"
    "DETRで物を見つけ，見つけた部分を切り抜き，CLIPに見せます．CLIPは画像と言葉を照らし合わせられるAIです．\n"
    "最後に，WordNetという辞書をたどって細かくします．\n"
    "ただ，問題が3つありました．\n"
    "1つ目，DETRは80種類以外の物を見落とします．\n"
    "2つ目，切り抜くと周りの様子が分からなくなります．\n"
    "3つ目，辞書のpersonの下には398語もあって，正しい語をほとんど選べません．正解率は0.6%でした．",
    # 4. 取り替えた部品
    "この3つの問題に対して，部品を4つ取り替えました．\n"
    "見つける係は，何でも見つけられるOWLv2にしました．80種類以外の物を見つける割合が12%から73%になります．\n"
    "名前を当てる係は，見る場所をマスクで指定できるAlpha-CLIPにしました．\n"
    "辞書は，写真によく出てくる言葉だけに絞りました．personの下は398語から136語になりました．\n"
    "細かさの決め方も，枝ごとに確率を足す新しいやり方にしました．これは後で説明します．\n"
    "どの部品も，学び直さずにそのまま使います．",
    # 5. 効果
    "前の研究と比べた結果です．\n"
    "80種類以外の物を見つけられる割合が12%から73%に増えました．\n"
    "正しく細かい名前を付けられた割合も，6%から11%と，ほぼ倍になりました．\n"
    "ただし正直に言うと，細かさを含めた総合点はほとんど変わっていません．\n"
    "F値は，正しい物を80種類の正しい名前で囲めたかの点数です．余計な箱が多くても，見逃しが多くても下がります．",
    # 6. 提案モデル
    "今回のモデルの流れです．\n"
    "OWLv2で物を見つけ，Alpha-CLIPで辞書の言葉ごとの確率を出し，枝ごとに確率を足して，つまみτで細かさを選びます．\n"
    "辞書は，写真によく出る3263語を木の形に並べたものです．\n"
    "どの部品も学び直しません．",
    # 6a. OWLv2 とは（order_054）
    "見つける係のOWLv2を紹介します．Googleが2023年に発表した，物を見つけるAIです．\n"
    "仕組みは上の3つの箱のとおりです．写真を小さなマスに分け，マスごとに『ここに物がありそうな箱』と『どれくらい物らしいか』を出します．"
    "物らしさの高い順に50個の箱を使います．\n"
    "YOLOやDETRは80種類の写真で学習したので，80種類の物しか探しません．OWLv2はWebの大量の写真と説明文で学習したので，80種類に限らず物を見つけられます．\n"
    "OWLv2は本来，『cat』のような言葉を入れて，その物を探すAIです．今回は言葉を使わず，物らしさだけで箱を出します．名前は次のAlpha-CLIPが付けます．\n"
    "写真の例では，コーン，ドレス，人形が80種類以外の物です．DETRは人と飛行機だけを囲み，これらを1つも囲めませんでした．OWLv2の50個の箱は，18個全てを囲みました．\n"
    "ただし，OWLv2は物でない所も囲むので，余計な箱が多くなります．これは後の弱点のスライドで説明します．",
    # 6b. Visual Genome とは（order_050）
    "辞書の元になったVisual Genome，略してVGを紹介します．\n"
    "左は，有名なCOCOというデータの注釈です．名前は決められた80種類から選ぶので，テニスの選手もpersonになります．\n"
    "右は，同じ写真のVGの注釈です．VGでは，注釈をする人が自由に名前を付けます．"
    "tennis player，tennis hat，tennis shoe，courtのように，細かい名前が付いています．\n"
    "VGには約10万枚の写真に，約290万件の名前があり，それぞれWordNetの語に対応付けられています．\n"
    "私たちはVGを使って，写真で実際によく使われる名前を調べました．10回以上出てくる語だけを残し，辞書を3263語に絞りました．"
    "また，AIに見せる名前には，注釈者が一番多く使った言い方を使いました．",
    # 7. 細かさの決め方（order_051：前回の探し方を前に加えた）
    "ここが一番大事なところです．左が前回，右が今回で，同じ木と同じ確率を使っています．\n"
    "前回は，辞書の木を上から1段ずつ下りて，その段で一番確率が高い語を選びました．赤い枠が，前回のモデルが選んだ語です．\n"
    "まず①で，80種類の名前からpersonを選びます．次に②で，personのすぐ下にある語だけを比べて，一番高い語を選びます．"
    "選んだ語の下にさらに語があれば，同じことを繰り返します．\n"
    "②で比べるのは，その語自身の確率だけです．skateboarderはskaterのさらに下にあるので，②では比べません．"
    "そのため，skater 7%より，9%のgathererが選ばれてしまいます．間違いです．一度選ぶと戻れません．\n"
    "今回は，右のように考えます．\n" + TREE_NOTE,
    # 7b. 前回と今回の決め方の違い（order_050）
    "前回と今回の決め方の違いを，同じ物体で比べます．正解は，果物を売っている人，sellerです．"
    "名前を当てる係はどちらもAlpha-CLIPにそろえて，決め方だけを比べています．\n"
    "前回は，1段ずつ一番高い語を選びました．1段目は80語からpersonを選びました．"
    "しかし2段目では，personの下の398語から1つを選ぶので，確率が散らばり，5%しかないgathererが1位になりました．間違いです．\n"
    "今回は，全ての語の確率を，枝ごとに足してから選びます．personの枝は合計76%，その下のvendor，つまり売り手の枝は合計69%です．"
    "τの0.05以上で一番細かいvendorを選び，正解しました．\n"
    "似た意味の語の確率が枝にまとまるので，自信のある深さで止まれます．",
    # 7c. 実際の例（order_050）
    "ほかの例です．\n"
    "上は競馬の騎手です．前回は1段目で，箱に一緒に写っている馬，horseを87%で選びました．"
    "一度horseを選ぶと，その下のracehorseやthoroughbredにしか進めず，戻れません．\n"
    "今回は，person，rider，horseback rider，jockeyと，正しい枝を進みました．\n"
    "下は自転車に乗る人です．前回はbicycleを選んで外れ，今回はcyclistと正しく答えました．\n"
    "このように，前回は最初の段の間違いがそのまま最後まで残りました．",
    # 8. 実験
    "COCOという有名な写真データで試しました．500枚でつまみを調整し，別の500枚で評価しました．\n"
    "細かい正解には，写真の説明文に出てくる言葉，たとえばskateboarderやtennis playerを使いました．912件あります．\n"
    "ただし，その85%は人です．\n"
    "80種類以外の物の正解には，LVISという別の注釈を使いました．\n"
    "つまみτは，調整用の写真で0.05に決めました．",
    # 9. τ の例
    "実際の例です．物の位置は正解を使い，つまみだけを変えています．\n"
    "左のスケートボードの人では，τを0.5から下げていくと，答えがperson，skater，skateboarderと細かくなります．\n"
    "牛はmammal，つまり哺乳類から，cow，そしてbull，雄牛になります．鳥はanimal，bird，owl，フクロウです．\n"
    "つまみ1つで細かさが変わる．これがこの研究で一番伝えたいことです．\n"
    "ただし，ここまできれいに変わるのは選んだ例です．全体の傾向は次のスライドで示します．",
    # 10. 細かさと正しさ
    "全体で見るとこうなります．\n"
    "赤が細かくて正しい，青が粗いけど正しい，濃い灰色が間違い，薄い灰色が答えなしです．\n"
    "τを下げると，細かい答えが増えますが，間違いも増えます．\n"
    "τを上げると，間違いは減りますが，粗い答えや答えなしが増えます．\n"
    "正直に言うと，細かくて正しい答えは最大でも15%で，まだ少ないです．ここが今後の課題です．",
    # 11. 枝ごとに足す効果
    "物の位置を正解で与えて，名前の付け方だけを比べました．\n"
    "80種類の名前だけを使う場合や，前の研究の辞書を1段ずつ下る方法では，総合点は0.27でした．\n"
    "枝ごとに足し合わせる方法では，0.64に上がりました．正解率も30%前後から78%になりました．\n"
    "細かい名前に自信がないときに，無理に間違えず，personで止まれるようになったからです．",
    # 12. 見つける係
    "次に，見つける係を比べました．\n"
    "80種類の物は，どれもある程度見つけられます．\n"
    "しかし80種類以外の物を見つけられたのは，OWLv2だけでした．73%です．YOLOとDETRは10%ほどです．\n"
    "見つけられない物には名前を付けられないので，OWLv2を選びました．\n"
    "ただし，OWLv2は余計な箱も多く出します．",
    # 13. 弱点
    "弱点もあります．\n"
    "80種類の名前だけで点数を付けると，YOLOやDETRを使った方が高くなります．OWLv2は余計な物も多く囲んでしまうからです．\n"
    "もう一つの弱点は，余計な箱を捨てる今のルールです．\n"
    "このルールで，せっかく見つけた80種類以外の物まで捨ててしまっています．73%が10%に落ちます．\n"
    "捨てるルールの改良は，今後の課題です．",
    # 14. 名前を当てる係
    "名前を当てる係の比較です．表の値はF値です．\n"
    "F値は，正しい物を80種類の正しい名前で囲めたかの点数です．たとえば正解の物が10個あり，箱を8個出して6個が正しかったとします．"
    "箱の正しさは8個中6個で0.75，見逃さなさは10個中6個で0.6です．この2つの平均をとったものがF値で，0.67になります．\n"
    "Alpha-CLIPは，どの見つける係と組み合わせても，前のCLIPより点数が高くなりました．\n"
    "細かくて正しい名前の割合も，約2倍です．",
    # 14b. CLIP と Alpha-CLIP の例（order_050）
    "では，CLIPとAlpha-CLIPは，具体的に何が違うのでしょうか．位置は正解の箱で比べました．\n"
    "左の例は，空中のスケートボードです．CLIPは切り抜いた板だけを見て，snowboardと答えました．"
    "Alpha-CLIPは周りの街並みも見て，skateboardと答えました．\n"
    "真ん中はスキー板です．板だけではsnowboardにも見えますが，Alpha-CLIPはskisと答えました．\n"
    "右は，手と顔の間にある暗い物です．切り抜くと何か分かりませんが，Alpha-CLIPは周りから，cell phoneと87%で答えました．\n"
    "このように，Alpha-CLIPは周りの様子を手がかりに，物の名前を見分けます．",
    # 14c. 人と人以外での差（order_050）
    "数字で見ると，差がはっきりするのは人以外の物です．\n"
    "人以外の物では，CLIPが51%，周りも見るAlpha-CLIPが66%で，15ポイント高くなります．\n"
    "ところが人では，周りを見るAlpha-CLIPは25%に下がります．人の周りにあるfrisbeeやsurfboardなど，持ち物の名前を付けてしまうからです．\n"
    "そのため全体では，50%と53%で，差は小さくなります．\n"
    "今回のモデルでは，OWLv2が人の箱を多く出すので，周りを見すぎないように，切り抜きとマスクの組み合わせを使いました．"
    "なお，同じ大きさのCLIPでも全体は48%なので，この差はモデルの大きさのせいではありません．",
    # 14d. Alpha-CLIP の仕組み（order_050）
    "では，なぜAlpha-CLIPは周りを見ながら，主役に注目できるのでしょうか．\n"
    "普通のCLIPの入力は，赤・緑・青の3チャネルの画像だけです．どの物の名前を聞いているかを伝える方法がないので，切り抜くしかありません．"
    "切り抜くと，周りが消えてしまいます．\n"
    "Alpha-CLIPは，4つ目の入力としてαチャネルを加えました．左の図の白い部分が主役です．周りの画素も入力に残ったまま，ここが主役だと伝えられます．\n"
    "そして，αを読む層を加えたうえで，マスク付きの写真と，その部分の説明文の組，数百万組で学習し直しています．"
    "その結果，白い部分を主役として説明しながら，周りを手がかりに使うことを学びました．\n"
    "CLIPの言葉の知識はそのまま受け継ぐので，どんな語でも使えます．ただし，周りを見すぎると，前のスライドのように，人に持ち物の名前を付けてしまいます．",
    # 15. 実際の出力
    "実際の出力です．\n"
    "どの組み合わせでも，tennis player，括弧person，のように，細かい名前と粗い名前の両方が出せます．\n"
    "一番右のYOLO11xだけの場合は，personしか出ません．",
    # 16. まとめ
    "まとめると3つです．\n"
    "1つ目，学び直さずに，つまみτだけで名前の細かさを変えられるモデルを作りました．\n"
    "2つ目，枝ごとに確率を足す方法で，総合点が0.27から0.64に上がりました．\n"
    "3つ目，80種類以外の物を見つけられるのは，OWLv2だけでした．\n"
    "課題は2つです．細かくて正しい名前がまだ15%と少ないこと．そして，余計な箱を捨てるルールで，80種類以外の物まで捨ててしまうことです．\n"
    "ご清聴ありがとうございました．",
]

NOTES_CT = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"
REL_NOTES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')


def notes_xml(text):
    """
    発表者ノートのページ（notesSlide）のXMLを作る．

    Args:
        text (str): ノートの本文（改行で段落を分ける）．

    Returns:
        str: notesSlide の XML．
    """
    from xml.sax.saxutils import escape
    paras = "".join(f'<a:p><a:r><a:rPr lang="ja-JP" altLang="en-US" sz="1200" dirty="0"/><a:t>{escape(line)}</a:t></a:r></a:p>'
                    for line in text.split("\n"))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<p:notes {NS}><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>'
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="スライド イメージ プレースホルダー 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr>'
            '<p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>'
            '<p:sp><p:nvSpPr><p:cNvPr id="3" name="ノート プレースホルダー 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            f'<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{paras}</p:txBody></p:sp>'
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')


def slide_order(files):
    """
    presentation.xml の sldIdLst の順に，スライドのファイル名を並べる．

    Args:
        files (dict[str, bytes]): pptx 内のパス → 内容．

    Returns:
        list[str]: スライドのファイル名（例："slide13.xml"）の表示順のリスト．
    """
    import re
    rels = files["ppt/_rels/presentation.xml.rels"].decode()
    target = {}
    for rel in re.findall(r"<Relationship [^>]*/>", rels):
        rid, tgt = re.search(r'Id="([^"]+)"', rel).group(1), re.search(r'Target="([^"]+)"', rel).group(1)
        target[rid] = tgt
    pres = files["ppt/presentation.xml"].decode()
    lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S).group(1)
    return [target[r].split("/")[-1] for r in re.findall(r'r:id="([^"]+)"', lst)]


def add_notes(path, notes):
    """
    作成した pptx の各スライドに発表者ノートを書き込む（既存のノートは置き換える）．

    Args:
        path (str): pptx のパス．
        notes (list[str]): スライドの表示順のノートの本文（長さはスライドの枚数）．

    Returns:
        None
    """
    import zipfile
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    order = slide_order(files)
    assert len(order) == len(notes), (len(order), len(notes))
    ct = files["[Content_Types].xml"].decode()
    for k, (slide, text) in enumerate(zip(order, notes), start=1):
        name = f"notesSlide{k}.xml"
        files[f"ppt/notesSlides/{name}"] = notes_xml(text).encode()
        files[f"ppt/notesSlides/_rels/{name}.rels"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" '
            'Target="../notesMasters/notesMaster1.xml"/>'
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/{slide}"/>'
            '</Relationships>').encode()
        rels_path = f"ppt/slides/_rels/{slide}.rels"
        rels = files[rels_path].decode()
        rels = re.sub(r'<Relationship [^>]*Type="' + re.escape(REL_NOTES) + r'"[^>]*/>', "", rels)
        rels = rels.replace("</Relationships>", f'<Relationship Id="rIdNotes" Type="{REL_NOTES}" Target="../notesSlides/{name}"/></Relationships>')
        files[rels_path] = rels.encode()
        part = f"/ppt/notesSlides/{name}"
        if part not in ct:
            ct = ct.replace("</Types>", f'<Override PartName="{part}" ContentType="{NOTES_CT}"/></Types>')
    files["[Content_Types].xml"] = ct.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", files.pop("[Content_Types].xml"))
        for n, data in files.items():
            z.writestr(n, data)


if __name__ == "__main__":
    if not bs.SKILL_SCRIPTS:
        sys.exit("環境変数 PPTX_SKILL_SCRIPTS に pptx スキルの scripts ディレクトリを指定すること")
    bs.REFS.clear()
    bs.REFS.update(REFS)
    bs.OUTPUT = OUTPUT
    media = make_media()
    bs.assemble(build(media), media)
    add_notes(OUTPUT, NOTES)
    print(OUTPUT)
