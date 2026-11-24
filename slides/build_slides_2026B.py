"""
paper_2026B の内容の研究発表スライド（slides_2026B.pptx）を作成するスクリプト．

.ai/ai-pptx-kit/templates/research.pptx を展開し，slides/build_slides.py の部品（OOXMLの組み立て，LaTeXからOMMLへの変換）を用いて
各スライドを書き込む．スライド用の図は，配置する大きさで文字が20ポイントになるように outputs/ex015〜 の結果から描き直す．
LibreOfficeは使用しない．

実行: PPTX_SKILL_SCRIPTS=<pptxスキルのscripts> .venv_pytorch/bin/python slides/build_slides_2026B.py
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_slides as bs  # noqa: E402
from build_slides import RED, Slide, bullet, math_para, math_run, note, para, run  # noqa: E402

ROOT = bs.ROOT
sys.path.insert(0, os.path.join(ROOT, "programs"))
from common.coco import load_coco, load_image  # noqa: E402

OUTPUT = os.path.join(ROOT, "slides_2026B.pptx")
EX015 = os.path.join(ROOT, "outputs", "ex015_lvis_open_vocabulary_annotation")
EX015A = os.path.join(ROOT, "outputs", "ex015a_annotation_score")
EX015B = os.path.join(ROOT, "outputs", "ex015b_baselines_robustness")
EX016 = os.path.join(ROOT, "outputs", "ex016_wholeness_rejection")
MEDIA_DIR = os.path.join(ROOT, "slides", "media_2026B")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
BLUE = "4472C4"

REFS = {
    1: "[1] G. Jocher, J. Qiu, and A. Chaurasia, “Ultralytics YOLO,” https://github.com/ultralytics/ultralytics, 2023.",
    2: "[2] M. Minderer, A. Gritsenko, and N. Houlsby, “Scaling Open-Vocabulary Object Detection,” Proc. NeurIPS, 2023.",
    3: "[3] G. A. Miller, “WordNet: A Lexical Database for English,” Communications of the ACM, vol.38, no.11, pp.39-41, 1995.",
    4: "[4] A. Gupta, P. Dollár, and R. Girshick, “LVIS: A Dataset for Large Vocabulary Instance Segmentation,” Proc. IEEE/CVF CVPR, pp.5356-5364, 2019.",
    5: "[5] T.-Y. Lin, et al., “Microsoft COCO: Common Objects in Context,” Proc. ECCV, pp.740-755, 2014.",
    6: "[6] S. Kiritchenko, et al., “Learning and Evaluation in the Presence of Class Hierarchies,” Proc. Canadian AI, pp.395-406, 2006.",
}


# ===================== スライド用の図 =====================

def _style():
    """スライド用の図の文字の大きさ（配置した大きさで20ポイント）を設定する．"""
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20,
                         "legend.fontsize": 20, "font.family": "DejaVu Sans"})


def fig_stages(path, w, h):
    """
    段階ごとのCOCO外の物体の検出・命名・細分化の割合（ST）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    t = pd.read_csv(os.path.join(EX015, "table_results.csv"))
    t = t[t["model"] == "st"].set_index(["stage", "name"])
    stages = [("S2", "ens_noperson", "K=10"), ("S4", "k20", "K=20"), ("S4", "k50", "K=50"), ("S4", "k100", "K=100")]
    fig, ax = plt.subplots(figsize=(w, h))
    x = range(len(stages))
    for k, (col, label) in enumerate([("noncoco_detected", "detected"), ("noncoco_named", "named"), ("noncoco_fine", "fine")]):
        vals = [t.loc[(a, b), col] for a, b, _ in stages]
        ax.bar([i + (k - 1) * 0.27 for i in x], vals, 0.27, color=COLORS[k], label=label)
    ax.set_xticks(list(x), [label for _, _, label in stages])
    ax.set_ylabel("Fraction of non-COCO objects")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_tradeoff(path, w, h):
    """
    加えた領域の適合率とCOCO外の物体の命名の割合のトレードオフ．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    fig, ax = plt.subplots(figsize=(w, h))
    for (key, label), color in zip([("st", "ST (B/16)"), ("st_large", "ST-L (L/14)"), ("ens", "FT (LVIS fine-tuned)")],
                                   [COLORS[0], COLORS[2], COLORS[1]]):
        p = os.path.join(EX015, f"tradeoff_{key}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        k = d[d["objectness"] == 0.0].sort_values("k_add")
        o = d[d["k_add"] == 100].sort_values("objectness")
        ax.plot(k["test_added_precision"], k["test_noncoco_named"], "-o", color=color, lw=2.5, ms=8, label=label)
        ax.plot(o["test_added_precision"], o["test_noncoco_named"], "--s", color=color, lw=2, ms=6, alpha=0.6)
    ax.set_xlabel("Precision of added regions")
    ax.set_ylabel("Non-COCO objects named")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_examples(path, key, image_ids=("64462", "283113", "304396"), w=12.6, label_size=13):
    """
    出力の例（青：YOLO11x，赤：校正したスコアが高い加えた領域．数値はアノテーションのスコア）を1行に並べる．

    Args:
        path (str): 保存先．
        key (str): OWLv2の重みの名前．
        image_ids (tuple[str]): 画像ID．
        w (float): 配置する幅（インチ）．
        label_size (int): ラベルの文字の大きさ（ポイント）．

    Returns:
        str: 保存先．
    """
    coco = load_coco("instances")
    res = json.load(open(os.path.join(EX015A, "qualitative", f"{key}.json")))
    images = [load_image(coco, int(i)) for i in image_ids]
    ratios = [im.size[0] / im.size[1] for im in images]
    gap = 0.08
    h = (w - gap * (len(images) - 1)) / sum(ratios)
    fig = plt.figure(figsize=(w, h))
    x = 0.0
    for image_id, im, r in zip(image_ids, images, ratios):
        ax = fig.add_axes([x / w, 0, r * h / w, 1])
        x += r * h + gap
        ax.imshow(im)
        ax.axis("off")
        for it in res[image_id]:
            x0, y0, x1, y1 = it["box"]
            c = COLORS[0] if it["source"] == "yolo" else COLORS[3]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=c, lw=1.5))
            ax.text(x0, y0, f"{it['word']} {it['score']:.2f}", fontsize=label_size, color="white", va="top", clip_on=True,
                    bbox=dict(facecolor=c, alpha=0.85, pad=1, lw=0))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_reliability(path, w, h, model="st"):
    """
    スコアの値と正解率の関係（左：アノテーションのスコア，右：経路上の各単語のスコア）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．
        model (str): OWLv2の重みの名前．

    Returns:
        str: 保存先．
    """
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    for kind, label, color in [("det", "detection score", COLORS[0]), ("calibrated", "proposed", COLORS[2])]:
        rel = json.load(open(os.path.join(EX015A, kind, model, "42", "reliability.json")))
        axes[0].plot([r["score"] for r in rel], [r["accuracy"] for r in rel], "-o", lw=2.5, ms=7, color=color, label=label)
    chain = json.load(open(os.path.join(EX015A, f"chain_reliability_{model}.json")))
    for key, label, color in [("table_q", "q", COLORS[1]), ("table_calibrated", "proposed", COLORS[2])]:
        axes[1].plot([r["score"] for r in chain[key]], [r["accuracy"] for r in chain[key]], "-o", lw=2.5, ms=7, color=color,
                     label=label)
    for ax, title in zip(axes, ["annotation score", "score of each word"]):
        ax.plot([0, 1], [0, 1], "k:", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_paired(path, w, h):
    """
    COCO外の物体と対応した同じ加えた領域に対する，flat と greedy の命名が正しい割合．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    keys = [("st", "ST"), ("st_large", "ST-L"), ("ens", "FT")]
    vals = [json.load(open(os.path.join(EX015B, f"paired_naming_{k}.json"))) for k, _ in keys]
    fig, ax = plt.subplots(figsize=(w, h))
    x = range(len(keys))
    for k, (field, label, color) in enumerate([("flat_named", "flat (1 word)", COLORS[1]), ("greedy_named", "proposed", COLORS[2])]):
        v = [d[field] for d in vals]
        bars = ax.bar([i + (k - 0.5) * 0.38 for i in x], v, 0.38, color=color, label=label)
        for b, y in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.01, f"{100 * y:.0f}%", ha="center", va="bottom")
    ax.set_xticks(list(x), [label for _, label in keys])
    ax.set_ylabel("Correctly named")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_errors(path, w, h):
    """
    加えた領域の出力の種類の割合（ST，校正したスコアの閾値以上，および全ての出力）．

    Args:
        path (str): 保存先．
        w, h (float): 配置する大きさ（インチ）．

    Returns:
        str: 保存先．
    """
    kinds = [("correct", "correct"), ("misnamed", "misnamed"), ("part", "part"), ("localization", "loc."),
             ("background", "background")]
    rows = [("all outputs", "greedy", "all"), ("flat, threshold", "flat", "strict_hF"), ("proposed, threshold", "greedy", "strict_hF")]
    colors = [COLORS[2], COLORS[3], "#8C8C8C", "#B0B0B0", "#D9D9D9"]
    fig, ax = plt.subplots(figsize=(w, h))
    for i, (label, method, key) in enumerate(rows):
        e = json.load(open(os.path.join(EX015B, f"{method}_calibrated", "st", "42", "errors.json")))[key]
        left = 0.0
        for (k, name), c in zip(kinds, colors):
            v = e[k] + (e["duplicate"] if k == "misnamed" else 0.0)
            ax.barh(i, v, left=left, color=c, label=name if i == 0 else None)
            if v > 0.07:
                ax.text(left + v / 2, i, f"{100 * v:.0f}", ha="center", va="center")
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of added-region outputs")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=5, frameon=False, columnspacing=0.8, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_media():
    """
    スライド用の図を作成する．

    Returns:
        dict: 図の名前 → パス．
    """
    os.makedirs(MEDIA_DIR, exist_ok=True)
    _style()
    m = {"stages": fig_stages(os.path.join(MEDIA_DIR, "stages.png"), 7.4, 5.2),
         "ex_st": fig_examples(os.path.join(MEDIA_DIR, "examples_st.png"), "st", w=11.5),
         "ex_ens": fig_examples(os.path.join(MEDIA_DIR, "examples_ens.png"), "ens"),
         "reliability": fig_reliability(os.path.join(MEDIA_DIR, "reliability.png"), 12.4, 4.3),
         "paired": fig_paired(os.path.join(MEDIA_DIR, "paired.png"), 7.0, 5.0),
         "errors": fig_errors(os.path.join(MEDIA_DIR, "errors.png"), 12.4, 3.4)}
    m["camel"] = os.path.join(ROOT, "slides", "media", "yolo_camel.png")
    return m


# ===================== スライドの内容 =====================

def metric(key, model="st", crit="fed_hF"):
    """
    operating_points.csv の値を返す．

    Args:
        key (str): 指標名．
        model (str): OWLv2の重みの名前．
        crit (str): o_min の選択の基準．

    Returns:
        float: 値．
    """
    t = pd.read_csv(os.path.join(EX015, "operating_points.csv"))
    return float(t[(t["model"] == model) & (t["criterion"] == crit)][key].iloc[0])


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


def build(media):
    """
    全スライドを定義する．

    Args:
        media (dict): 図の名前 → パス．

    Returns:
        list[Slide]: スライド．
    """
    slides = []
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731

    # 1. タイトル
    s = Slide("", layout=1)
    s.shapes.append(
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="タイトル 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="ctrTitle"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="125506" y="772319"/><a:ext cx="11940525" cy="2656681"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/>'
        '<a:t>WordNetの上位語への確率の集約による</a:t></a:r><a:br><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/></a:br>'
        '<a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/><a:t>再学習なしの粒度可変アノテーション</a:t></a:r></a:p></p:txBody></p:sp>')
    s.shapes.append(
        '<p:sp><p:nvSpPr><p:cNvPr id="4" name="字幕 3"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="subTitle" idx="1"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="125506" y="4339531"/><a:ext cx="11940524" cy="1210860"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>所属機関名</a:t></a:r></a:p>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>○著者名</a:t></a:r></a:p></p:txBody></p:sp>')
    slides.append(s)

    # 2. 背景
    s = Slide("背景：学習したクラスしか出力できない")
    s.text(0.3, 1.0, 7.9, 5.0, [
        bullet("物体認識モデル（YOLO[1]等）は"),
        note("COCO[5]の80クラス等，決まったクラスで学習"),
        bullet("学習時に無い物体は"),
        note("見逃すか，近いクラスと誤る"),
        bullet("語彙の追加・粒度の変更には"),
        para(run("　再学習が必要", 2800, True, RED)),
    ])
    w, h = s.image(media["camel"], 8.5, 1.1, w=4.5)
    s.text(8.5, 1.15 + h, 4.5, 0.5, [para(run("YOLO11x：ラクダ → dog", 2000), align="ctr")])
    s.refs(1, 5)
    slides.append(s)

    # 3. 目的
    s = Slide("目的：再学習なしで，学習時に無い物体と語彙を扱う")
    s.text(0.3, 1.0, 12.7, 1.2, [
        para(run("学習時に無い物体を", 2800), run("検出", 2800, True, RED), run("し，学習時に無い語彙で", 2800),
             run("命名", 2800, True, RED), run("する", 2800)),
        note("確信度に応じて，粗い語から細かい語まで（粒度可変）"),
    ])
    chain = ["artifact", "structure", "tower", "clock tower"]
    for i, word in enumerate(chain):
        x = 0.6 + i * 3.15
        s.box(x, 3.0, 2.6, 1.0, [para(run(word, 2800, True), align="ctr")], fill="DDEBF7", line="accent1", anchor="ctr")
        if i < len(chain) - 1:
            s.arrow(x + 2.65, 3.5, x + 3.1, 3.5)
    s.text(0.6, 4.1, 3.0, 0.5, [para(run("粗い", 2400), align="ctr")])
    s.text(10.05, 4.1, 3.0, 0.5, [para(run("細かい", 2400), align="ctr")])
    s.text(0.3, 4.9, 12.7, 1.5, [
        bullet("語彙：WordNet[3]の7806語（COCOのキャプションに出る語）"),
        bullet("評価：LVIS[4]（1203カテゴリ，WordNetのSynset付き）"),
    ])
    s.refs(3, 4)
    slides.append(s)

    # 4. 以前の版の課題
    s = Slide("以前の版の課題：細かい語まで届かない")
    s.text(0.3, 1.0, 12.7, 2.4, [
        bullet("切り抜き画像＋SigLIPで，上位語から1段ずつ下位語を選択"),
        bullet("COCO外の物体の命名13.7%，LVISのカテゴリまでの細分化0.1%"),
        note("多くは artifact，natural object 等の粗い語で停止"),
    ])
    s.box(0.8, 3.8, 11.7, 2.0, [
        para(run("今回の方針", 2800, True), align="ctr"),
        bullet("OWLv2[2]の領域の特徴で，語彙全体の確率を一度に計算"),
        bullet("確率をWordNetの上位語へ集約し，確信度に応じて止める"),
    ], fill="FFF2CC", line="accent1")
    s.refs(2)
    slides.append(s)

    # 5. 全体像
    s = Slide("提案手法：領域の追加と，確率の集約による命名")
    s.box(0.3, 1.2, 1.6, 1.0, [para(run("画像", 2400), align="ctr")], line="accent1", anchor="ctr")
    s.box(2.5, 1.2, 4.2, 1.0, [para(run("YOLO11x：COCOのクラス", 2400), align="ctr")], line="accent1", anchor="ctr")
    s.box(2.5, 2.7, 4.2, 1.0, [para(run("OWLv2：物体らしさ上位K件", 2400), align="ctr")], fill="DDEBF7", line="accent1",
          anchor="ctr")
    s.arrow(1.95, 1.7, 2.45, 1.7)
    s.arrow(1.95, 1.9, 2.45, 3.2)
    steps = ["領域の特徴から7806語の確率（1）", "WordNetの上位語へ集約（2）", "上位語から貪欲に下位語へ（3）"]
    for i, t in enumerate(steps):
        y = 2.7 + i * 1.25
        s.box(7.3, y, 5.7, 0.95, [para(run(t, 2400), align="ctr")], fill="DDEBF7" if i else "FFFFFF", line="accent1",
              anchor="ctr")
        if i:
            s.arrow(10.15, y - 0.28, 10.15, y - 0.02)
    s.arrow(6.75, 3.2, 7.25, 3.2)
    s.text(0.3, 4.3, 6.6, 2.0, [
        bullet("YOLOと重なる領域は除く"),
        bullet("どの段階も学習なし"),
    ])
    s.refs(1, 2, 3)
    slides.append(s)

    # 6. 数式（1）
    s = Slide("語彙全体の確率：領域の特徴と単語の文章の類似度")
    equation(s, 1.1, 1.3, r"p(w \mid \mathbf{b}) = \frac{\exp(\beta s_{\mathbf{b}} \mathbf{e}_{\mathbf{b}} \mathbf{u}_w^{\top})}"
                          r"{\sum_{w' \in \mathbb{V}} \exp(\beta s_{\mathbf{b}} \mathbf{e}_{\mathbf{b}} \mathbf{u}_{w'}^{\top})}", 1)
    s.text(0.3, 2.7, 12.7, 3.9, [
        para(math_run(r"\mathbf{e}_{\mathbf{b}}", 2400), run("：OWLv2の領域", 2400), math_run(r"\mathbf{b}", 2400),
             run("の埋め込み，", 2400), math_run(r"s_{\mathbf{b}}", 2400), run("：OWLv2のロジットの倍率", 2400)),
        para(math_run(r"\mathbf{u}_w", 2400), run("：単語", 2400), math_run("w", 2400),
             run("の文章の特徴（「a photo of a w」と「w, 定義文」の平均）", 2400)),
        para(math_run(r"\mathbb{V}", 2400), run("：語彙（7806語，人の役割の語は除く），", 2400), math_run(r"\beta", 2400),
             run("：温度の倍率", 2400)),
        bullet("OWLv2の倍率のままでは7806語でほぼ一様", space_before=1200),
        note("→ " + "β を検証用の画像で選択（β = 2）"),
    ])
    slides.append(s)

    # 7. 数式（2）（3）
    s = Slide("上位語へ集約し，確信度がある限り下位語へ進む")
    equation(s, 1.0, 0.95, r"q(n \mid \mathbf{b}) = \sum_{w \in \mathbb{V}} p(w \mid \mathbf{b}) \, \mathbf{1}[n \in \mathrm{Anc}(w)]", 2)
    equation(s, 2.05, 0.95, r"n_{k+1} = \underset{c \in \mathrm{Hypo}_1(n_k)}{\mathrm{arg\,max}}\; q(c \mid \mathbf{b})", 3)
    s.text(0.3, 3.1, 12.7, 3.6, [
        para(math_run(r"\mathrm{Anc}(w)", 2400), run("：", 2400), math_run("w", 2400), run("自身と上位語，", 2400),
             math_run(r"\mathrm{Hypo}_1(n)", 2400), run("：", 2400), math_run("n", 2400), run("の直接の下位語", 2400)),
        para(math_run(r"n_0", 2400), run("：physical entity（根）", 2400)),
        bullet("止める条件："),
        note("1. " + "下位語の確率 q が閾値 τ（= 0.1）未満"),
        note("2. " + "下位語の確率 q が自身の単語の確率 p 以下"),
    ])
    slides.append(s)

    # 7b. アノテーションのスコア
    s = Slide("各アノテーションに，正しさのスコアを付ける")
    equation(s, 0.95, 0.9, r"s(\mathbf{b}, n) = \sigma(\mathbf{w} \mathbf{f}(\mathbf{b}, n)^{\top} + w_0)", 4)
    equation(s, 1.9, 0.9, r"r(n_j \mid \mathbf{b}) = \min_{j' \leq j} \sigma(\mathbf{v} \mathbf{g}(\mathbf{b}, n_{j'})^{\top} + v_0)", 5)
    s.text(0.3, 2.9, 12.7, 3.8, [
        para(math_run(r"\mathbf{f}", 2400), run("：検出のスコア，", 2400), math_run("q", 2400),
             run("，確率の最大値とエントロピー，単語の深さ，出力元", 2400)),
        para(math_run(r"\mathbf{g}", 2400), run("：経路上の単語の", 2400), math_run(r"\log q", 2400), run("，深さ等，", 2400),
             math_run(r"\sigma", 2400), run("：シグモイド関数", 2400)),
        para(math_run(r"\mathbf{w}", 2400), run("，", 2400), math_run(r"\mathbf{v}", 2400),
             run("：検証用の画像のみで求めたロジスティック回帰の重み（9個，5個）", 2400)),
        bullet("(4)：出力全体のスコア → 閾値で出力の数を選ぶ", space_before=900),
        bullet("(5)：経路上の各単語のスコア → 閾値で粒度を選ぶ"),
    ])
    slides.append(s)

    # 8. 実験条件
    s = Slide("実験条件：LVISでCOCO外の物体を評価")
    s.table(0.5, 1.1, [3.0, 9.3], [
        ["項目", "設定"],
        ["画像", "COCO val2017のLVIS[4]注釈付き（検証用2387，テスト用2365枚）"],
        ["COCO外の物体", "COCOのクラスの上位語・下位語でないもの（テスト用13150件）"],
        ["OWLv2[2]", "ST：自己学習のみ（主），ST-L：L/14，FT：LVISで微調整（参考）"],
        ["命名／細分化", "GTの上位語・自身・下位語を出力／GT以下の語まで出力"],
        ["AP（strict）[6]", "主指標：上位語に部分点．GTと対応しない出力は全て誤り"],
        ["AP（exact）", "既存研究との比較：LVISのカテゴリまで正しい出力のみ正解"],
    ], size=2000, row_h=0.55)
    s.text(0.5, 5.3, 12.3, 1.0, [note("設定は全て検証用の画像で選び，テスト用の画像で報告", indent=False),
                                 note("「学習時に無い」＝LVISのカテゴリを正解ラベルとして学習していない", indent=False)])
    s.refs(2, 4, 6)
    slides.append(s)

    # 9. 原因の切り分け
    d = json.load(open(os.path.join(EX015, "diagnosis.json")))
    s = Slide("LVISで微調整していないOWLv2の分類は弱い")
    s.table(0.8, 1.2, [2.6, 4.4, 4.4], [
        ["OWLv2", "LVISの語彙（1202語）", "キャプションの語彙（7806語）"],
        ["ST（主）", pc(d["st"]["lvis_top1_exact"]), pc(d["st"]["caption_top1_exact"])],
        ["ST-L（L/14）", pc(d["st_large"]["lvis_top1_exact"]), pc(d["st_large"]["caption_top1_exact"])],
        ["FT（参考）", pc(d["ens"]["lvis_top1_exact"]), pc(d["ens"]["caption_top1_exact"])],
    ], size=2000, row_h=0.6, highlight=[(1, 2)])
    s.text(0.8, 3.65, 12.0, 0.5, [para(run("COCO外の物体と重なる領域での1位の単語の正解率（テスト用）", 2000))])
    s.text(0.3, 4.3, 12.7, 2.3, [
        bullet("以前の版の高い分類能力は，主にLVISでの微調整による"),
        bullet("学習時に無い物体では，1語の選択は信頼できない"),
        note("→ 確信度に応じて粗い語で止める必要", color=RED),
    ])
    slides.append(s)

    # 10. 語の選び方
    t = pd.read_csv(os.path.join(EX015, "table_results.csv"))
    t = t[t["model"] == "st"].set_index(["stage", "name"])
    s = Slide("確率を集約して止めると，命名が増える")
    rows = [["規則（ST，K = 10）", "命名", "細分化", "hF（fed）"]]
    for name, label in [("flat", "1語を選ぶ（flat）"), ("deepest", "最も深い語（deepest）"), ("backoff", "1位から遡る（backoff）"),
                        ("greedy", "上位語から貪欲（提案）")]:
        r = t.loc[("S1", name)]
        rows.append([label, f3(r["noncoco_named"]), f3(r["noncoco_fine"]), f3(r["fed_hF"])])
    r = t.loc[("S2", "ens_noperson")]
    rows.append(["＋定義文・人の役割を除く", f3(r["noncoco_named"]), f3(r["noncoco_fine"]), f3(r["fed_hF"])])
    s.table(0.5, 1.1, [5.0, 2.2, 2.2, 2.4], rows, size=2000, row_h=0.55, highlight=[(5, 2), (5, 3)])
    s.text(0.3, 4.6, 12.7, 2.0, [
        bullet("1語を選ぶと細かいが誤りが多い（命名3.4%）"),
        bullet("集約して止めると命名5〜6%，hFも向上"),
    ])
    slides.append(s)

    # 11. 領域の数
    s = Slide("加える領域を増やすと，検出と命名が大きく向上する")
    s.image(media["stages"], 0.3, 1.0, w=7.4)
    k10 = t.loc[("S2", "ens_noperson")]
    s.text(7.9, 1.2, 5.2, 5.2, [
        bullet("K = 10 → 100"),
        note(f"検出　{pc(k10['noncoco_detected'])} → {pc(metric('noncoco_detected'))}"),
        note(f"命名　{pc(k10['noncoco_named'])} → {pc(metric('noncoco_named'))}"),
        note(f"細分化 {pc(k10['noncoco_fine'])} → {pc(metric('noncoco_fine'))}"),
        bullet("最も効果が大きい"),
        note("以前：命名13.7%"),
        note("　　　細分化0.1%"),
    ])
    slides.append(s)

    # 13. 重みの比較
    s = Slide("分類能力の高い領域の特徴ほど，細かく命名できる")
    rows = [["OWLv2", "検出", "命名", "細分化"]]
    for key, label in [("st", "ST（B/16）"), ("st_large", "ST-L（L/14）"), ("ens", "FT（参考）")]:
        try:
            rows.append([label] + [f3(metric(k, key)) for k in ["noncoco_detected", "noncoco_named", "noncoco_fine"]])
        except IndexError:
            continue
    s.table(1.5, 1.1, [3.4, 2.4, 2.4, 2.4], rows, size=2000, row_h=0.6)
    s.text(0.3, 1.2 + 0.6 * len(rows), 12.7, 0.5, [para(run("COCO外の物体に対する割合．K = 100，テスト用", 2000))])
    s.text(0.3, 4.0, 12.7, 2.5, [
        bullet(f"大きいST-L：細分化 {pc(metric('noncoco_fine'))} → {pc(metric('noncoco_fine', 'st_large'))}"),
        bullet("FTはLVISのカテゴリを学習時に見ている（参考値）"),
        bullet("再学習なしで精度を上げる鍵は，領域の特徴の分類能力"),
    ])
    slides.append(s)

    # 13a. 同じ領域での命名の比較
    pn = json.load(open(os.path.join(EX015B, "paired_naming_st.json")))
    s = Slide("集約すると，同じ領域の命名が正しくなる")
    s.image(media["paired"], 0.3, 1.0, w=7.0)
    s.text(7.5, 1.1, 5.6, 5.4, [
        bullet("COCO外の物体の領域で比較"),
        note(f"ST：{pc(pn['flat_named'])} → {pc(pn['greedy_named'])}"),
        bullet("flatの誤りを粗い語で回避"),
        note("jersey：jersey → shirt"),
        note("ring：band → jewelry"),
        bullet("分類能力が高いと差は縮む"),
    ])
    slides.append(s)

    # 13a2. APでの比較
    ts = pd.read_csv(os.path.join(EX015B, "table_results.csv"))
    ts = ts[(ts["model"] == "st") & (ts["score"] == "calibrated") & (ts["seed"] == 42)].set_index("method")
    bt = json.load(open(os.path.join(EX015B, "bootstrap_st.json")))["greedy_calibrated - flat_calibrated"]
    s = Slide("APでの差は小さく，細かい語の正解は増えない")
    rows = [["方法（ST）", "AP（strict）", "AP（exact）"]]
    for m, label in [("flat", "1語を選ぶ（flat）"), ("greedy", "提案（greedy）"), ("flat_lvis", "LVISの語彙＋flat（参考）"),
                     ("greedy_lvis", "LVISの語彙＋greedy（参考）")]:
        r = ts.loc[m]
        rows.append([label, f3(r["AP_strict"]), f3(r["AP_exact"])])
    s.table(1.2, 1.1, [5.4, 2.8, 2.8], rows, size=2000, row_h=0.55, highlight=[(2, 1)])
    s.text(1.2, 3.95, 11.5, 0.5, [para(run("テスト用．exact：LVISのカテゴリまで正しい出力のみ正解", 2000))])
    s.text(0.3, 4.55, 12.7, 2.0, [
        bullet(f"提案−flat：strict ＋{bt['AP_strict']['diff']:.3f}（95%区間 {bt['AP_strict']['lo']:.3f}〜{bt['AP_strict']['hi']:.3f}）"),
        bullet("exactではflatが上回る．語彙を与えると効果は無い"),
    ])
    slides.append(s)

    # 13b. スコアの評価
    sc = pd.read_csv(os.path.join(EX015A, "table_results.csv")).set_index(["model", "score"])
    s = Slide("校正したスコアで，正しい出力を上位に並べられる")
    rows = [["スコア（ST）", "AP（strict）", "ECE"]]
    for kind, label in [("det", "検出のスコア c"), ("name", "命名の確率 q"), ("det_x_name_added", "c × q（追加のみ）"),
                        ("calibrated", "提案（式(4)）")]:
        r = sc.loc[("st", kind)]
        rows.append([label, f3(r["AP_strict"]), f3(r["ECE"])])
    s.table(1.5, 1.1, [5.0, 2.8, 2.4], rows, size=2000, row_h=0.55, highlight=[(4, 1), (4, 2)])
    s.text(1.5, 3.95, 11.0, 0.5, [para(run("テスト用．ECE：スコアの値と正解率のずれ（小さいほど良い）", 2000))])
    det, cal = sc.loc[("st", "det")], sc.loc[("st", "calibrated")]
    s.text(0.3, 4.5, 12.7, 2.0, [
        bullet(f"AP（strict） {f3(det['AP_strict'])} → {f3(cal['AP_strict'])}（差の95%区間 ＋0.021〜0.031）"),
        bullet("q のみでは物体でない領域を区別できない"),
    ])
    slides.append(s)

    # 13c. 校正
    ch = json.load(open(os.path.join(EX015A, "chain_reliability_st.json")))
    s = Slide("スコアの値が，そのまま正解率を表す")
    s.image(media["reliability"], 0.45, 1.0, w=12.4)
    s.text(0.3, 5.45, 12.7, 1.3, [
        bullet(f"出力のスコア：ECE {f3(det['ECE'])} → {f3(cal['ECE'])}，各単語：ECE {f3(ch['ECE_q'])} → {f3(ch['ECE_calibrated'])}"),
    ])
    slides.append(s)

    # 13d. 誤りの分解
    s = Slide("誤りの大部分は，物体でない領域への命名")
    s.image(media["errors"], 0.45, 1.0, w=12.4)
    s.text(0.3, 4.6, 12.7, 2.0, [
        bullet("閾値以上でも，物体の一部・位置ずれ・背景が約77%"),
        bullet("GTと重なる領域のみ残すと，AP（strict） 0.197 → 0.265"),
        note("→ 次に改善すべきは，領域が物体の全体かの判定", color=RED),
    ])
    slides.append(s)

    # 13e. 全体らしさの特徴による棄却（order_025，ex016）
    wt = pd.read_csv(os.path.join(EX016, "table_seeds.csv")).set_index(["model", "method", "score"])
    diag = json.load(open(os.path.join(EX016, "diagnosis_st.json")))
    s = Slide("位置関係とWordNetだけでは，物体の一部を棄却できない")
    rows = [["スコアに加える特徴", "ST", "FT"]]
    for kind, label in [("base", "加えない（式(4)）"), ("geom", "＋包含・大きさ"), ("wordnet", "＋WordNetの部分関係"),
                        ("full", "＋両方"), ("rule", "部分関係の領域を棄却")]:
        rows.append([label] + [f3(wt.loc[(m, "greedy", kind)]["AP_strict_seed42"]) for m in ["st", "ens"]])
    s.table(1.5, 1.0, [5.6, 2.4, 2.4], rows, size=2000, row_h=0.5)
    s.text(1.5, 4.05, 11.0, 0.5, [para(run("AP（strict），テスト用．例：carのボックス内のwheelを部分と判定", 2000))])
    s.text(0.3, 4.6, 12.7, 2.0, [
        bullet(f"部分と判定した領域の正解は{pc(diag['meronym_flagged']['correct'])}"),
        note(f"ただし，物体の一部のうち判定できたのは{pc(diag['part']['meronym'])}のみ"),
        bullet("人の衣服など，物体内の物体と物体の一部を区別できない"),
        note("→ 画像の内容に基づく判定が必要", color=RED),
    ])
    slides.append(s)

    # 14. 定性評価
    s = Slide("STでも，COCO外の物体に正しい語とスコアが付く")
    w, h = s.image(media["ex_st"], 0.9, 1.0, w=11.5)
    s.text(0.35, 1.0 + h, 12.6, 0.5, [para(run("ST（赤：加えた領域，青：YOLO11x，数値：スコア）", 2000))])
    s.text(0.35, 1.45 + h, 12.6, 1.4, [
        bullet("経路：artifact 0.89 → jacket 0.82 → ski parka 0.43"),
        note("粗い語ほどスコアが高く，閾値で粒度を選べる"),
    ])
    slides.append(s)

    # 15. まとめ
    s = Slide("まとめ")
    s.text(0.3, 1.0, 12.7, 5.6, [
        bullet("再学習なしで，学習時に無い物体と語彙を扱う方法を提案"),
        note("OWLv2の領域の追加＋確率の上位語への集約による命名"),
        bullet(f"ST：検出{pc(metric('noncoco_detected'))}，命名{pc(metric('noncoco_named'))}，細分化{pc(metric('noncoco_fine'))}"),
        bullet(f"同じ領域の命名：{pc(pn['flat_named'])} → {pc(pn['greedy_named'])}（ただし多くは粗い語）"),
        bullet(f"スコア：AP（strict） {f3(det['AP_strict'])} → {f3(cal['AP_strict'])}，ECE {f3(det['ECE'])} → {f3(cal['ECE'])}"),
        bullet("今後の課題"),
        note("1. " + "画像の内容に基づく，物体の一部・背景の領域の棄却"),
        note("2. " + "学習時に無い物体に対する領域の分類能力の向上"),
    ])
    slides.append(s)
    return slides


if __name__ == "__main__":
    if not bs.SKILL_SCRIPTS:
        sys.exit("環境変数 PPTX_SKILL_SCRIPTS に pptx スキルの scripts ディレクトリを指定すること")
    bs.REFS.clear()
    bs.REFS.update(REFS)
    bs.OUTPUT = OUTPUT
    media = make_media()
    bs.assemble(build(media), media)
    print(OUTPUT)
