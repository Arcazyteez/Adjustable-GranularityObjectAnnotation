"""
学科横断発表（第二回）のスライドを作成するスクリプト（order_048）．

前回の発表スライド「学科横断発表(第一回).pptx」を基準とし，その前半（研究背景，従来手法，目的，前回の提案手法と結果）を
そのまま使い，後半に今回の研究（slides_2026C の内容を学科横断の聴衆向けに簡単にしたもの）をつなげる．
python-pptx で前回の pptx を直接編集する（LibreOffice は使用しない）．元のファイルは変更しない．

実行: .venv_pytorch/bin/python slides/build_slides_gakka2.py
出力: 学科横断発表(第二回).pptx
"""
import copy
import os
import sys

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "slides"))
import build_slides_2026C as C  # noqa: E402

SOURCE = os.path.join(ROOT, "学科横断発表(第一回).pptx")
OUTPUT = os.path.join(ROOT, "学科横断発表(第二回).pptx")
MEDIA = os.path.join(ROOT, "slides", "media_2026C")
RED = RGBColor(0xFF, 0x00, 0x00)
BLUE_FILL = RGBColor(0xDD, 0xEB, 0xF7)
YELLOW_FILL = RGBColor(0xFF, 0xF2, 0xCC)
GRAY_FILL = RGBColor(0xF2, 0xF2, 0xF2)

#: 前回のスライドから使うもの（前回の番号, 新しいタイトル．None ならそのまま）．この順に並べる．
KEEP_BEFORE = [(1, None), (2, None), (3, None), (4, None), (5, None), (6, None), (7, None), (8, None), (9, None),
               (10, "前回の提案手法"), (11, None), (15, "前回のパイプライン"), (26, "前回の提案手法"),
               (14, "従来手法との比較"), (20, "前回の結果"), (21, "前回の結果")]


# ===================== スライドの部品 =====================

def new_slide(prs, title):
    """
    「タイトルとコンテンツ」のレイアウトでスライドを追加する．本文のプレースホルダーは削除する．

    Args:
        prs (Presentation): プレゼンテーション．
        title (str): タイトル．

    Returns:
        Slide: 追加したスライド．
    """
    layout = next(l for l in prs.slide_layouts if l.name == "タイトルとコンテンツ")
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = title
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx == 1:
            ph._element.getparent().remove(ph._element)
    return s


def text(s, x, y, w, h, items, size=24):
    """
    箇条書きのテキストボックスを追加する．

    Args:
        s (Slide): スライド．
        x, y, w, h (float): 位置と大きさ（インチ）．
        items (list[tuple]): (文字列, 階層, オプションの辞書) の組．階層0は「・」付き，1は字下げの注．
            オプション："color"（RGBColor），"bold"（bool），"size"（int，ポイント）．
        size (int): 階層0の文字の大きさ（ポイント）．階層1は4ポイント小さくする．

    Returns:
        Shape: テキストボックス．
    """
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for k, item in enumerate(items):
        body, level, opt = (item + ({},))[:3] if len(item) == 2 else item
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ("・" if level == 0 else "　") + body
        r.font.size = Pt(opt.get("size", size if level == 0 else size - 4))
        r.font.bold = opt.get("bold", False)
        if opt.get("color") is not None:
            r.font.color.rgb = opt["color"]
        if level == 0 and k > 0:
            p.space_before = Pt(8)
    return tb


def box(s, x, y, w, h, lines, fill=BLUE_FILL, size=20, bold=False, color=None, line_color=None):
    """
    塗りつぶした角丸四角形に文字を入れる．

    Args:
        s (Slide): スライド．
        x, y, w, h (float): 位置と大きさ（インチ）．
        lines (list[str]): 行ごとの文字列．
        fill (RGBColor): 塗りの色．
        size (int): 文字の大きさ（ポイント）．
        bold (bool): 太字にするか．
        color (RGBColor | None): 文字の色．
        line_color (RGBColor | None): 枠線の色（None はテーマの色）．

    Returns:
        Shape: 図形．
    """
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.1
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line_color is not None:
        sh.line.color.rgb = line_color
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for k, line in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color or RGBColor(0, 0, 0)
    return sh


def arrow(s, x1, y1, x2, y2):
    """
    矢印（直線コネクタ，終点に三角形）を追加する．

    Args:
        s (Slide): スライド．
        x1, y1, x2, y2 (float): 始点と終点（インチ）．

    Returns:
        Shape: コネクタ．
    """
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor(0, 0, 0)
    c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def pipeline(s, y, steps, fill=YELLOW_FILL, size=20):
    """
    処理の流れの箱を横に並べ，矢印でつなぐ．

    Args:
        s (Slide): スライド．
        y (float): 上端（インチ）．
        steps (list[str]): 箱の文字．
        fill (RGBColor): 塗りの色．
        size (int): 文字の大きさ（ポイント）．

    Returns:
        None
    """
    n = len(steps)
    gap = 0.4
    w = (12.6 - gap * (n - 1)) / n
    for i, t in enumerate(steps):
        x = 0.35 + i * (w + gap)
        box(s, x, y, w, 0.95, t.split("\n"), fill=fill, size=size)
        if i < n - 1:
            arrow(s, x + w + 0.02, y + 0.475, x + w + gap - 0.02, y + 0.475)


def table(s, x, y, widths, rows, size=18, row_h=0.55, highlight=()):
    """
    表を追加する（1行目は見出し．テンプレートの表のスタイルを用いる）．

    Args:
        s (Slide): スライド．
        x, y (float): 位置（インチ）．
        widths (list[float]): 列の幅（インチ）．
        rows (list[list[str]]): セルの文字列．
        size (int): 文字の大きさ（ポイント）．
        row_h (float): 行の高さ（インチ）．
        highlight (iterable[tuple[int, int]]): 赤字・太字にするセルの (行, 列)．

    Returns:
        Shape: 表．
    """
    gf = s.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows)))
    tbl = gf.table
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = value
            r.font.size = Pt(size)
            r.font.bold = i == 0 or (i, j) in highlight
            if (i, j) in highlight:
                r.font.color.rgb = RED
    return gf


def picture(s, path, x, y, w):
    """
    画像を幅 w で追加する（縦横比を保つ）．

    Args:
        s (Slide): スライド．
        path (str): 画像のパス．
        x, y, w (float): 位置と幅（インチ）．

    Returns:
        tuple[float, float]: 配置した幅と高さ（インチ）．
    """
    pic = s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    return w, pic.height / 914400


def notes(s, body):
    """
    発表者ノートを書き込む（既存のノートは置き換える）．

    Args:
        s (Slide): スライド．
        body (str): ノートの本文（改行で段落を分ける）．

    Returns:
        None
    """
    s.notes_slide.notes_text_frame.text = body


# ===================== 前回のスライドの整理 =====================

def reorder(prs, order):
    """
    スライドを指定の順に並べ，order に無いスライドを削除する．

    Args:
        prs (Presentation): プレゼンテーション．
        order (list[Slide]): 新しい順のスライド．

    Returns:
        None
    """
    lst = prs.slides._sldIdLst
    ids = {sid.get("id"): sid for sid in lst}
    by_slide = {s.slide_id: s for s in prs.slides}
    keep = [ids[str(s.slide_id)] for s in order]
    for sid in list(lst):
        lst.remove(sid)
        if sid not in keep:
            prs.part.drop_rel(sid.get(qn("r:id")))
    for sid in keep:
        lst.append(sid)
    assert len(prs.slides) == len(order), (len(prs.slides), len(order), len(by_slide))


def retitle(slide, title):
    """
    スライドのタイトルを置き換える（最初のランの書式を保つ）．

    Args:
        slide (Slide): スライド．
        title (str): 新しいタイトル．

    Returns:
        None
    """
    tf = slide.shapes.title.text_frame
    p = tf.paragraphs[0]
    runs = p.runs
    runs[0].text = title
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def set_title_slide(slide):
    """
    表紙のタイトルを今回の発表の題名にする（字幕の学籍番号と氏名はそのまま）．

    Args:
        slide (Slide): 表紙のスライド．

    Returns:
        None
    """
    tf = slide.shapes.title.text_frame
    p = tf.paragraphs[0]
    first = p.runs[0]
    first.text = "進捗報告（第二回）"
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    q = copy.deepcopy(p._p)
    p._p.addnext(q)
    sub = tf.paragraphs[1]
    sub.runs[0].text = "学び直さずに，名前の細かさを変えられる物体アノテーション"
    sub.runs[0].font.size = Pt(24)


# ===================== 今回のスライド =====================

def add_new_slides(prs):
    """
    今回の研究のスライドを追加する（学科横断の聴衆向けの言い方）．

    Args:
        prs (Presentation): プレゼンテーション．

    Returns:
        list[Slide]: 追加したスライド（表示順）．
    """
    df, e3, det = C.summary(), C.e3_summary(), C.detector_table()
    pct = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    old = df.loc[C.OLD]
    new = e3.loc["OWLv2@50_Alpha-crop_proposed"]
    dets = [("YOLO11x", "YOLO11x"), ("DETR", "DETR"), ("OWLv2@50", "OWLv2")]
    out = []

    # 前回の課題
    s = new_slide(prs, "前回の課題：3つの問題が見えた")
    text(s, 0.3, 1.0, 12.7, 5.8, [
        ("課題1：DETRは80種類以外の物を見落とす", 0),
        ("学習した80種類以外は背景として扱われ，箱が出ない", 1),
        ("課題2：切り抜くと，周りの様子が分からない", 0),
        ("人と持ち物を取り違えやすい", 1),
        ("課題3：1段目の誤りが，2段目へ伝わる", 0),
        ("前回の結果のObj3・Obj4の失敗．WordNetのpersonの下は398語もある", 1),
        ("→ 今回：部品と，細かさの決め方を見直した", 0, {"color": RED, "bold": True}),
    ], size=26)
    notes(s, "前回の結果から，課題が3つ見えました．\n"
             "1つ目，DETRは学習した80種類以外の物を見落とします．\n"
             "2つ目，物を切り抜いてから名前を当てるので，周りの様子が分かりません．\n"
             "3つ目，前回の結果のObj3やObj4のように，1段目で間違えると，その誤りが2段目に伝わります．"
             "WordNetではpersonの下に398語もあり，正しい語を選ぶのが難しいです．\n"
             "そこで今回は，部品と，細かさの決め方を見直しました．")
    out.append(s)

    # 前回の「これから」と今回
    s = new_slide(prs, "前回の「これから」と，今回やったこと")
    table(s, 0.3, 1.1, [5.2, 7.5], [
        ["前回の「これから」", "今回やったこと"],
        ["Depth=1：定量評価", "COCOの写真500枚で，正解と自動で照合した"],
        ["Depth≧2：人手評価（正解が無い）", "写真の説明文から細かい正解を自動で作った（912件）"],
        ["確信度の閾値で，安全に止める", "枝ごとに確率を足し，つまみτ以上で止める"],
    ], size=20, row_h=0.8, highlight=[(1, 1), (2, 1), (3, 1)])
    text(s, 0.3, 4.6, 12.7, 1.5, [
        ("前回の計画を，3つとも実行した", 0),
        ("加えて，物を見つける部品と名前を当てる部品も取り替えた", 1),
    ], size=24)
    notes(s, "前回の発表の最後に，これからやることを3つ挙げました．\n"
             "1つ目の定量評価は，COCOの写真500枚で，正解と自動で照合しました．\n"
             "2つ目の細かい粒度の評価は，正解が無いので人手で評価する予定でしたが，"
             "写真の説明文に出てくる言葉から，細かい正解を自動で作りました．912件あります．\n"
             "3つ目の，確信度で安全に止める仕組みは，枝ごとに確率を足して，つまみτ以上で止める方法として作りました．")
    out.append(s)

    # 部品の取り替え
    s = new_slide(prs, "今回：4つの部品を取り替えた")
    table(s, 0.3, 1.0, [2.6, 2.4, 3.0, 4.7], [
        ["部品", "前回", "今回", "理由"],
        ["見つける係", "DETR", "OWLv2", f"80種類以外も見つける（{pct(det.loc['DETR', 'noncoco_recall'])}→{pct(det.loc['OWLv2@50', 'noncoco_recall'])}）"],
        ["名前を当てる係", "CLIP", "Alpha-CLIP", "周りも見ながら，注目する"],
        ["辞書", "WordNet全部", "WordNet∩VG", "写真によく出る語だけ（398→136語）"],
        ["細かさの決め方", "1段ずつ下る", "枝ごとに足す", "誤りが伝わらない"],
    ], size=18, row_h=0.62, highlight=[(1, 3), (2, 3), (3, 3), (4, 3)])
    text(s, 0.3, 4.3, 12.7, 2.4, [
        ("どの部品も，学び直さずにそのまま使う", 0),
        ("OWLv2：クラスによらず「物らしさ」で箱を出す検出器", 1, {"size": 18}),
        ("Alpha-CLIP：注目する領域を指定でき，周りの様子も使うCLIP", 1, {"size": 18}),
        ("VG（Visual Genome）：写真に写った物の名前を大量に集めたデータ", 1, {"size": 18}),
    ], size=24)
    notes(s, "前回の課題に対して，部品を4つ取り替えました．\n"
             "見つける係は，何でも見つけられるOWLv2にしました．80種類以外の物を見つける割合が12%から73%になります．\n"
             "名前を当てる係は，周りも見ながら注目するAlpha-CLIPにしました．\n"
             "辞書は，写真によく出てくる言葉だけに絞りました．personの下は398語から136語になりました．\n"
             "細かさの決め方は，1段ずつ下るのをやめ，枝ごとに確率を足す方法にしました．これで1段目の誤りが伝わらなくなります．\n"
             "どの部品も，学び直さずにそのまま使います．")
    out.append(s)

    # 今回のモデル
    s = new_slide(prs, "今回のパイプライン")
    pipeline(s, 1.2, ["OWLv2で\n見つける", "Alpha-CLIPで\n名前の確率", "枝ごとに\n確率を足す", "余計な箱を\n捨てる", "つまみτで\n細かさを選ぶ"], size=18)
    text(s, 0.3, 2.6, 12.7, 4.2, [
        ("見つけるのは1回だけ（前回と同じ）", 0),
        ("辞書：写真によく出る3263語の木（WordNet∩VG）", 0),
        ("余計な箱を捨てる：80種類のどれかに見える箱だけを残す", 0),
        ("この規則が，後で弱点になる", 1, {"color": RED}),
        ("どれも学び直さない", 0, {"color": RED, "bold": True}),
    ], size=24)
    notes(s, "今回のパイプラインです．\n"
             "OWLv2で物を1回だけ見つけ，Alpha-CLIPで辞書の言葉ごとの確率を出します．\n"
             "次に，辞書の木の枝ごとに確率を足します．\n"
             "OWLv2は箱をたくさん出すので，80種類のどれかに見える箱だけを残し，余計な箱を捨てます．この規則が後で弱点になります．\n"
             "最後に，つまみτで細かさを選びます．どの部品も学び直しません．")
    out.append(s)

    # 枝ごとに足す
    s = new_slide(prs, "枝ごとに確率を足し，一番細かい名前を選ぶ")
    box(s, 0.8, 1.15, 3.6, 0.8, ["person　合計50%"], size=22, bold=True)
    box(s, 0.8, 2.55, 3.6, 0.8, ["skater　合計15%"], size=22, bold=True)
    box(s, 0.8, 3.95, 3.6, 0.8, ["skateboarder　8%"], size=22, bold=True)
    box(s, 4.9, 2.55, 3.0, 0.8, ["その他の人の語　35%"], fill=GRAY_FILL, size=20)
    arrow(s, 2.6, 1.95, 2.6, 2.55)
    arrow(s, 2.6, 3.35, 2.6, 3.95)
    arrow(s, 3.6, 1.95, 6.4, 2.55)
    text(s, 0.5, 4.85, 12.0, 0.5, [("skaterの合計15% ＝ skater自身7% ＋ skateboarder 8%（数値は例）", 1, {"size": 18})])
    b = box(s, 8.4, 1.15, 4.6, 2.9, ["合計がτ以上で，一番細かい名前", "τ = 0.3 → person", "τ = 0.1 → skater", "τ = 0.05 → skateboarder"],
            fill=RGBColor(0xFF, 0xFF, 0xFF), size=22)
    b.text_frame.paragraphs[0].runs[0].font.bold = True
    b.text_frame.paragraphs[0].runs[0].font.size = Pt(20)
    b.text_frame.paragraphs[3].runs[0].font.color.rgb = RED
    b.text_frame.paragraphs[3].runs[0].font.bold = True
    text(s, 0.3, 5.45, 12.7, 1.4, [
        ("τ＝細かさのつまみ：高いと粗い名前，低いと細かい名前", 0, {"color": RED}),
        ("前回のように1段ずつ選ばないので，1段目の誤りが伝わらない", 0),
    ], size=24)
    notes(s, "ここが一番大事なところです．" + C.TREE_NOTE + "\n前回は1段ずつ選んでいたので，1段目で間違えると止まりませんでした．今回は枝ごとに足すので，その心配がありません．")
    out.append(s)

    # 実験条件
    s = new_slide(prs, "実験：COCOの写真500枚で試した")
    text(s, 0.3, 1.0, 12.7, 3.4, [
        ("写真：COCO 2017（つまみの調整に500枚，評価に500枚）", 0),
        ("細かい正解：写真の説明文に出る言葉（912件，85%は人）", 0),
        ("80種類以外の物の正解：LVISの注釈（2687件）", 0),
        ("つまみ：τ = 0.05（調整用の写真で，総合点が一番高い値）", 0),
    ], size=24)
    box(s, 0.8, 4.4, 11.7, 2.0, ["総合点（hF）：細かい正解にどれだけ近いか",
                                  "正解がskateboarderのとき　skateboarder → 満点，skater → 部分点，person → 少し，car → 0点"],
        size=20, fill=BLUE_FILL)
    notes(s, C.NOTES[11] + "\n総合点は，細かい正解にどれだけ近いかを表します．"
             "正解がskateboarderなら，skateboarderで満点，skaterで部分点，personでも少し点が入り，carのような間違いは0点です．")
    out.append(s)

    # τ の例
    s = new_slide(prs, "つまみτを下げると，同じ物の名前が細かくなる")
    w, h = picture(s, os.path.join(MEDIA, "tau_examples.png"), 0.45, 0.95, 12.4)
    text(s, 0.3, 0.95 + h + 0.02, 12.7, 1.0, [
        ("物の位置は正解を使用．赤：一番細かい名前．きれいに細かくなる例を選んだ", 1, {"size": 16}),
        ("つまみ1つで，学び直さずに細かさが変わる", 0, {"color": RED, "bold": True, "size": 22}),
    ])
    notes(s, C.NOTES[12])
    out.append(s)

    # 細かさと正しさ
    s = new_slide(prs, "細かくすると，間違いも増える")
    picture(s, os.path.join(MEDIA, "tau_tradeoff.png"), 0.2, 1.0, 7.6)
    text(s, 7.9, 1.05, 5.2, 5.6, [
        ("τを下げる", 0),
        ("細かい名前も，間違いも増える", 1, {"size": 20}),
        ("τを上げる", 0),
        ("粗い名前や答えなしが増える", 1, {"size": 20}),
        ("細かくて正しい名前は最大15%", 0, {"color": RED}),
        ("ここが今後の課題", 1, {"size": 20}),
    ], size=24)
    notes(s, C.NOTES[13])
    out.append(s)

    # 枝ごとに足す効果
    s = new_slide(prs, "位置を正解にすると，総合点0.27→0.64")
    table(s, 1.3, 1.05, [6.0, 2.2, 2.2], [
        ["名前の決め方（位置は正解）", "総合点（hF）", "正解率"],
        ["80種類の名前だけ", "0.27", "31%"],
        ["WordNetを1段ずつ下る（前回）", "0.27", "30%"],
        ["枝ごとに足す（今回）", "0.64", "78%"],
    ], size=20, row_h=0.6, highlight=[(3, 1), (3, 2)])
    text(s, 0.3, 3.7, 12.7, 2.9, [
        ("自信がないときは，無理に細かくせずpersonで止まる", 0),
        ("物の位置は正解を使い，名前の付け方だけを比べた", 1, {"size": 18}),
        ("正解率：正しい名前（粗くても可）を出した割合", 1, {"size": 18}),
    ], size=24)
    notes(s, C.NOTES[14])
    out.append(s)

    # 見つける係
    rows = [["見つける係", "80種類の物", "80種類以外の物"]]
    for key, label in dets:
        rows.append([label, pct(det.loc[key, "recall"]), pct(det.loc[key, "noncoco_recall"])])
    s = new_slide(prs, "80種類以外の物を見つけられるのは，OWLv2だけ")
    table(s, 1.8, 1.05, [3.4, 2.8, 3.2], rows, size=20, row_h=0.6, highlight=[(3, 1), (3, 2)])
    text(s, 0.3, 3.7, 12.7, 2.9, [
        ("見つけられない物には，名前を付けられない", 0),
        ("→ 見つける係にOWLv2を選んだ", 0, {"color": RED}),
        ("表の値：正解の物のうち，箱で囲めた割合（80種類以外はLVISの注釈）", 1, {"size": 18}),
        ("ただしOWLv2は余計な箱も多い（87%は物に当たらない）", 1, {"size": 18}),
    ], size=24)
    notes(s, C.NOTES[15])
    out.append(s)

    # 弱点
    rows = [["見つける係（名前はAlpha-CLIP）", "80種類の点数（F値）", "細かくて正しい"]]
    for key, label in dets:
        r = e3.loc[f"{key}_Alpha-crop_proposed"]
        rows.append([label, f2(r.f1), pct(r.sub_finer_acc)])
    rows.append(["前回（DETR＋CLIP）", f2(old.f1), pct(old.sub_finer_acc)])
    s = new_slide(prs, "弱点：80種類の点数はYOLO・DETRが上")
    table(s, 1.3, 1.05, [5.4, 2.8, 2.4], rows, size=20, row_h=0.6, highlight=[(1, 1)])
    text(s, 0.3, 4.2, 12.7, 2.6, [
        ("OWLv2は，余計な物も多く囲んでしまう", 0),
        ("余計な箱を捨てる規則で，80種類以外の物も捨てる", 0, {"color": RED}),
        (f"80種類以外の物：見つけた{pct(det.loc['OWLv2@50', 'noncoco_recall'])} → 残したのは{pct(new.noncoco_recall)}．"
         "捨てる規則の改良が今後の課題", 1, {"color": RED, "size": 18}),
    ], size=24)
    notes(s, C.NOTES[16] + "\nつまり，今はOWLv2の良さが最終的な出力に出ていません．捨てる規則を直せば，この良さが生きます．")
    out.append(s)

    # 名前を当てる係
    rows = [["見つける係", "CLIP", "Alpha-CLIP"]]
    for key, label in dets:
        rows.append([label, f2(e3.loc[f"{key}_CLIP-B32_proposed", "f1"]), f2(e3.loc[f"{key}_Alpha-crop_proposed", "f1"])])
    s = new_slide(prs, "名前を当てる係は，Alpha-CLIPがCLIPより良い")
    table(s, 2.3, 1.05, [3.4, 2.4, 2.6], rows, size=20, row_h=0.6, highlight=[(1, 2), (2, 2), (3, 2)])
    text(s, 0.3, 3.7, 12.7, 2.9, [
        ("どの見つける係と組み合わせても，Alpha-CLIPが高い", 0),
        ("細かくて正しい名前の割合も約2倍（5%前後 → 11〜14%）", 0),
        ("表の値：80種類の名前の点数（F値）", 1, {"size": 18}),
    ], size=24)
    notes(s, C.NOTES[17])
    out.append(s)

    # 実際の出力
    s = new_slide(prs, "細かい名前と粗い名前を，同時に出せる")
    w, h = picture(s, os.path.join(ROOT, "paper_2026C", "figures", "fig_qualitative_slide.png"), 0.6, 0.95, 12.1)
    text(s, 0.35, 0.95 + h, 12.6, 0.5, [("(a)〜(c) 今回のモデル（見つける係を替えた3通り），(d) YOLO11xだけ．括弧内は粗い名前", 1, {"size": 16})])
    notes(s, C.NOTES[21])
    out.append(s)

    # 前回と今回（全体）
    s = new_slide(prs, "前回と今回：見つける力と細かい名前は良くなった")
    table(s, 1.3, 1.05, [6.0, 2.2, 2.2], [
        ["見つけるところから全部（写真500枚）", "前回", "今回"],
        ["80種類以外の物を見つけた割合", pct(det.loc["DETR", "noncoco_recall"]), pct(det.loc["OWLv2@50", "noncoco_recall"])],
        ["細かくて正しい名前の割合", pct(old.sub_finer_acc), pct(new.sub_finer_acc)],
        ["80種類の名前の点数（F値）", f2(old.f1), f2(new.f1)],
        ["総合点（hF）", f2(old.sub_hF), f2(new.sub_hF)],
    ], size=20, row_h=0.6, highlight=[(1, 2), (2, 2), (3, 2)])
    text(s, 0.3, 4.3, 12.7, 2.4, [
        ("見つける力と，細かい名前は良くなった", 0),
        ("ただし，全部を通した総合点（hF）はほぼ同じ", 0, {"color": RED}),
        ("位置を正解にした場合（0.27→0.64）と違い，見つける段階の誤りや捨てた箱が影響する", 1, {"size": 18}),
    ], size=24)
    notes(s, "最後に，見つけるところから全部を通して，前回と今回を比べます．\n"
             "80種類以外の物を見つけた割合は12%から73%に，細かくて正しい名前の割合は6%から11%に良くなりました．\n"
             "ただし，全部を通した総合点はほぼ同じです．"
             "先ほどの0.27から0.64は，物の位置を正解で与えた場合の値でした．"
             "全部を通すと，見つける段階の誤りや，捨ててしまった箱が影響するためです．")
    out.append(s)

    # まとめ
    s = new_slide(prs, "まとめとこれから")
    text(s, 0.3, 1.0, 12.7, 5.8, [
        ("前回：DETR＋CLIP＋WordNetで，学び直さずに名前を細かくした", 0),
        ("今回：部品を取り替え，つまみτで細かさを選べるようにした", 0),
        ("例：person → skater → skateboarder", 1),
        ("位置を正解にすると，総合点が0.27 → 0.64", 1),
        ("80種類以外の物を見つけられるのは，OWLv2だけ", 1),
        ("これから1：細かくて正しい名前を増やす（今は最大15%）", 0, {"color": RED}),
        ("これから2：80種類以外の物を捨てない規則を作る", 0, {"color": RED}),
        ("これから3：人以外の物（LVISの注釈）でも細かさを評価する", 0, {"color": RED}),
    ], size=24)
    notes(s, "まとめです．\n"
             "前回は，DETRとCLIPとWordNetで，学び直さずに名前を細かくする方法を作りました．\n"
             "今回は，部品を取り替えて，つまみτで細かさを選べるようにしました．"
             "person，skater，skateboarderのように細かさが変わり，位置を正解にすると総合点は0.27から0.64に上がりました．"
             "また，80種類以外の物を見つけられるのはOWLv2だけでした．\n"
             "これからは3つです．細かくて正しい名前を増やすこと．80種類以外の物を捨てない規則を作ること．"
             "そして，人以外の物でも細かさを評価することです．\n"
             "ご清聴ありがとうございました．")
    out.append(s)
    return out


def main():
    """
    前回の pptx を読み込み，使うスライドを並べ替え，今回のスライドを加えて保存する．

    Returns:
        None
    """
    prs = Presentation(SOURCE)
    old = list(prs.slides)
    keep = []
    for number, title in KEEP_BEFORE:
        s = old[number - 1]
        if number == 1:
            set_title_slide(s)
        elif title:
            retitle(s, title)
        keep.append(s)
    new = add_new_slides(prs)
    reorder(prs, keep + new)
    prs.save(OUTPUT)
    print(len(prs.slides), "slides ->", OUTPUT)


if __name__ == "__main__":
    main()
