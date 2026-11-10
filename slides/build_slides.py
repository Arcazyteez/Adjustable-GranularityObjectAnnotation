"""
研究発表スライド（slides_object_annotation.pptx）を作成するスクリプト．

.ai/ai-pptx-kit/templates/research.pptx を展開し，スライドを追加した上で各スライドのOOXMLを直接書き込む．
数式はLaTeXからpandocでOMMLへ変換し，PowerPointのネイティブ数式（a14:m）として挿入する．
LibreOfficeは使用しない．

実行: .venv_pytorch/bin/python slides/build_slides.py
"""
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from xml.sax.saxutils import escape

import matplotlib
import matplotlib.pyplot as plt
import pypandoc
from PIL import Image

matplotlib.use("Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, ".ai", "ai-pptx-kit", "templates", "research.pptx")
OUTPUT = os.path.join(ROOT, "slides_object_annotation.pptx")
PAPER_FIG = os.path.join(ROOT, "paper", "figures")
MEDIA_DIR = os.path.join(ROOT, "slides", "media")
SKILL_SCRIPTS = os.environ.get("PPTX_SKILL_SCRIPTS", "")
EMU = 914400  # 1インチあたりのEMU
FONT = '<a:latin typeface="BIZ UDGothic" panose="020B0400000000000000" pitchFamily="49" charset="-128"/>' \
       '<a:ea typeface="BIZ UDGothic" panose="020B0400000000000000" pitchFamily="49" charset="-128"/>'
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS_A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
RED = "FF0000"

# 引用文献（スライド下部に記載する）
REFS = {
    1: "[1] J. Redmon, et al., “You Only Look Once: Unified, Real-Time Object Detection,” Proc. IEEE CVPR, pp.779-788, 2016.",
    2: "[2] N. Carion, et al., “End-to-End Object Detection with Transformers,” Proc. ECCV, pp.213-229, 2020.",
    3: "[3] M. Minderer, et al., “Simple Open-Vocabulary Object Detection with Vision Transformers,” Proc. ECCV, pp.728-755, 2022.",
    4: "[4] S. Liu, et al., “Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection,” Proc. ECCV, 2024.",
    5: "[5] A. Radford, et al., “Learning Transferable Visual Models From Natural Language Supervision,” Proc. ICML, pp.8748-8763, 2021.",
    6: "[6] G. A. Miller, “WordNet: A Lexical Database for English,” Communications of the ACM, vol.38, no.11, pp.39-41, 1995.",
    7: "[7] T.-Y. Lin, et al., “Microsoft COCO: Common Objects in Context,” Proc. ECCV, pp.740-755, 2014.",
    8: "[8] G. Jocher, J. Qiu, and A. Chaurasia, “Ultralytics YOLO,” https://github.com/ultralytics/ultralytics, 2023.",
    9: "[9] A. Gupta, P. Dollár, and R. Girshick, “LVIS: A Dataset for Large Vocabulary Instance Segmentation,” Proc. IEEE/CVF CVPR, pp.5356-5364, 2019.",
}


def emu(inch):
    """
    インチをEMUに変換する．

    Args:
        inch (float): 長さ（インチ）．

    Returns:
        int: 長さ（EMU）．
    """
    return int(round(inch * EMU))


# ===================== 数式（LaTeX → OMML） =====================

@functools.lru_cache(maxsize=None)
def latex_to_omml(latex, display=True, size=2800):
    """
    LaTeXの数式をpandoc（docx経由）でOMMLに変換し，PowerPoint用に整形する．

    Args:
        latex (str): LaTeXの数式（$ を含まない）．
        display (bool): Trueで独立した数式（m:oMathPara），Falseで文中の数式（m:oMath）．
        size (int): 文字サイズ（1/100ポイント）．

    Returns:
        str: a14:m の中に入れるOMML文字列．
    """
    with tempfile.TemporaryDirectory() as tmp:
        md, docx = os.path.join(tmp, "m.md"), os.path.join(tmp, "m.docx")
        with open(md, "w") as f:
            f.write(f"$${latex}$$\n" if display else f"x ${latex}$\n")
        pypandoc.convert_file(md, "docx", outputfile=docx)
        xml = zipfile.ZipFile(docx).read("word/document.xml").decode()
    tag = "m:oMathPara" if display else "m:oMath"
    omml = re.search(rf"<{tag}>.*?</{tag}>", xml, re.S).group(0)
    omml = omml.replace(f"<{tag}>", f'<{tag} xmlns:m="{NS_M}">', 1)
    omml = re.sub(r"<m:r>(<m:rPr>.*?</m:rPr>)?<m:t>(.*?)</m:t></m:r>", to_math_alphanumeric, omml)
    a_rpr = (f'<a:rPr lang="en-US" altLang="ja-JP" sz="{size}" i="1">'
             '<a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/></a:rPr>')
    # 日本語を含むランは東アジア文字のフォントを指定する（Cambria Mathには和文の字形がないため）
    a_rpr_ja = (f'<a:rPr lang="ja-JP" altLang="en-US" sz="{size}" i="0">'
                '<a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/>'
                f'{FONT.split("/>")[1]}/></a:rPr>')
    omml = re.sub(r"(<m:r>(?:<m:rPr>.*?</m:rPr>)?)(<m:t>[^<]*</m:t>)",
                  lambda m: m.group(1) + (a_rpr_ja if re.search(r"[\u3000-\u9fff]", m.group(2)) else a_rpr) + m.group(2), omml)
    return re.sub(r">\s+<", "><", omml)


# 白抜き文字（double-struck）でBMPの文字様記号に割り当てられている文字
DOUBLE_STRUCK_EXCEPTIONS = {"C": "\u2102", "H": "\u210d", "N": "\u2115", "P": "\u2119", "Q": "\u211a", "R": "\u211d", "Z": "\u2124"}


def map_letters(text, upper_base, lower_base, exceptions=None):
    """
    英字を数学用英数字（Unicode）に置き換える．

    Args:
        text (str): 文字列．
        upper_base (int | None): 大文字Aに対応するコードポイント．
        lower_base (int | None): 小文字aに対応するコードポイント．
        exceptions (dict | None): 個別に割り当てる文字．

    Returns:
        str: 置き換えた文字列．
    """
    out = []
    for c in text:
        if exceptions and c in exceptions:
            out.append(exceptions[c])
        elif "A" <= c <= "Z" and upper_base:
            out.append(chr(upper_base + ord(c) - ord("A")))
        elif "a" <= c <= "z" and lower_base:
            out.append(chr(lower_base + ord(c) - ord("a")))
        else:
            out.append(c)
    return "".join(out)


def to_math_alphanumeric(match):
    """
    OMMLのランの書式（斜体・太字・白抜き）を，PowerPointと同様にUnicodeの数学用英数字で表す．
    斜体は変数（スカラー），太字・非斜体はベクトル・行列，白抜きは集合に用いる．

    Args:
        match (re.Match): <m:r> 要素のマッチ（group(1): m:rPr，group(2): 文字列）．

    Returns:
        str: 置き換えた <m:r> 要素．
    """
    rpr, text = match.group(1) or "", match.group(2)
    if "double-struck" in rpr:
        return f"<m:r><m:t>{map_letters(text, 0x1D538, 0x1D552, DOUBLE_STRUCK_EXCEPTIONS)}</m:t></m:r>"
    if 'm:val="bi"' in rpr or 'm:val="b"' in rpr:
        return f'<m:r><m:rPr><m:sty m:val="b"/></m:rPr><m:t>{map_letters(text, 0x1D400, 0x1D41A)}</m:t></m:r>'
    if 'm:val="p"' in rpr:
        return f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>{text}</m:t></m:r>'
    return f"<m:r><m:t>{map_letters(text, 0x1D434, 0x1D44E, {'h': chr(0x210E)})}</m:t></m:r>"


# ===================== DrawingML の部品 =====================

def run(text, size=2800, bold=False, color=None):
    """
    テキストのランを作る．

    Args:
        text (str): 文字列．
        size (int): 文字サイズ（1/100ポイント）．
        bold (bool): 太字にするか．
        color (str | None): 文字色（16進）．Noneでテーマの既定色．

    Returns:
        str: a:r 要素．
    """
    fill = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else ""
    b = ' b="1"' if bold else ""
    return (f'<a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="{size}"{b} dirty="0">{fill}{FONT}</a:rPr>'
            f'<a:t>{escape(text)}</a:t></a:r>')


def math_run(latex, size=2800):
    """
    文中の数式のランを作る．

    Args:
        latex (str): LaTeXの数式．
        size (int): 文字サイズ（1/100ポイント）．

    Returns:
        str: a14:m 要素．
    """
    return f"<a14:m>{latex_to_omml(latex, False, size)}</a14:m>"


def para(*runs, align="l", size=2800, space_before=0):
    """
    段落を作る．runs には run()/math_run() の戻り値，または文字列（既定書式のラン）を渡す．

    Args:
        *runs (str): ラン．
        align (str): 揃え（"l"，"ctr"，"r"）．
        size (int): 空段落の文字サイズ．
        space_before (int): 段落前の間隔（1/100ポイント）．

    Returns:
        str: a:p 要素．
    """
    body = "".join(r if r.startswith("<") else run(r, size) for r in runs)
    spc = f'<a:spcBef><a:spcPts val="{space_before}"/></a:spcBef>' if space_before else ""
    return (f'<a:p><a:pPr marL="0" indent="0" algn="{align}">{spc}<a:buNone/></a:pPr>{body}'
            f'<a:endParaRPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="{size}" dirty="0">{FONT}</a:endParaRPr></a:p>')


def math_para(latex, size=2800):
    """
    独立した数式の段落を作る．

    Args:
        latex (str): LaTeXの数式．
        size (int): 文字サイズ（1/100ポイント）．

    Returns:
        str: a:p 要素．
    """
    return (f'<a:p><a:pPr marL="0" indent="0" algn="ctr"><a:buNone/></a:pPr><a14:m>{latex_to_omml(latex, True, size)}</a14:m>'
            f'<a:endParaRPr lang="en-US" altLang="ja-JP" sz="{size}" dirty="0"/></a:p>')


class Slide:
    """1枚のスライドの図形と画像の関係を保持し，slideN.xml を組み立てる．"""

    def __init__(self, title, layout=2):
        """
        Args:
            title (str): スライドのタイトル．
            layout (int): レイアウト番号（1: タイトル，2: タイトルとコンテンツ）．
        """
        self.title, self.layout = title, layout
        self.shapes, self.images, self.next_id = [], [], 10

    def _id(self):
        """
        図形IDを払い出す．

        Returns:
            int: 未使用の図形ID．
        """
        self.next_id += 1
        return self.next_id

    def text(self, x, y, w, h, paras, anchor="t", name="テキスト"):
        """
        テキストボックスを追加する（数式を含む場合はmc:AlternateContentで包む）．

        Args:
            x, y, w, h (float): 位置と大きさ（インチ）．
            paras (list[str]): 段落のリスト．
            anchor (str): 縦方向の揃え（"t"，"ctr"，"b"）．
            name (str): 図形名．

        Returns:
            None
        """
        sid = self._id()
        body = "".join(paras)
        sp = (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
              f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
              '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
              f'<p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720" anchor="{anchor}"><a:noAutofit/></a:bodyPr>'
              f'<a:lstStyle/>{body}</p:txBody></p:sp>')
        self.shapes.append(self._wrap_math(sp) if "<a14:m>" in sp else sp)

    @staticmethod
    def _wrap_math(sp):
        """
        数式を含む図形を mc:AlternateContent で包む（a14非対応の環境では数式を除いた図形を表示）．

        Args:
            sp (str): p:sp 要素．

        Returns:
            str: mc:AlternateContent 要素．
        """
        fallback = re.sub(r"<a14:m>.*?</a14:m>", "", sp, flags=re.S)
        return (f'<mc:AlternateContent xmlns:mc="{NS_MC}"><mc:Choice xmlns:a14="{NS_A14}" Requires="a14">{sp}</mc:Choice>'
                f'<mc:Fallback>{fallback}</mc:Fallback></mc:AlternateContent>')

    def box(self, x, y, w, h, paras, fill="FFFFFF", line="tx1", geom="roundRect", anchor="t", line_w=19050):
        """
        枠付きの図形（角丸四角形等）を追加する．

        Args:
            x, y, w, h (float): 位置と大きさ（インチ）．
            paras (list[str]): 段落のリスト．
            fill (str): 塗りつぶし色（16進）または "none"．
            line (str): 枠線の色（16進，"tx1"，"accent1"）または "none"．
            geom (str): 図形の種類（prstGeom）．
            anchor (str): 縦方向の揃え．
            line_w (int): 枠線の太さ（EMU）．

        Returns:
            None
        """
        sid = self._id()
        fill_xml = "<a:noFill/>" if fill == "none" else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        if line == "none":
            ln = "<a:ln><a:noFill/></a:ln>"
        else:
            color = f'<a:schemeClr val="{line}"/>' if line in ("tx1", "accent1") else f'<a:srgbClr val="{line}"/>'
            ln = f'<a:ln w="{line_w}"><a:solidFill>{color}</a:solidFill></a:ln>'
        adj = '<a:gd name="adj" fmla="val 8000"/>' if geom == "roundRect" else ""
        sp = (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="図形 {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
              f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
              f'<a:prstGeom prst="{geom}"><a:avLst>{adj}</a:avLst></a:prstGeom>{fill_xml}{ln}</p:spPr>'
              f'<p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="91440" rIns="91440" bIns="91440" anchor="{anchor}"><a:noAutofit/></a:bodyPr>'
              f'<a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')
        self.shapes.append(self._wrap_math(sp) if "<a14:m>" in sp else sp)

    def arrow(self, x1, y1, x2, y2, color="tx1", width=28575):
        """
        矢印（直線コネクタ）を追加する．

        Args:
            x1, y1 (float): 始点（インチ）．
            x2, y2 (float): 終点（インチ）．
            color (str): 線の色（"tx1" 等のテーマ色）．
            width (int): 線の太さ（EMU）．

        Returns:
            None
        """
        sid = self._id()
        flip = (' flipH="1"' if x2 < x1 else "") + (' flipV="1"' if y2 < y1 else "")
        self.shapes.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="矢印 {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm{flip}><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/>'
            f'<a:ext cx="{emu(abs(x2 - x1))}" cy="{emu(abs(y2 - y1))}"/></a:xfrm>'
            '<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>'
            f'<a:ln w="{width}"><a:solidFill><a:schemeClr val="{color}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln>'
            '</p:spPr></p:cxnSp>')

    def image(self, path, x, y, w=None, h=None):
        """
        画像を追加する（w，hの一方のみ指定した場合は縦横比を保つ）．

        Args:
            path (str): PNG画像のパス．
            x, y (float): 位置（インチ）．
            w, h (float | None): 幅・高さ（インチ）．

        Returns:
            tuple[float, float]: 配置した幅と高さ（インチ）．
        """
        iw, ih = Image.open(path).size
        if w is None:
            w = h * iw / ih
        elif h is None:
            h = w * ih / iw
        sid = self._id()
        rid = f"rId{len(self.images) + 2}"
        self.images.append((rid, path))
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="図 {sid}" descr="{escape(os.path.basename(path))}"/>'
            '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')
        return w, h

    def table(self, x, y, col_widths, rows, size=2000, row_h=0.45, highlight=()):
        """
        表を追加する（1行目を見出し行とし，テンプレートの表スタイルを用いる）．

        Args:
            x, y (float): 位置（インチ）．
            col_widths (list[float]): 列幅（インチ）．
            rows (list[list[str]]): セルの文字列（1行目が見出し）．$ で囲んだセルはLaTeXの数式として扱う．
            size (int): 文字サイズ（1/100ポイント）．
            row_h (float): 行の高さ（インチ）．
            highlight (iterable[tuple[int, int]]): 赤字・太字で強調するセルの (行, 列)．

        Returns:
            None
        """
        sid = self._id()
        grid = "".join(f'<a:gridCol w="{emu(c)}"/>' for c in col_widths)
        trs = []
        for i, row in enumerate(rows):
            tcs = []
            for j, cell in enumerate(row):
                emph = (i, j) in highlight
                if cell.startswith("$"):
                    r = math_run(cell.strip("$"), size)
                    if i == 0:  # 見出し行の数式は他の見出しと同じ白色にする
                        r = r.replace("<a:latin", '<a:solidFill><a:schemeClr val="bg1"/></a:solidFill><a:latin')
                else:
                    r = run(cell, size, bold=emph or i == 0, color=RED if emph else None) if cell else ""
                tcs.append(f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:pPr algn="ctr"/>{r}'
                           f'<a:endParaRPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="{size}" dirty="0"/></a:p></a:txBody>'
                           '<a:tcPr anchor="ctr"/></a:tc>')
            trs.append(f'<a:tr h="{emu(row_h)}">{"".join(tcs)}</a:tr>')
        frame = (
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{sid}" name="表 {sid}"/>'
            '<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(sum(col_widths))}" cy="{emu(row_h * len(rows))}"/></p:xfrm>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
            '<a:tblPr firstRow="1" bandRow="1"><a:tableStyleId>{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}</a:tableStyleId></a:tblPr>'
            f'<a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')
        self.shapes.append(self._wrap_math(frame) if "<a14:m>" in frame else frame)

    def refs(self, *numbers):
        """
        スライド下部に引用文献を記載する．

        Args:
            *numbers (int): REFS の番号．

        Returns:
            None
        """
        n = len(numbers)
        self.text(0.14, 7.0 - 0.27 * n - 0.05, 12.9, 0.27 * n + 0.05, [para(run(REFS[k], 1200)) for k in numbers],
                  anchor="b", name="引用")

    def xml(self, number):
        """
        slideN.xml の内容を組み立てる．

        Args:
            number (int): スライド番号．

        Returns:
            str: スライドのXML．
        """
        num = (f'<p:sp><p:nvSpPr><p:cNvPr id="3" name="スライド番号プレースホルダー 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
               '<p:nvPr><p:ph type="sldNum" sz="quarter" idx="12"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>'
               '<a:p><a:fld id="{9C0FEE9A-3EB6-844C-8184-5D808DF01412}" type="slidenum">'
               f'<a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" smtClean="0"/><a:t>{number}</a:t></a:fld></a:p></p:txBody></p:sp>')
        if self.layout == 1:
            title = self.shapes.pop(0)
        else:
            title = ('<p:sp><p:nvSpPr><p:cNvPr id="2" name="タイトル 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
                     '<p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr><a:normAutofit fontScale="90000"/></a:bodyPr>'
                     f'<a:lstStyle/><a:p><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>{escape(self.title)}</a:t></a:r></a:p></p:txBody></p:sp>')
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree>'
                '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
                '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
                f'{title}{num}{"".join(self.shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')

    def rels(self, media_names):
        """
        slideN.xml.rels の内容を組み立てる．

        Args:
            media_names (dict): 画像パス → ppt/media 内のファイル名．

        Returns:
            str: リレーションシップのXML．
        """
        base = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        items = [f'<Relationship Id="rId1" Type="{base}/slideLayout" Target="../slideLayouts/slideLayout{self.layout}.xml"/>']
        items += [f'<Relationship Id="{rid}" Type="{base}/image" Target="../media/{media_names[p]}"/>' for rid, p in self.images]
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(items)}</Relationships>')


# ===================== スライド用の画像 =====================

def make_media():
    """
    論文の図を複製し，スライド専用の画像（YOLOと提案手法の検出結果の例）を作成する．

    Returns:
        dict: 画像名 → パス．
    """
    os.makedirs(MEDIA_DIR, exist_ok=True)
    media = {name: os.path.join(PAPER_FIG, f"{name}.png") for name in
             ["fig_vlm_comparison", "fig_pipeline_accuracy", "fig_depth_confidence", "fig_crop_preprocess",
              "fig_open_vocab", "fig_coco_qualitative", "fig_yolo_comparison", "fig_yolo_open_vocab",
              "fig_hybrid_qualitative", "fig_stopping"]}
    data_dir = os.path.join(ROOT, "datasets", "ex004_open_vocab_qualitative")
    yolo = json.load(open(os.path.join(ROOT, "outputs", "ex005_coco_val500_yolo_comparison", "yolo11x",
                                       "open_vocab_conf0.25", "42", "annotations.json")))
    prop = json.load(open(os.path.join(ROOT, "outputs", "ex004_open_vocab_qualitative", "CLIP-ViT-B32",
                                       "open_vocab_detr_threshold0.5_crop_pad", "42", "annotations.json")))
    for name in ["camel", "kangaroo"]:
        image = Image.open(os.path.join(data_dir, f"{name}.jpg")).convert("RGB")
        labels = [f"{t[0]} ({c[0]:.2f})" for t, c in zip(yolo[name]["top5"], yolo[name]["top5_conf"])]
        media[f"yolo_{name}"] = draw_boxes(image, yolo[name]["boxes"], labels, f"yolo_{name}.png", max_boxes=1)
        regions = prop[name]["regions"]
        labels = [" → ".join(f"{s['word']} ({s['conf']:.2f})" for s in r["chain"]) for r in regions]
        media[f"prop_{name}"] = draw_boxes(image, [r["box"] for r in regions], labels, f"prop_{name}.png", max_boxes=1)
    media["hybrid_book"] = make_book_crop()
    return media


def make_book_crop():
    """
    ハイブリッド構成で深さ0の切替が誤った領域（COCO 262938の本，YOLOの検出スコア0.30）の周辺を切り出して保存する．

    Returns:
        str: 保存したパス．
    """
    ann = json.load(open(os.path.join(ROOT, "outputs", "ex006_coco_val500_yolo_clip_hybrid", "hybrid_switch",
                                      "coco_qualitative_conf0.25", "42", "annotations.json")))
    box = next(r["box"] for r in ann["coco_dog_262938"]["regions"] if r["chain"][0]["word"] == "parrot")
    image = Image.open(os.path.join(ROOT, "datasets", "coco", "val2017", "000000262938.jpg")).convert("RGB")
    left, top = max(0, box[0] - 120), 0
    crop = image.crop((int(left), top, image.size[0], int(min(image.size[1], box[3] + 120))))
    shifted = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
    fig, ax = plt.subplots(figsize=(4, 4 * crop.size[1] / crop.size[0]))
    ax.imshow(crop)
    ax.axis("off")
    ax.add_patch(plt.Rectangle(shifted[:2], shifted[2] - shifted[0], shifted[3] - shifted[1], fill=False, edgecolor="red", lw=4))
    path = os.path.join(MEDIA_DIR, "hybrid_book.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


def draw_boxes(image, boxes, labels, file_name, max_boxes=3):
    """
    画像にボックスとラベルを大きな文字で描画して保存する．

    Args:
        image (PIL.Image.Image): 画像．
        boxes (list[list[float]]): xyxy形式のボックス．
        labels (list[str]): ラベル．
        file_name (str): 保存するファイル名．
        max_boxes (int): 描画するボックスの最大数（先頭から）．

    Returns:
        str: 保存したパス．
    """
    fig, ax = plt.subplots(figsize=(6, 6 * image.size[1] / image.size[0]))
    ax.imshow(image)
    ax.axis("off")
    for box, label in list(zip(boxes, labels))[:max_boxes]:
        ax.add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor="red", lw=4))
        ax.text(box[0] + 4, box[1] + 4, label, fontsize=20, color="white", va="top",
                bbox=dict(facecolor="red", alpha=0.85, pad=2, lw=0))
    path = os.path.join(MEDIA_DIR, file_name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


# ===================== スライドの内容 =====================

def bullet(text, size=2800, color=None, bold=False, space_before=600):
    """
    「・」で始まる箇条書きの段落を作る．

    Args:
        text (str): 本文．
        size (int): 文字サイズ（1/100ポイント）．
        color (str | None): 文字色．
        bold (bool): 太字にするか．
        space_before (int): 段落前の間隔．

    Returns:
        str: a:p 要素．
    """
    return para(run("・" + text, size, bold, color), size=size, space_before=space_before)


def note(text, size=2400, color=None, indent=True):
    """
    注釈（インデント1，24ポイント）の段落を作る．

    Args:
        text (str): 本文．
        size (int): 文字サイズ．
        color (str | None): 文字色．
        indent (bool): 先頭に全角空白を入れるか．

    Returns:
        str: a:p 要素．
    """
    return para(run(("　" if indent else "") + text, size, color=color), size=size)


def build_slides(media):
    """
    全スライドを定義する．

    Args:
        media (dict): 画像名 → パス．

    Returns:
        list[Slide]: スライドのリスト．
    """
    slides = []

    # 1. タイトル
    s = Slide("", layout=1)
    s.shapes.append(
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="タイトル 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="ctrTitle"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="125506" y="772319"/><a:ext cx="11940525" cy="2656681"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/>'
        '<a:t>物体領域検出とカテゴリアノテーションの分離に基づく，</a:t></a:r><a:br><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/></a:br>'
        '<a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" sz="3600" dirty="0"/><a:t>WordNetを用いた粒度可変物体アノテーション</a:t></a:r></a:p></p:txBody></p:sp>')
    s.shapes.append(
        '<p:sp><p:nvSpPr><p:cNvPr id="4" name="字幕 3"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="subTitle" idx="1"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="125506" y="4339531"/><a:ext cx="11940524" cy="1210860"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>所属機関名</a:t></a:r></a:p>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr><a:r><a:rPr kumimoji="1" lang="ja-JP" altLang="en-US" dirty="0"/><a:t>○著者名</a:t></a:r></a:p></p:txBody></p:sp>')
    slides.append(s)

    # 2. 研究背景
    s = Slide("研究背景：閉域的な物体検出モデルの限界")
    s.text(0.3, 1.0, 8.2, 4.6, [
        bullet("物体アノテーション：「どこ」＋「何」"),
        bullet("YOLO[1]やDETR[2]は1つのモデルで学習"),
        bullet("出力できるのは学習したクラスのみ"),
        note("例：COCO[7]の80クラス"),
        note("→ 語の追加・粒度の変更には再学習が必要", color=RED, indent=False),
    ])
    w, h = s.image(media["yolo_camel"], 8.7, 1.05, w=4.35)
    s.text(8.7, 1.05 + h + 0.05, 4.35, 0.5, [para(run("YOLO11x[8]：ラクダ → “dog”", 2000), align="ctr")])
    s.refs(1, 2, 7, 8)
    slides.append(s)

    # 3. 従来手法の課題
    s = Slide("従来手法の課題")
    s.box(0.3, 1.0, 6.2, 4.2, [
        para(run("閉域的な物体検出", 2800, True), align="ctr"),
        para(run("YOLO[1]，DETR[2]", 2400), align="ctr"),
        bullet("語彙が学習時に固定", space_before=1200),
        bullet("新しい語には再学習が必要"),
    ])
    s.box(6.83, 1.0, 6.2, 4.2, [
        para(run("オープンボキャブラリ検出", 2800, True), align="ctr"),
        para(run("OWL-ViT[3]，Grounding DINO[4]", 2400), align="ctr"),
        bullet("語彙をテキストで指定できる", space_before=1200),
        bullet("語彙の変更ごとに検出を再実行"),
        note("→ 多段階の語彙では非効率", color=RED),
    ])
    s.refs(1, 2, 3, 4)
    slides.append(s)

    # 4. 研究目的
    s = Slide("研究目的")
    s.text(0.3, 1.0, 12.7, 1.5, [
        para(run("検出とカテゴリ付与を分離し，", 2800), run("再学習なし", 2800, True, RED), run("で語彙と粒度を変える", 2800)),
        note("検出は1回だけ行い，同じ領域で語彙を変えながら繰り返す"),
    ])
    chain = [("animal", "粗い"), ("dog", ""), ("doberman", "細かい")]
    for i, (word, label) in enumerate(chain):
        x = 1.2 + i * 4.0
        s.box(x, 3.2, 3.0, 1.2, [para(run(word, 2800, True), align="ctr")], fill="DDEBF7", line="accent1", anchor="ctr")
        if label:
            s.text(x, 4.5, 3.0, 0.5, [para(run(f"{label}粒度", 2400), align="ctr")])
        if i < 2:
            s.arrow(x + 3.05, 3.8, x + 3.95, 3.8)
    s.text(1.2, 5.3, 11.0, 0.6, [para(run("同じクロップに対し，WordNetの下位語を次の候補語彙にする", 2400), align="ctr")])
    slides.append(s)

    # 5. 提案手法の全体像
    s = Slide("提案手法：検出とカテゴリ付与の分離")
    y = 1.3
    steps = [("画像", 1.4), ("DETR[2]で検出\n（1回のみ）", 2.6), ("クロップ\n（正方形化）", 2.2), ("CLIP[5]で\nゼロショット分類", 2.8)]
    x = 0.3
    xs = []
    for i, (label, w) in enumerate(steps):
        fill = "DDEBF7" if i == 1 else "FFFFFF"
        s.box(x, y, w, 1.4, [para(run(t, 2000, i == 3), align="ctr") for t in label.split("\n")], fill=fill,
              line="accent1", anchor="ctr")
        xs.append((x, w))
        if i < len(steps) - 1:
            s.arrow(x + w + 0.05, y + 0.7, x + w + 0.45, y + 0.7)
        x += w + 0.5
    vx, vw = xs[3]
    s.box(vx, 3.2, vw, 1.2, [para(run("WordNet[6]", 2000, True), align="ctr"), para(run("下位語を次の語彙に", 2000), align="ctr")],
          fill="FFF2CC", line="accent1", anchor="ctr")
    s.arrow(vx + vw * 0.3, 3.15, vx + vw * 0.3, y + 1.45)
    s.arrow(vx + vw * 0.7, y + 1.45, vx + vw * 0.7, 3.15)
    s.box(vx + vw + 0.45, y, 1.75, 1.4, [para(run("animal", 2000), align="l"), para(run("→ dog", 2000), align="l"),
                                         para(run("→ doberman", 2000), align="l")], fill="none", line="none", anchor="ctr")
    s.arrow(vx + vw + 0.05, y + 0.7, vx + vw + 0.4, y + 0.7)
    s.text(0.3, 4.65, 12.7, 1.3, [
        bullet("検出は語彙に依存しない → 語彙や粒度を変えても再検出は不要"),
        bullet("繰り返すのは軽量なVLMの推論のみ"),
    ])
    s.refs(2, 5, 6)
    slides.append(s)

    # 6. カテゴリのアノテーション（数式）
    s = Slide("VLMによるカテゴリのアノテーション")
    s.text(0.3, 0.95, 12.0, 0.6, [para(run("クロップに対するスコアが最大の単語を選ぶ", 2800))])
    s.text(1.0, 1.55, 10.5, 0.9, [math_para(r"\hat{v}_k = \underset{v \in \mathbb{V}}{\mathrm{arg\,max}}\; f(\mathbf{X}_k; \mathbb{V})_v")],
           anchor="ctr")
    s.text(11.8, 1.55, 1.2, 0.9, [para(run("(1)", 2800), align="r")], anchor="ctr")
    s.text(0.3, 2.5, 12.0, 0.6, [para(run("CLIP[5]：プロンプトとの類似度のソフトマックス", 2800))])
    s.text(1.0, 3.1, 10.5, 1.2, [math_para(
        r"f(\mathbf{X}; \mathbb{V})_v = \frac{\exp(\tau \cos(g_I(\mathbf{X}), g_T(t(v))))}{\sum_{u \in \mathbb{V}} \exp(\tau \cos(g_I(\mathbf{X}), g_T(t(u))))}",
        2400)], anchor="ctr")
    s.text(11.8, 3.1, 1.2, 1.2, [para(run("(2)", 2800), align="r")], anchor="ctr")
    s.text(0.3, 4.4, 12.7, 2.0, [
        para(math_run(r"\mathbf{X}_k", 2400), run("：領域", 2400), math_run("k", 2400), run("のクロップ，", 2400),
             math_run(r"\mathbb{V}", 2400), run("：候補語彙集合，", 2400), math_run(r"\hat{v}_k", 2400), run("：推定単語", 2400)),
        para(math_run("g_I", 2400), run("，", 2400), math_run("g_T", 2400), run("：画像・テキストエンコーダ，", 2400),
             math_run(r"t(v)", 2400), run("：“a photo of a {", 2400), math_run("v", 2400), run("}”", 2400), space_before=600),
        para(math_run(r"\tau", 2400), run("：温度パラメータ（100），", 2400), math_run(r"\cos", 2400), run("：コサイン類似度", 2400),
             space_before=600),
    ])
    s.refs(5)
    slides.append(s)

    # 7. クロップの前処理
    s = Slide("クロップの前処理：正方形パディング")
    s.text(0.3, 1.0, 6.8, 4.8, [
        bullet("ボックスに合わせて切り出す"),
        note("周辺の物体の写り込みを防ぐ"),
        bullet("中央切り出しでは端が欠ける"),
        note("VLMの標準の前処理"),
        bullet("黒で正方形にパディング"),
        note("ボックス外の画像内容は含めない"),
    ])
    s.image(media["fig_crop_preprocess"], 7.2, 1.05, w=5.85)
    slides.append(s)

    # 8. WordNetによる粒度可変アノテーション
    s = Slide("WordNetによる粒度可変アノテーション")
    s.text(0.3, 0.95, 12.0, 0.6, [para(run("1つ浅い段階で選ばれた語の下位語を，次の候補語彙にする", 2800))])
    s.text(1.0, 1.55, 10.5, 0.9, [math_para(r"\mathbb{V}_k^{(d)} = \mathrm{Hyponyms}(\mathrm{synset}(\hat{v}_k^{(d-1)}))")],
           anchor="ctr")
    s.text(11.8, 1.55, 1.2, 0.9, [para(run("(3)", 2800), align="r")], anchor="ctr")
    s.text(0.3, 2.5, 6.6, 3.6, [
        para(math_run("d", 2400), run("：深さ（", 2400), math_run(r"\mathbb{V}^{(0)}", 2400), run("はCOCOの80クラス）", 2400)),
        para(run("Hyponyms：直接の下位語の集合", 2400), space_before=600),
        bullet("下位語がなくなるまで(1)を反復", space_before=1800),
        bullet("出力：粗い語→細かい語の列"),
        note("各段階の確信度も出力"),
    ])
    s.image(media["prop_camel"], 7.3, 2.55, w=5.7)
    s.refs(6)
    slides.append(s)

    # 9. 実験設定
    s = Slide("実験設定")
    s.table(0.6, 1.1, [3.0, 9.1], [
        ["項目", "設定"],
        ["データ", "COCO 2017 Val.[7]の500枚（GTインスタンス3206件）"],
        ["物体領域検出", "DETR[2]（ResNet-50），検出スコアの閾値 θ"],
        ["VLM", "CLIP-ViT-B/32[5]（比較：ResNet50-IN1k，CLIP-RN50）"],
        ["比較する検出器", "OWL-ViT[3]，Grounding DINO[4]，YOLO11 / YOLOv8[8]"],
        ["評価指標", "Top-1/Top-5正解率，P@k / R@k，IoU，正命名率"],
        ["学習", "なし（すべて学習済みモデルの推論のみ）"],
    ], size=2000, row_h=0.55)
    s.refs(2, 3, 4, 5, 7, 8)
    slides.append(s)

    # 10. 実験1
    s = Slide("実験1：オープンボキャブラリなVLMの必要性")
    s.image(media["fig_vlm_comparison"], 0.4, 0.95, h=5.5)
    s.text(6.3, 1.0, 6.8, 5.0, [
        bullet("画像単位の単語推定（500枚）"),
        note("タスクA：COCOの80クラス"),
        note("タスクB：キャプション由来の1371語"),
        para(run("・タスクAのP@1：", 2800), run("0.216 → 0.855", 2800, True, RED), space_before=1200),
        para(run("・全指標で有意差（", 2800), math_run(r"p < 10^{-11}", 2800), run("）", 2800), space_before=600),
        bullet("CLIP-RN50とViT-B/32の差は小さい"),
    ])
    slides.append(s)

    # 11. 実験2
    s = Slide("実験2：物体領域検出モデルの比較")
    s.text(0.3, 0.95, 12.7, 0.6, [para(run("DETRの位置検出精度は他の検出器と同程度", 2800))])
    s.table(1.4, 1.75, [3.3, 1.3, 2.0, 2.3, 2.6], [
        ["手法", "閾値", "平均提案数", "Recall@IoU0.5", "平均最良IoU"],
        ["DETR[2]", "0.5", "2.25", "0.800", "0.705"],
        ["DETR[2]", "0.9", "1.50", "0.600", "0.583"],
        ["OWL-ViT[3]", "0.1", "1.50", "0.500", "0.517"],
        ["Grounding DINO[4]", "0.35", "1.50", "0.500", "0.546"],
        ["YOLOv8x[8]", "0.25", "2.25", "0.800", "0.767"],
    ], size=2000, row_h=0.55, highlight=[(1, 3), (1, 4)])
    s.text(0.3, 5.15, 12.7, 1.0, [
        note("4画像・GTインスタンス10件．極小の車や画像全体を覆う食卓は全手法で未検出", indent=False),
    ])
    s.refs(2, 3, 4, 8)
    slides.append(s)

    # 12. 実験3
    s = Slide("実験3：パイプライン全体のカテゴリアノテーション精度")
    s.image(media["fig_pipeline_accuracy"], 0.3, 1.0, w=7.0)
    for i, (label, value) in enumerate([("Top-1正解率", "0.604"), ("Top-5正解率", "0.894")]):
        s.box(7.7 + i * 2.75, 1.1, 2.55, 1.9, [para(run(value, 4400, True, RED), align="ctr"), para(run(label, 2000), align="ctr")],
              fill="FFFFFF", line="accent1", anchor="ctr")
    s.text(7.6, 3.2, 5.5, 3.0, [
        note("CLIP-ViT-B/32，θ=0.97，pad", indent=False),
        bullet("padでTop-1が約0.08向上"),
        note("bottle：0.344 → 0.844"),
        bullet("人物は約0.38にとどまる"),
    ])
    slides.append(s)

    # 13. 実験4
    s = Slide("実験4：WordNet階層の深さごとの確信度")
    s.image(media["fig_depth_confidence"], 0.3, 1.0, w=6.9)
    s.text(7.4, 1.0, 5.7, 5.4, [
        para(run("・深さ0→1：", 2800), run("0.642 → 0.442", 2800, True, RED)),
        note("候補が80語→平均109語に増加"),
        bullet("下位語が1語の分岐が18%", space_before=1200),
        note("確信度は常に1（ソフトマックス）"),
        note("除いた深さ1〜3の確信度"),
        note("0.378，0.571，0.481"),
    ])
    slides.append(s)

    # 14. 実験5
    s = Slide("実験5：COCO以外の語を含む実画像")
    s.image(media["fig_open_vocab"], 0.3, 0.95, h=4.45)
    s.text(0.3, 5.45, 12.7, 1.1, [
        para(run("・penguin，helicopter，camel，kangarooを", 2400), run("再学習なしで正しく付与", 2400, True, RED)),
        note("camel → Arabian camel（0.78）．灯台の塔はDETRで検出できない", indent=False),
    ])
    slides.append(s)

    # 15. 定性結果と限界
    s = Slide("WordNetの階層をそのまま使うことの限界")
    s.image(media["fig_coco_qualitative"], 0.3, 1.0, w=12.7)
    s.text(0.3, 4.75, 12.7, 1.8, [
        para(run("・giraffe → okapi（1.00）：下位語が1語しかない", 2800)),
        bullet("person → suspect など，外見で判断できない語も選ばれる"),
    ])
    slides.append(s)

    # 16. YOLOとの比較（COCOの80クラス）
    s = Slide("YOLOとの比較①：COCOの80クラスの範囲内")
    s.table(0.3, 1.05, [3.0, 1.05, 1.15, 1.25], [
        ["手法", "n", "Top-1", "Top-5"],
        ["提案手法（θ=0.97）", "1519", "0.604", "0.894"],
        ["YOLO11x[8]", "2210", "0.981", "0.999"],
        ["YOLO11x + CLIP", "1474", "0.604", "0.907"],
    ], size=2000, row_h=0.55, highlight=[(2, 2)])
    s.text(0.3, 3.4, 6.7, 3.0, [
        bullet("80クラス内はYOLOが高精度"),
        note("共通1480件：0.989 vs 0.609"),
        bullet("YOLOの領域をCLIPで分類"),
        note("→ 提案手法と同等（0.604）"),
        note("差の原因は検出器ではなく分類器", color=RED),
    ])
    s.image(media["fig_yolo_comparison"], 7.2, 1.0, w=5.9)
    s.refs(8)
    slides.append(s)

    # 17. YOLOとの比較（COCO外）
    s = Slide("YOLOとの比較②：COCO以外の語")
    s.table(0.3, 1.05, [2.3, 2.6, 3.4], [
        ["被写体", "YOLO11x", "提案手法"],
        ["ペンギン", "bird", "penguin"],
        ["ラクダ", "dog", "camel"],
        ["カンガルー", "bear，sheep", "kangaroo"],
        ["ヘリコプター", "airplane", "helicopter"],
        ["灯台", "検出なし", "検出なし"],
    ], size=2000, row_h=0.55, highlight=[(1, 2), (2, 2), (3, 2), (4, 2)])
    s.text(0.3, 4.6, 8.3, 1.8, [
        bullet("YOLOは80クラスの語しか出力できない"),
        bullet("提案手法は語彙を変えるだけで対応できる"),
    ])
    w, h = s.image(media["yolo_camel"], 9.2, 1.05, w=3.6)
    s.text(9.2, 1.05 + h, 3.6, 0.42, [para(run("YOLO11x", 2000), align="ctr")])
    _, h2 = s.image(media["prop_camel"], 9.2, 1.55 + h, w=3.6)
    s.text(9.2, 1.55 + h + h2, 3.6, 0.42, [para(run("提案手法", 2000), align="ctr")])
    s.refs(8)
    slides.append(s)

    # 18. ハイブリッド構成の考え方
    s = Slide("ハイブリッド構成：深さ0の分類をYOLOに任せる")
    y = 1.15
    steps = [("画像", 1.3, "FFFFFF"), ("YOLO11x[8]\n検出＋80クラス", 2.9, "DDEBF7"), ("深さ0の単語\n（切替規則）", 2.6, "FFFFFF"),
             ("CLIP[5]＋WordNet[6]\n深さ1以降を細分化", 3.6, "FFF2CC")]
    x = 0.3
    for i, (label, w, fill) in enumerate(steps):
        s.box(x, y, w, 1.4, [para(run(t, 2000, j == 0 and i > 0), align="ctr") for j, t in enumerate(label.split("\n"))],
              fill=fill, line="accent1", anchor="ctr")
        if i < len(steps) - 1:
            s.arrow(x + w + 0.05, y + 0.7, x + w + 0.45, y + 0.7)
        x += w + 0.5
    s.text(0.3, 2.85, 12.7, 0.6, [para(run("例：YOLO: dog（0.92）→ CLIP: basenji（0.72）", 2400), align="ctr")])
    s.text(0.3, 3.6, 12.7, 2.7, [
        bullet("80クラス内の分類はYOLOが高精度（0.981）"),
        bullet("深さ0の誤りが減る → 誤った分岐をたどりにくい"),
        bullet("細分化はこれまでと同じく再学習なし"),
    ])
    s.refs(5, 6, 8)
    slides.append(s)

    # 19. COCO以外の語への切替規則
    s = Slide("COCO以外の語への切替規則と閾値γ")
    s.text(0.3, 0.95, 12.7, 0.6, [para(run("CLIPがCOCO外の語を高い確信度で選んだときのみ切り替える", 2800))])
    s.text(0.6, 1.6, 11.0, 1.5, [math_para(
        r"\hat{v}_k^{(0)} = \begin{cases} u_k & (u_k \in \mathbb{V}_E,\; f(\mathbf{X}_k; \mathbb{V}^{(0)} \cup \mathbb{V}_E)_{u_k} \geq \gamma) \\ y_k & (\mathrm{otherwise}) \end{cases}",
        2400)], anchor="ctr")
    s.text(11.8, 1.6, 1.2, 1.5, [para(run("(4)", 2800), align="r")], anchor="ctr")
    s.text(0.3, 3.2, 12.7, 1.6, [
        para(math_run("y_k", 2400), run("：YOLOのクラス，", 2400), math_run(r"\mathbb{V}_E", 2400), run("：COCO外の22語，", 2400),
             math_run(r"\gamma", 2400), run("：切替に必要な確信度の下限", 2400)),
        para(math_run("u_k", 2400), run("：80クラスと22語の計102語を候補として(1)で推定した単語", 2400), space_before=600),
    ])
    s.table(1.4, 4.6, [2.2, 2.9, 2.9, 2.5], [
        ["構成", "検証用の500枚", "テスト用の500枚", "COCO外の画像"],
        ["γ=0", "47件を誤って切替", "50件を誤って切替", "切り替わる"],
        ["γ=0.5", "2件", "8件（6件は下位語）", "切り替わる"],
    ], size=2000, row_h=0.5, highlight=[(2, 1), (2, 2), (2, 3)])
    s.text(1.4, 6.15, 10.5, 0.5, [para(run("下位語：bird → flamingo など，WordNetでGTカテゴリの下位語となる切替", 2000))])
    slides.append(s)

    # 20. ハイブリッド構成の定量結果（評価数 n が近い組で比較）
    s = Slide("ハイブリッド構成の結果：細分化できる物体が約1.6倍")
    s.table(0.9, 1.0, [4.4, 1.3, 1.3, 1.5, 1.5, 1.5], [
        ["手法", "閾値", "$n$", "Top-1", "$r$", "$N_h$"],
        ["提案手法（DETR）", "θ=0.5", "2343", "0.560", "0.410", "1313"],
        ["ハイブリッド", "0.25", "2210", "0.981", "0.676", "2167"],
        ["提案手法（DETR）", "θ=0.97", "1519", "0.604", "0.286", "917"],
        ["ハイブリッド", "0.7", "1474", "0.993", "0.456", "1463"],
    ], size=2000, row_h=0.5, highlight=[(2, 5), (4, 5)])
    s.text(0.3, 3.7, 12.7, 2.7, [
        para(run("・評価数", 2800), math_run("n", 2800), run("が近い組で比較：", 2800), run("1.65倍，1.60倍", 2800, True, RED)),
        note("検出スコアの定義が異なるため，閾値の値では揃えない"),
        para(run("・", 2800), math_run("N_h", 2800), run("：深さ0が正解の物体数，", 2800), math_run("r", 2800),
             run("：GT 3206件に対する割合", 2800), space_before=600),
    ])
    s.refs(8)
    slides.append(s)

    # 21. ハイブリッド構成の定性結果
    s = Slide("ハイブリッド構成の定性結果：COCO外の語も細分化")
    s.image(media["fig_hybrid_qualitative"], 0.3, 0.95, h=4.45)
    s.text(0.3, 5.45, 12.7, 1.1, [
        para(run("・bird / dog → ", 2400), run("CLIP: penguin（0.99）/ camel（0.99）", 2400, True, RED),
             run(" に切替後，細分化", 2400)),
        note("YOLOが検出しない灯台や，giraffe → okapi の分岐は解消されない", indent=False),
    ])
    slides.append(s)

    # 22. 限界：深さ0の誤りの展開
    s = Slide("限界：深さ0の誤りはそのまま下位語へ展開される")
    s.image(media["hybrid_book"], 0.3, 1.0, h=3.2)
    s.text(0.3, 4.25, 3.4, 0.45, [para(run("YOLO: book（0.30）", 2000), align="ctr")])
    chains = [("γ=0", [("parrot", "0.18", True), ("parakeet", "0.25", False), ("budgerigar", "0.78", False)]),
              ("γ=0.5", [("book", "0.30", False), ("journal", "0.34", False), ("diary", "0.78", False)])]
    for i, (label, chain) in enumerate(chains):
        cy = 1.1 + i * 1.6
        s.text(3.9, cy, 1.3, 1.0, [para(run(label, 2400, True), align="ctr")], anchor="ctr")
        for j, (word, conf, wrong) in enumerate(chain):
            bx = 5.3 + j * 2.6
            s.box(bx, cy, 2.2, 1.0, [para(run(word, 2000, True, RED if wrong else None), align="ctr"),
                                      para(run(f"（{conf}）", 2000), align="ctr")],
                  fill="FCE4E4" if i == 0 else "DDEBF7", line="accent1", anchor="ctr")
            if j < 2:
                s.arrow(bx + 2.25, cy + 0.5, bx + 2.55, cy + 0.5)
    s.text(3.9, 4.25, 9.2, 2.2, [
        bullet("誤った語の下位語の中から選ぶ", space_before=0),
        note("→ 深い段階の確信度は高く見える（0.78）"),
        bullet("COCOの評価は深さ0が正解の物体のみ集計"),
        note("誤りの展開は表の数値に含まれない"),
    ])
    slides.append(s)

    # 23. 別の500枚による検証と誤りの伝播（テスト用の500枚）
    s = Slide("誤った枝は深い段階の確信度では見分けられない")
    s.table(0.6, 1.0, [3.8, 2.4, 2.9, 2.9], [
        ["テスト用の500枚", "閾値", "提案手法", "ハイブリッド"],
        ["細分化できる物体数", "θ=0.5 / 0.25", "1509", "2492（1.65倍）"],
        ["誤った枝に展開", "θ=0.5 / 0.25", "44.8%", "3.1%"],
        ["AUROC：最深段の確信度", "θ=0.5 / 0.25", "0.611", "0.418"],
        ["AUROC：深さ0の確信度", "θ=0.5 / 0.25", "0.725", "0.715"],
    ], size=2000, row_h=0.5, highlight=[(1, 3), (4, 2), (4, 3)])
    s.text(0.3, 3.75, 12.7, 2.7, [
        bullet("閾値やγを決めた500枚とは別の500枚で再評価", space_before=0),
        bullet("誤った枝の最深段の確信度は正しい枝と同程度"),
        note("→ 誤りの手がかりは深さ0の確信度のみ", color=RED),
    ])
    slides.append(s)

    # 24. LVISによる深さ1以降の評価
    s = Slide("LVISによる評価：深さ1の正解は半数未満")
    s.text(0.3, 0.95, 12.7, 1.6, [
        bullet("LVIS[9]：COCOの画像に1203カテゴリ（WordNetのSynset付き）", space_before=0),
        note("COCOより細かい正解の例：bird → pigeon，bottle → wine bottle"),
    ])
    s.table(0.9, 2.65, [4.6, 3.3, 3.3], [
        ["LVISのテスト用分割", "提案手法（θ=0.5）", "ハイブリッド（0.25）"],
        ["評価インスタンス数", "489", "481"],
        ["深さ1が正しい割合", "0.440", "0.407"],
        ["LVISの語まで到達", "0.272", "0.372"],
    ], size=2000, row_h=0.5, highlight=[(2, 1), (2, 2)])
    s.text(0.3, 4.9, 12.7, 1.4, [
        bullet("深さ0が正しい物体のみで集計しても半数未満", space_before=0),
        note("深さ1での確信度の低下（0.642→0.442）は実際の誤りを伴う"),
    ])
    s.refs(9)
    slides.append(s)

    # 25. 累積確信度による細分化の停止
    s = Slide("累積確信度で細分化を止める")
    s.text(0.3, 0.95, 6.9, 1.2, [math_para(r"c_k^{(d)} = \prod_{j=0}^{d} f(\mathbf{X}_k; \mathbb{V}_k^{(j)})_{\hat{v}_k^{(j)}} \geq \tau", 2400)],
           anchor="ctr")
    s.text(6.3, 0.95, 0.9, 1.2, [para(run("(5)", 2400), align="r")], anchor="ctr")
    s.text(0.3, 2.2, 6.9, 4.3, [
        para(run("・満たす最も深い段階まで出力", 2400)),
        para(run("　", 2400), math_run(r"c_k^{(0)} < \tau", 2400), run(" なら棄却", 2400)),
        para(run("・τは検証用の分割で決定", 2400), space_before=600),
        para(run("・テスト用：", 2400), math_run("hP", 2400), run(" ", 2400), run("0.594 → 0.801", 2400, True, RED)),
        note("ハイブリッド構成，τ=0.3"),
        para(run("・深さ1で止める規則と", 2400), math_run("hF", 2400), run("は同程度", 2400), space_before=600),
    ])
    s.image(media["fig_stopping"], 7.4, 1.0, w=5.6)
    slides.append(s)

    # 26. まとめ
    s = Slide("まとめと今後の課題")
    s.box(0.3, 1.0, 6.25, 5.3, [
        para(run("まとめ", 2800, True), align="ctr"),
        para(run("1. 検出とカテゴリ付与の分離", 2400), space_before=700),
        para(run("2. 再学習なしの粒度可変化", 2400), space_before=700),
        para(run("3. Top-1 0.604，Top-5 0.894", 2400), space_before=700),
        para(run("4. COCO外の語も付与できる", 2400), space_before=700),
        para(run("5. 80クラス内はYOLOが優位", 2400), space_before=700),
        para(run("6. YOLOとの組み合わせで", 2400), space_before=700),
        para(run("　 細分化できる物体が約1.6倍", 2400)),
        para(run("7. 深さ1の正解は半数未満", 2400), space_before=700),
    ])
    s.box(6.8, 1.0, 6.25, 5.3, [
        para(run("今後の課題", 2800, True), align="ctr"),
        para(run("1. WordNet階層の選別", 2400), space_before=700),
        para(run("2. 深い段階の確信度の改善", 2400), space_before=700),
        para(run("3. 人物クラスの細分化の評価", 2400), space_before=700),
        para(run("4. 語彙外の領域の棄却", 2400), space_before=700),
        para(run("5. 検出の学習分布依存の緩和", 2400), space_before=700),
    ])
    slides.append(s)
    return slides


# ===================== パッケージの組み立て =====================

def assemble(slides, media):
    """
    テンプレートを展開してスライドを差し替え，pptxとして保存する．

    Args:
        slides (list[Slide]): スライド．
        media (dict): 画像名 → パス．

    Returns:
        None
    """
    work = tempfile.mkdtemp()
    zipfile.ZipFile(TEMPLATE).extractall(work)
    add_slide = os.path.join(SKILL_SCRIPTS, "add_slide.py")
    new_files = []
    for s in slides:
        out = subprocess.run([sys.executable, add_slide, work, f"slideLayout{s.layout}.xml"], capture_output=True, text=True, check=True)
        new_files.append(re.search(r"(slide\d+\.xml)", out.stdout).group(1))
    # sldIdLst を新しいスライドのみにする
    pres_path = os.path.join(work, "ppt", "presentation.xml")
    rels = open(os.path.join(work, "ppt", "_rels", "presentation.xml.rels")).read()
    rid_of = {t: r for r, t in re.findall(r'Id="(rId\d+)"[^>]*Target="slides/(slide\d+\.xml)"', rels)}
    rid_of.update({t: r for t, r in re.findall(r'Target="slides/(slide\d+\.xml)"[^>]*Id="(rId\d+)"', rels)})
    pres = open(pres_path).read()
    ids = re.findall(r'<p:sldId [^>]*/>', pres)
    keep = [next(i for i in ids if f'r:id="{rid_of[f]}"' in i) for f in new_files]
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", "<p:sldIdLst>" + "".join(keep) + "</p:sldIdLst>", pres, flags=re.S)
    open(pres_path, "w").write(pres)
    subprocess.run([sys.executable, os.path.join(SKILL_SCRIPTS, "clean.py"), work], check=True, capture_output=True)
    # 画像とスライドを書き込む
    media_names = {}
    for i, path in enumerate(sorted({p for s in slides for _, p in s.images})):
        media_names[path] = f"slide_image{i + 1}.png"
        shutil.copy(path, os.path.join(work, "ppt", "media", media_names[path]))
    for number, (s, file_name) in enumerate(zip(slides, new_files), start=1):
        open(os.path.join(work, "ppt", "slides", file_name), "w").write(s.xml(number))
        open(os.path.join(work, "ppt", "slides", "_rels", f"{file_name}.rels"), "w").write(s.rels(media_names))
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as z:
        content_types = os.path.join(work, "[Content_Types].xml")
        z.write(content_types, "[Content_Types].xml")
        for dirpath, _, files in os.walk(work):
            for f in files:
                full = os.path.join(dirpath, f)
                if full != content_types:
                    z.write(full, os.path.relpath(full, work))
    shutil.rmtree(work)


if __name__ == "__main__":
    if not SKILL_SCRIPTS:
        sys.exit("環境変数 PPTX_SKILL_SCRIPTS に pptx スキルの scripts ディレクトリを指定すること")
    media = make_media()
    assemble(build_slides(media), media)
    print(OUTPUT)
