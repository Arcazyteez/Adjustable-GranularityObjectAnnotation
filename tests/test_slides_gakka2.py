"""
slides/build_slides_gakka2.py の並べ替え（reorder）とタイトルの置き換え（retitle）のテスト（order_048）．
"""
import os
import sys

from pptx import Presentation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "slides"))
import build_slides_gakka2 as G  # noqa: E402


def _deck(n):
    """
    タイトルだけのスライドを n 枚持つプレゼンテーションを作る．

    Args:
        n (int): 枚数．

    Returns:
        Presentation: プレゼンテーション．
    """
    prs = Presentation()
    for k in range(n):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = f"slide{k}"
    return prs


def test_reorder_keeps_and_drops():
    """指定した順に並び，指定しないスライドは削除される．"""
    prs = _deck(4)
    s = list(prs.slides)
    G.reorder(prs, [s[3], s[0], s[2]])
    assert [x.shapes.title.text for x in prs.slides] == ["slide3", "slide0", "slide2"]


def test_retitle():
    """タイトルの文字だけが置き換わる．"""
    prs = _deck(1)
    G.retitle(prs.slides[0], "新しいタイトル")
    assert prs.slides[0].shapes.title.text == "新しいタイトル"
