"""
slides/build_slides_2026C.py の発表者ノート（notes_xml，NOTES）のテスト（order_047）．
"""
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "slides"))
import build_slides_2026C as B  # noqa: E402

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def test_notes_xml_paragraphs_and_escape():
    """改行ごとに段落が分かれ，& や < がエスケープされた正しいXMLになる．"""
    root = ET.fromstring(B.notes_xml("1行目 A&B\n2行目 <x>").encode())
    texts = [t.text for t in root.iter(f"{A}t")]
    assert texts == ["1行目 A&B", "2行目 <x>"]


def test_notes_count_matches_slides():
    """ノートは23枚分あり，空のノートは無い．"""
    assert len(B.NOTES) == 23
    assert all(n.strip() for n in B.NOTES)
