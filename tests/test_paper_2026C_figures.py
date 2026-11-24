"""paper_2026C/make_figures.py の2つの粒度の表示のテスト（order_040）．"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_make_figures():
    """
    paper_2026C/make_figures.py をモジュールとして読み込む．

    Returns:
        module: make_figures モジュール．
    """
    spec = importlib.util.spec_from_file_location("make_figures_2026C", os.path.join(ROOT, "paper_2026C", "make_figures.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_level_label_shows_coarse_when_finer():
    """細かい語と深さ0の語が異なる場合は "細かい語 (深さ0の語)" になる．"""
    mf = load_make_figures()
    record = {"top_mass": [["person", "chair"], ["tennis racket", "person"]]}
    assert mf.two_level_label(record, 0, "tennis player") == "tennis player (person)"


def test_two_level_label_same_word():
    """細かい語と深さ0の語が同じ場合は語のみになる．"""
    mf = load_make_figures()
    record = {"top_mass": [["person"], ["tennis racket"]]}
    assert mf.two_level_label(record, 1, "tennis racket") == "tennis racket"
