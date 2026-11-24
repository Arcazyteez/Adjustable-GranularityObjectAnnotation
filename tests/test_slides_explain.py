"""
slides/explain_examples_2026C.py の判定（is_correct）と上位の取り出し（topk）のテスト（order_050）．
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "slides"))
import explain_examples_2026C as X  # noqa: E402


def test_is_correct():
    """細かい正解そのものと上位語は正しく，別の枝の語と出力なしは誤りになる．"""
    assert X.is_correct("jockey.n.01", "jockey.n.01")
    assert X.is_correct("rider.n.01", "jockey.n.01")
    assert not X.is_correct("horse.n.01", "jockey.n.01")
    assert not X.is_correct(None, "jockey.n.01")


def test_topk_order():
    """確率の高い順に，名前と確率を返す．"""
    out = X.topk(np.array([0.1, 0.6, 0.3]), ["a", "b", "c"], k=2)
    assert [w for w, _ in out] == ["b", "c"]
