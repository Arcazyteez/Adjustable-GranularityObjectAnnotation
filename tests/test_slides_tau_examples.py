"""
slides/tau_examples_2026C.py の例の選び方（is_refining）のテスト（order_044）．
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "slides"))
import tau_examples_2026C as T  # noqa: E402


def test_refining_path_to_fine_label():
    """person → skater → skateboarder と細かい正解まで下る列は True になる．"""
    nodes = ["person.n.01", "skater.n.01", "skateboarder.n.01", "skateboarder.n.01"]
    assert T.is_refining(nodes, "skateboarder.n.01")


def test_not_refining_when_wrong_or_short():
    """細かい正解の経路から外れる語を含む列や，語が2つ以下の列は False になる．"""
    assert not T.is_refining(["person.n.01", "skateboarder.n.01"], "skateboarder.n.01")
    assert not T.is_refining(["person.n.01", "swimmer.n.01", "skateboarder.n.01"], "skateboarder.n.01")
    assert not T.is_refining([None, "person.n.01", "skater.n.01", "skater.n.01"], "skateboarder.n.01")
