"""ex014（下から確率を集める命名）のテスト．"""
import importlib.util
import os
import sys

import torch

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
DIRECTORY = os.path.abspath(os.path.join(PROGRAMS, "ex014_hierarchical_naming"))
sys.path.insert(0, PROGRAMS)
sys.path.insert(0, DIRECTORY)
_spec = importlib.util.spec_from_file_location("ex014_model", os.path.join(DIRECTORY, "model.py"))
ex014_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex014_model)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != DIRECTORY]
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def test_hierarchy_choose_backs_off_to_common_ancestor():
    """細かい語の確率が分かれる場合は共通の上位語を出力し，1つの語に集中する場合はその語を出力することを確認する．"""
    vocab = [{"synset": "beagle.n.01"}, {"synset": "basset.n.01"}, {"synset": "car.n.01"}]
    h = ex014_model.Hierarchy(vocab)
    split = torch.tensor([[0.45, 0.45, 0.10]], device=ex014_model.DEVICE)
    s, p = h.choose(split, 0.8)[0]
    assert s == "hound.n.01" and abs(p - 0.9) < 1e-6
    peaked = torch.tensor([[0.9, 0.05, 0.05]], device=ex014_model.DEVICE)
    assert h.choose(peaked, 0.8)[0][0] == "beagle.n.01"


def test_gloss_head():
    """定義文の最初の節のみを用いることを確認する．"""
    assert ex014_model.gloss_head("a small dog; used for hunting") == "a small dog"
