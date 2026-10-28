"""WordNet関連（programs/common/wordnet_utils.py）のテスト．"""
import os
import sys

import numpy as np
from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs"))
from common.wordnet_utils import (COCO_TO_SYNSET, EXTRA_TO_SYNSET, hierarchical_annotation,  # noqa: E402
                                  hyponym_candidates)


def test_all_synsets_exist():
    """対応表の全Synsetが存在し，COCO 80クラス・追加22語を網羅していることを確認する．"""
    assert len(COCO_TO_SYNSET) == 80 and len(EXTRA_TO_SYNSET) == 22
    for name in list(COCO_TO_SYNSET.values()) + list(EXTRA_TO_SYNSET.values()):
        wn.synset(name)


def test_hyponym_candidates_unique():
    """下位語の候補語に重複がなく，Synsetと同じ長さであることを確認する．"""
    words, synsets = hyponym_candidates(wn.synset("dog.n.01"))
    assert len(words) == len(set(words)) == len(synsets) > 0


class _FirstWordVLM:
    """常に先頭の候補語を選ぶダミーVLM．"""

    def score(self, images, vocab):
        """
        先頭の候補語に確率1を与える．

        Args:
            images (list): 画像のリスト（長さN）．
            vocab (list[str]): 候補語彙（長さV）．

        Returns:
            np.ndarray: スコア行列（N×V）．
        """
        scores = np.zeros((len(images), len(vocab)))
        scores[:, 0] = 1.0
        return scores


def test_hierarchical_annotation_terminates_at_leaf():
    """下位語集合が空になる（葉に到達する）まで深さが1ずつ増えることを確認する．"""
    chain = hierarchical_annotation(_FirstWordVLM(), None, "giraffe", "giraffe.n.01", 0.9)
    assert [s["depth"] for s in chain] == list(range(len(chain)))
    assert chain[1]["word"] == "okapi" and not wn.synset(chain[-1]["synset"]).hyponyms()
