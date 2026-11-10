"""粒度可変アノテーションの評価の改善（programs/ex007_coco_val_hierarchy_eval）の停止規則と評価指標のテスト．"""
import importlib.util
import os
import sys

import numpy as np

EX007_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex007_coco_val_hierarchy_eval")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs"))
sys.path.insert(0, EX007_DIR)
# 他の実験のテストも data，model，train という名前のモジュールを読み込むため，別名で読み込む
_spec = importlib.util.spec_from_file_location("ex007_train", os.path.join(EX007_DIR, "train.py"))
ex007_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex007_train)
ex007_data = sys.modules["data"]
# ex007の data，model がキャッシュされると他の実験のテストが誤ったモジュールを読み込むため，元に戻す
sys.path.remove(EX007_DIR)
for _name in ("data", "model"):
    sys.modules.pop(_name, None)


def make_chain(confs, synsets):
    """
    テスト用の下位語の展開結果を作る．

    Args:
        confs (list[float]): 各深さの確信度．
        synsets (list[str]): 各深さのSynset名．

    Returns:
        list[dict]: 展開結果．
    """
    return [{"depth": d, "word": s.split(".")[0], "synset": s, "conf": c, "n_candidates": None}
            for d, (c, s) in enumerate(zip(confs, synsets))]


def test_output_depth_cumulative_conf():
    """累積確信度が τ を下回る直前の深さで止まり，深さ0で下回れば棄却されることを確認する．"""
    chain = make_chain([0.9, 0.5, 0.8], ["bird.n.01", "pigeon.n.01", "dove.n.01"])
    # 累積確信度は 0.9，0.45，0.36
    assert ex007_train.output_depth(chain, "tau", 0.0) == 2
    assert ex007_train.output_depth(chain, "tau", 0.4) == 1
    assert ex007_train.output_depth(chain, "tau", 0.95) == -1
    assert ex007_train.output_depth(chain, "depth", 1) == 1
    assert ex007_train.output_depth(chain, "depth", 5) == 2


def test_gt_path_and_hierarchical_counts():
    """LVISのSynsetまでの経路が求まり，LVISより細かい出力が正誤の判定から除かれることを確認する．"""
    path = ex007_data.gt_path_synsets("bird", "pigeon.n.01")
    assert "bird.n.01" in path and "pigeon.n.01" in path
    assert ex007_data.gt_path_synsets("bird", "bird.n.01") is None
    assert ex007_data.gt_path_synsets("dog", "pigeon.n.01") is None
    record = {"gt_path": path, "lvis_synset": "pigeon.n.01",
              "chain": make_chain([1.0] * len(path), sorted(path, key=lambda s: len(path)))}
    inter, pred, gt = ex007_train.hierarchical_counts(record, -1 + len(path))
    assert inter == pred == gt == len(path)
    # 下位語が細かすぎる場合（LVISのpigeonより下）は出力の経路に数えない
    finer = "homing_pigeon.n.01"
    record["chain"] = make_chain([1.0, 1.0], ["bird.n.01", finer])
    assert ex007_train.hierarchical_counts(record, 1) == (1, 1, len(path))


def test_lvis_level_metrics_abstain_counts_in_recall():
    """棄却したインスタンスは適合率に含まれず，再現率の分母のみに含まれることを確認する．"""
    path = ex007_data.gt_path_synsets("bird", "pigeon.n.01")
    good = {"gt_path": path, "lvis_synset": "pigeon.n.01", "chain": make_chain([0.9], ["bird.n.01"])}
    low = {"gt_path": path, "lvis_synset": "pigeon.n.01", "chain": make_chain([0.1], ["bird.n.01"])}
    metrics = ex007_train.lvis_level_metrics([good, low], "tau", 0.5)
    assert metrics["hP"] == 1.0 and abs(metrics["hR"] - 1 / len(path) / 2) < 1e-9 and metrics["coverage"] == 0.5


def test_decide_root_switch_gate():
    """γ=0.5の構成ではCOCO外の語の確信度が0.5未満のときYOLOの出力が深さ0になることを確認する．"""
    vocab = ["dog", "cat", "camel"]
    ins = {"yolo_word": "dog", "det_score": 0.8}
    scores = np.array([0.3, 0.3, 0.4])
    assert ex007_train.decide_root("hybrid_switch", ins, scores, vocab, {"camel"})[0] == "camel"
    assert ex007_train.decide_root("hybrid_switch_c0.5", ins, scores, vocab, {"camel"}) == ("dog", 0.8, "yolo")
    assert ex007_train.decide_root("proposed", ins, scores, vocab, {"camel"}) == ("camel", 0.4, "clip")
