"""ex009（COCOでの3手法の比較）とex010（COCO外の物体の検出から粒度変更まで）の評価関数のテスト．"""
import importlib.util
import os
import sys

import numpy as np

PROGRAMS = os.path.join(os.path.dirname(__file__), "..", "programs")
sys.path.insert(0, PROGRAMS)


def load_train(experiment, alias):
    """
    実験ディレクトリの train.py を別名で読み込み，sys.path と sys.modules を元に戻す．

    Args:
        experiment (str): 実験ディレクトリ名．
        alias (str): モジュールの別名．

    Returns:
        module: 読み込んだ train．
    """
    directory = os.path.abspath(os.path.join(PROGRAMS, experiment))
    sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(alias, os.path.join(directory, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != directory]
    for name in ("data", "model"):
        sys.modules.pop(name, None)
    return module


ex009 = load_train("ex009_coco_siglip_wordnet_annotation", "ex009_train")
ex010 = load_train("ex010_lvis_open_world_detection", "ex010_train")


def chain(confs, synsets):
    """
    テスト用の下位語の展開結果を作る．

    Args:
        confs (list[float]): 各深さの確信度．
        synsets (list[str]): 各深さのSynset名．

    Returns:
        list[dict]: 展開結果．
    """
    return [{"depth": d, "word": s.split(".")[0], "synset": s, "conf": c} for d, (c, s) in enumerate(zip(confs, synsets))]


def test_output_depth_and_lvis_metrics():
    """累積確信度による停止と，棄却したインスタンスが再現率の分母のみに含まれることを確認する．"""
    c = chain([0.9, 0.5, 0.8], ["bird.n.01", "pigeon.n.01", "dove.n.01"])
    assert ex009.output_depth(c, "tau", 0.4) == 1 and ex009.output_depth(c, "tau", 0.95) == -1
    path = ["bird.n.01", "pigeon.n.01"]
    good = {"gt_path": path, "lvis_synset": "pigeon.n.01", "chain": chain([0.9, 0.9], path)}
    low = {"gt_path": path, "lvis_synset": "pigeon.n.01", "chain": chain([0.1], ["bird.n.01"])}
    metrics = ex009.lvis_metrics([good, low], "tau", 0.5)
    assert metrics["hP"] == 1.0 and metrics["hR"] == 0.5 and metrics["coverage"] == 0.5


def test_roc_auc():
    """完全に分離できる場合にAUROCが1，同点の場合に0.5となることを確認する．"""
    assert ex009.roc_auc(np.array([True, False]), np.array([0.9, 0.1])) == 1.0
    assert ex009.roc_auc(np.array([True, False]), np.array([0.5, 0.5])) == 0.5


def test_recall_at_k_counts_only_top_k():
    """上位 K 件のボックスのみで再現率を数えることを確認する．"""
    gts = {"1": [{"box": [0, 0, 10, 10], "kind": "noncoco", "synset": "x.n.01"}]}
    boxes = {"1": [[50, 50, 60, 60], [0, 0, 10, 10]]}
    recall = ex010.recall_at_k(gts, boxes, [1, 2])
    assert recall["noncoco_recall@1"] == 0.0 and recall["noncoco_recall@2"] == 1.0


def test_output_boxes_adds_only_non_overlapping_supplements():
    """YOLO11xのボックスと重なる候補は加えず，重ならない候補を上位 K 件まで加えることを確認する．"""
    ctx = {"detections": {"yolo": {"1": {"boxes": [[0, 0, 10, 10]], "scores": [0.9], "top5": [["dog"] * 5]}}}}
    proposals = {"owlv2": {"1": {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], "scores": [0.9, 0.8, 0.7]}}}
    out = ex010.output_boxes("hybrid+owlv2", "owlv2", 1, "1", ctx, proposals)
    assert [b["source"] for b in out] == ["yolo", "owlv2"] and out[1]["box"] == [20, 20, 30, 30]
