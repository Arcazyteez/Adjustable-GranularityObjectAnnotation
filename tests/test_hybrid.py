"""YOLOとCLIPのハイブリッド構成（programs/ex006_coco_val500_yolo_clip_hybrid/train.py）の判定・集計関数のテスト．"""
import importlib.util
import os
import sys

import numpy as np

EX006_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "ex006_coco_val500_yolo_clip_hybrid")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs"))
sys.path.insert(0, EX006_DIR)
# ex005のテストも train という名前のモジュールを読み込むため，別名で読み込む
_spec = importlib.util.spec_from_file_location("ex006_train", os.path.join(EX006_DIR, "train.py"))
ex006_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex006_train)
# ex006の data，model がキャッシュされると他の実験のテストが誤ったモジュールを読み込むため，元に戻す
sys.path.remove(EX006_DIR)
for _name in ("data", "model"):
    sys.modules.pop(_name, None)

VOCAB = ["dog", "cat", "camel", "penguin"]
EXTRA = {"camel", "penguin"}


def test_decide_root_hybrid_always_uses_yolo():
    """hybrid ではCLIPのスコアに関わらずYOLOの出力が深さ0になることを確認する．"""
    word, conf, source = ex006_train.decide_root("dog", 0.87, None, VOCAB, EXTRA, "hybrid")
    assert (word, conf, source) == ("dog", 0.87, "yolo")


def test_decide_root_switch_only_for_extra_words():
    """hybrid_switch ではCLIPの最上位がCOCO外の語のときのみ切り替わることを確認する．"""
    to_camel = ex006_train.decide_root("dog", 0.87, np.array([0.1, 0.0, 0.8, 0.1]), VOCAB, EXTRA, "hybrid_switch")
    assert to_camel[0] == "camel" and to_camel[2] == "clip" and abs(to_camel[1] - 0.8) < 1e-9
    keep_yolo = ex006_train.decide_root("dog", 0.87, np.array([0.1, 0.7, 0.1, 0.1]), VOCAB, EXTRA, "hybrid_switch")
    assert keep_yolo == ("dog", 0.87, "yolo")


def test_decide_root_switch_gate():
    """hybrid_switch_c0.5 ではCOCO外の語のCLIPの確信度が0.5未満なら切り替わらないことを確認する．"""
    low = np.array([0.3, 0.0, 0.4, 0.3])
    assert ex006_train.decide_root("dog", 0.87, low, VOCAB, EXTRA, "hybrid_switch")[2] == "clip"
    assert ex006_train.decide_root("dog", 0.87, low, VOCAB, EXTRA, "hybrid_switch_c0.5") == ("dog", 0.87, "yolo")


def test_summarize_and_compare_chains():
    """正命名率がGT総数で割られ，共通インスタンスで深さ1の一致率が計算されることを確認する．"""
    chain_a = [{"word": "dog"}, {"word": "basenji"}]
    chain_b = [{"word": "dog"}, {"word": "corgi"}, {"word": "Cardigan"}]
    records = [{"image_id": 1, "gt_index": 0, "gt_name": "dog", "yolo_word": "dog", "root_source": "yolo",
                "top1_correct": True, "chain": chain_a},
               {"image_id": 1, "gt_index": 1, "gt_name": "cat", "yolo_word": "cat", "root_source": "clip",
                "top1_correct": False}]
    metrics = ex006_train.summarize(records, n_gt=4)
    assert metrics["annotation_recall"] == 0.25 and metrics["n_switched_from_correct_yolo"] == 1
    proposed = {(1, 0): {"chain": chain_b}, (1, 1): {"chain": None}}
    result = ex006_train.compare_chains(records, proposed)
    assert result["n_common"] == 1 and result["depth1_agreement"] == 0.0 and result["proposed_mean_depth"] == 2.0
