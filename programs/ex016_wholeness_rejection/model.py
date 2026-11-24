"""
ex016のモデル：全体らしさの特徴を加えたアノテーションのスコア．

ex015a と同じロジスティック回帰（検証用の画像で，検出のスコアの順に対応付けた場合の正誤を教師とする）に，
ex016 data の全体らしさの特徴を加える．加える特徴は加えた領域のみ0以外の値をとる．
- base：ex015a の9個の重み（8特徴＋切片）のみ（ex015b の calibrated と同じ）
- geom：base ＋ 幾何の特徴（4個）
- wordnet：base ＋ WordNetの特徴（2個）
- full：base ＋ 幾何 ＋ WordNet（提案）
- rule：base のスコアのうち，meronym が1の加えた領域のスコアを0にする（重みを求め直さない規則のみの版）
"""
import numpy as np

from data import GEOM_COLS, WORDNET_COLS, ex015b_modules

FEATURE_SETS = {"base": [], "geom": GEOM_COLS, "wordnet": WORDNET_COLS, "full": GEOM_COLS + WORDNET_COLS}
SCORERS = list(FEATURE_SETS) + ["rule"]


def stack_features(base_x, whole_x, cols):
    """
    ex015a の特徴に，全体らしさの特徴の指定した列を加える．

    Args:
        base_x (np.ndarray): ex015a の特徴（N×8）．
        whole_x (np.ndarray): 全体らしさの特徴（N×6）．
        cols (list[int]): 加える列．

    Returns:
        np.ndarray: 特徴（N×(8＋len(cols))，float32）．
    """
    return np.concatenate([base_x, whole_x[:, cols]], axis=1).astype(np.float32)


def fit_scorers(dev_x, dev_whole, dev_label):
    """
    検証用の画像で，全ての特徴の組についてロジスティック回帰の重みを求める．

    Args:
        dev_x (np.ndarray): 検証用の ex015a の特徴（N×8）．
        dev_whole (np.ndarray): 検証用の全体らしさの特徴（N×6）．
        dev_label (np.ndarray): 正誤（1，0，評価しない出力は −1）（N）．

    Returns:
        dict[str, LogisticCalibrator]: 特徴の組 → 求めたモデル．
    """
    ma = ex015b_modules()[0].ex015a_modules()[1]
    ok = dev_label >= 0
    return {name: ma.LogisticCalibrator().fit(stack_features(dev_x, dev_whole, cols)[ok], dev_label[ok])
            for name, cols in FEATURE_SETS.items()}


def predict_scores(cals, x, whole):
    """
    全てのスコアの方法でスコアを求める．

    Args:
        cals (dict): fit_scorers の戻り値．
        x (np.ndarray): ex015a の特徴（N×8）．
        whole (np.ndarray): 全体らしさの特徴（N×6）．

    Returns:
        dict[str, np.ndarray]: SCORERS の各方法 → スコア（N）．
    """
    out = {name: cals[name].predict(stack_features(x, whole, cols)) for name, cols in FEATURE_SETS.items()}
    out["rule"] = np.where(whole[:, 4] > 0, 0.0, out["base"])
    return out
