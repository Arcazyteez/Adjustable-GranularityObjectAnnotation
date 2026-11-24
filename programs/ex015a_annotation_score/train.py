"""
ex015aの実行スクリプト：粒度可変アノテーションへのスコアの付与と改善（order_020）．

ex015で選んだ設定（greedy，β = 2，τ = 0.1，ens の文章，人の役割を除く，加える領域 K = 100）で全ての出力を求め，
各出力（ボックスと単語の組）に以下のスコアを付けて比較する．物体認識モデル・埋め込みモデルの再学習は行わない．
- det：検出のスコア（YOLO11xの確信度，OWLv2の物体らしさ．ex015の棄却に相当）
- name：出力した単語の集約した確率 q（YOLO11xのボックスはOWLv2の領域の特徴によるクラスのSynsetの q）
- det_x_name：両者の積
- det_x_name_added：加えた領域のみ両者の積とし，YOLO11xのボックスは検出のスコアのまま
- calibrated：検出・命名の特徴（det，q，語彙上の確率の最大値とエントロピー，単語の深さ，出力元）を入力とするロジスティック回帰．
  重み（9個）は検証用の画像のみで求める
評価：スコアの高い順に出力を対応付け（COCOのAPと同じ方法），閾値を下げた場合の階層的な適合率・再現率の曲線からAP（fed，strict）を求め，
検証用の画像で hF_fed または hF_strict が最大となる閾値をテスト用の画像に適用する．スコアを確率とみなした校正誤差（ECE）も求める．

出力: outputs/ex015a_annotation_score/{スコア}/{OWLv2の重み}/{seed}/metrics.json（dev，test），config.json，curve_{分割}.csv，
    reliability.json，および直下の table_results.csv，calibrator_{重み}.json，qualitative/
"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定論的なCUDAの行列計算に必要な設定

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from nltk.corpus import wordnet as wn  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import (K_ADD, build_records, ex015_modules, features, load_images, selected_settings)  # noqa: E402
from data import ancestors_of, max_depth  # noqa: E402
from model import (SCORES, LogisticCalibrator, at_threshold, average_precision, best_threshold, collect, curve,  # noqa: E402
                   expected_calibration_error, gt_totals, match_greedy, raw_score)

# ===== 実験条件（order_020） =====
EXPERIMENT = "ex015a_annotation_score"
SEEDS = [42]
MODELS = ["st", "st_large", "ens"]   # ST（主），ST-L，FT（参考）
SPLITS = ["dev", "test"]
CURVE_POINTS = 300                    # 保存する曲線の点数
N_QUALITATIVE = 8                     # 定性評価で表示する加えた領域の数


def flatten(records):
    """
    画像ごとの記録を平坦化する．

    Args:
        records (list[list[dict]]): 画像ごとの記録．

    Returns:
        list[dict]: 全出力の記録．
    """
    return [r for recs in records for r in recs]


def split_scores(records, flat_scores):
    """
    平坦化したスコアを画像ごとに分ける．

    Args:
        records (list[list[dict]]): 画像ごとの記録．
        flat_scores (np.ndarray): 全出力のスコア．

    Returns:
        list[np.ndarray]: 画像ごとのスコア．
    """
    out, pos = [], 0
    for recs in records:
        out.append(flat_scores[pos:pos + len(recs)])
        pos += len(recs)
    return out


def save_curve(cur, path):
    """
    曲線を間引いてCSVに保存する．

    Args:
        cur (dict): curve の戻り値．
        path (str): 保存先．

    Returns:
        None
    """
    idx = np.unique(np.linspace(0, len(cur["score"]) - 1, CURVE_POINTS).astype(int))
    pd.DataFrame({k: v[idx] for k, v in cur.items()}).to_csv(path, index=False)


def evaluate_score(images, records, totals, scores):
    """
    1つのスコアについて，分割ごとの曲線，AP，校正誤差を求める．

    Args:
        images (dict): 分割 → 画像．
        records (dict): 分割 → 画像ごとの記録．
        totals (dict): 分割 → gt_totals の戻り値．
        scores (dict): 分割 → 全出力のスコア．

    Returns:
        tuple[dict, dict, dict]: 分割 → 曲線，分割 → 指標，テスト用の校正の表．
    """
    curves, metrics, rel = {}, {}, None
    for s in SPLITS:
        flat = collect(images[s], records[s], split_scores(records[s], scores[s]))
        cur = curve(flat, totals[s])
        ece, table = expected_calibration_error(flat["score"], flat["label"])
        curves[s] = cur
        metrics[s] = {"AP_fed": average_precision(cur["hR"], cur["fed_hP"]),
                      "AP_strict": average_precision(cur["hR"], cur["strict_hP"]), "ECE": ece,
                      "max_fed_hF": float(cur["fed_hF"].max()), "max_strict_hF": float(cur["strict_hF"].max())}
        if s == "test":
            rel = table
    for crit in ["fed_hF", "strict_hF"]:
        thr = best_threshold(curves["dev"], crit)
        for s in SPLITS:
            metrics[s] |= {f"{k}@{crit}": v for k, v in at_threshold(curves[s], thr).items()}
    return curves, metrics, rel


def word_features(rec):
    """
    加えた領域の経路上の各単語について，単語の正しさの校正に用いる特徴を並べる．

    Args:
        rec (dict): 出力の記録．

    Returns:
        np.ndarray: 特徴（経路の長さ×4）．列は [log q，単語の深さ/10，log pmax，entropy]．
    """
    return np.array([[np.log(max(q, 1e-6)), max_depth(n) / 10.0, np.log(rec["pmax"] + 1e-12), rec["entropy"]]
                     for n, q in rec["chain"]], dtype=np.float32)


def chain_data(images, records):
    """
    加えた領域のうちGTと対応したもの（検出のスコアの順に対応付け）の経路上の各単語について，特徴と正しさ
    （GTの上位語・自身・下位語であるか）を集める．

    Args:
        images (list[dict]): 画像．
        records (list[list[dict]]): 画像ごとの記録．

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 特徴（M×4），集約した確率 q（M），正しさ（M）．
    """
    xs, qs, ys = [], [], []
    for img, recs in zip(images, records):
        matched = match_greedy(img["iou"], np.array([r["det"] for r in recs]))
        for r, g in zip(recs, matched):
            if r["source"] != "owlv2" or g < 0:
                continue
            t = img["gts"][g]["synset"]
            xs.append(word_features(r))
            qs += [q for _, q in r["chain"]]
            ys += [float(n in ancestors_of(t) or t in ancestors_of(n)) for n, _ in r["chain"]]
    return np.concatenate(xs), np.array(qs), np.array(ys)


def chain_scores(word_cal, rec):
    """
    加えた領域の経路上の各単語のスコア（物体であるとした場合にその単語が正しい確率）を求める．
    粗い語ほど正しい可能性が高いため，経路の上位からの累積の最小値をとり，細かい語ほどスコアが下がるようにする．

    Args:
        word_cal (LogisticCalibrator): 単語の正しさを推定するモデル．
        rec (dict): 出力の記録．

    Returns:
        list[float]: 経路上の各単語のスコア．
    """
    return np.minimum.accumulate(word_cal.predict(word_features(rec))).tolist()


def chain_reliability(word_cal, images, records):
    """
    テスト用の画像で，経路上の各単語の集約した確率 q と校正したスコアの校正誤差を求める．

    Args:
        word_cal (LogisticCalibrator): 単語の正しさを推定するモデル．
        images (list[dict]): テスト用の画像．
        records (list[list[dict]]): テスト用の記録．

    Returns:
        dict: n（単語の数），accuracy，ECE_q，ECE_calibrated，表．
    """
    _, q, y = chain_data(images, records)
    scores = []
    for img, recs in zip(images, records):
        matched = match_greedy(img["iou"], np.array([r["det"] for r in recs]))
        scores += [s for r, g in zip(recs, matched) if r["source"] == "owlv2" and g >= 0 for s in chain_scores(word_cal, r)]
    scores = np.array(scores)
    e_q, t_q = expected_calibration_error(q, y)
    e_c, t_c = expected_calibration_error(scores, y)
    return {"n": int(len(y)), "accuracy": float(y.mean()), "ECE_q": e_q, "ECE_calibrated": e_c, "table_q": t_q,
            "table_calibrated": t_c}


def qualitative(images, records, scores, word_cal, model_key, root):
    """
    定性評価の画像（ex013と同じ6枚）について，YOLO11xの出力と，校正したスコアが高い加えた領域の上位 N_QUALITATIVE 件を，
    経路上の各単語の確率と出力のスコアを付けて保存する．

    Args:
        images (list[dict]): テスト用の画像．
        records (list[list[dict]]): テスト用の記録．
        scores (list[np.ndarray]): 画像ごとの校正したスコア．
        word_cal (LogisticCalibrator): 経路上の各単語のスコアのモデル．
        model_key (str): OWLv2の重みの名前．
        root (str): 実験の出力ディレクトリ．

    Returns:
        None
    """
    t15 = ex015_modules()[2]
    keep = set(json.load(open(os.path.join(t15.OUTPUT_DIR, "ex013_open_vocabulary_granular", "qualitative", "ex012.json"))))
    word = lambda s: wn.synset(s).lemma_names()[0].replace("_", " ")  # noqa: E731
    res = {}
    for img, recs, sc in zip(images, records, scores):
        if img["image_id"] not in keep:
            continue
        added = sorted([j for j, r in enumerate(recs) if r["source"] == "owlv2"], key=lambda j: -sc[j])[:N_QUALITATIVE]
        sel = [j for j, r in enumerate(recs) if r["source"] == "yolo"] + added
        res[img["image_id"]] = [{"box": img["boxes"][j]["box"], "source": recs[j]["source"], "synset": recs[j]["synset"],
                                 "word": word(recs[j]["synset"]), "score": float(sc[j]),
                                 "chain": [{"word": word(n), "q": float(q), "score": float(c)}
                                           for (n, q), c in zip(recs[j]["chain"],
                                                                 chain_scores(word_cal, recs[j]) if recs[j]["source"] == "owlv2"
                                                                 else [recs[j]["det"]])]}
                                for j in sel]
    d = os.path.join(root, "qualitative")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{model_key}.json"), "w") as f:
        json.dump(res, f, indent=1)


def run_model(ctx, model_key, seed, root):
    """
    1つのOWLv2の重みについて，全てのスコアを評価する．

    Args:
        ctx (dict): ex015の load_context の戻り値．
        model_key (str): OWLv2の重みの名前．
        seed (int): シード．
        root (str): 実験の出力ディレクトリ．

    Returns:
        list[dict]: 結果の表の行．
    """
    d15, _, t15 = ex015_modules()
    settings = selected_settings()
    owl = t15.OWLv2Regions(model_key)
    namer = t15.Namer(owl, t15.vocab_variant(d15.caption_vocabulary(), settings["exclude_person"]), settings["text"])
    images = {s: load_images(ctx, s, model_key) for s in SPLITS}
    records = {s: build_records(images[s], namer, settings) for s in SPLITS}
    totals = {s: gt_totals(images[s]) for s in SPLITS}
    del owl, namer
    torch.cuda.empty_cache()
    flat_recs = {s: flatten(records[s]) for s in SPLITS}
    # 校正：検証用の画像で，検出のスコアの順に対応付けた場合の正誤（不明は除く）を教師とする
    dev_det = collect(images["dev"], records["dev"], split_scores(records["dev"], raw_score(flat_recs["dev"], "det")))
    x_dev, y_dev = features(flat_recs["dev"]), dev_det["label"]
    cal = LogisticCalibrator().fit(x_dev[y_dev >= 0], y_dev[y_dev >= 0])
    with open(os.path.join(root, f"calibrator_{model_key}.json"), "w") as f:
        json.dump({"w": cal.w.tolist(), "b": cal.b, "mean": cal.mean.tolist(), "std": cal.std.tolist(),
                   "n_train": int((y_dev >= 0).sum()), "positive_rate": float(y_dev[y_dev >= 0].mean())}, f, indent=4)
    # 経路上の各単語のスコア：検証用の画像で，GTと対応した加えた領域の経路上の単語の正しさを教師とする
    xw, _, yw = chain_data(images["dev"], records["dev"])
    word_cal = LogisticCalibrator().fit(xw, yw)
    with open(os.path.join(root, f"word_calibrator_{model_key}.json"), "w") as f:
        json.dump({"w": word_cal.w.tolist(), "b": word_cal.b, "mean": word_cal.mean.tolist(), "std": word_cal.std.tolist(),
                   "n_train": int(len(yw)), "positive_rate": float(yw.mean())}, f, indent=4)
    rows = []
    for kind in SCORES:
        scores = {s: cal.predict(features(flat_recs[s])) if kind == "calibrated" else raw_score(flat_recs[s], kind)
                  for s in SPLITS}
        curves, metrics, rel = evaluate_score(images, records, totals, scores)
        d = os.path.join(root, kind, model_key, str(seed))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(settings | {"score": kind, "model": model_key, "k_add": K_ADD, "seed": seed}, f, indent=4)
        with open(os.path.join(d, "reliability.json"), "w") as f:
            json.dump(rel, f, indent=4)
        for s in SPLITS:
            save_curve(curves[s], os.path.join(d, f"curve_{s}.csv"))
        rows.append({"score": kind, "model": model_key} | {k: v for k, v in metrics["test"].items()})
        if kind == "calibrated":
            qualitative(images["test"], records["test"], split_scores(records["test"], scores["test"]), word_cal, model_key, root)
            rel_chain = chain_reliability(word_cal, images["test"], records["test"])
            with open(os.path.join(root, f"chain_reliability_{model_key}.json"), "w") as f:
                json.dump(rel_chain, f, indent=4)
            print(model_key, "chain", {k: rel_chain[k] for k in ["n", "accuracy", "ECE_q", "ECE_calibrated"]}, flush=True)
        print(model_key, kind, {k: round(metrics["test"][k], 4) for k in ["AP_fed", "AP_strict", "ECE", "fed_hF@fed_hF",
                                                                         "strict_hF@strict_hF"]}, flush=True)
    return rows


def main():
    """全てのOWLv2の重みとスコアを評価し，集計表を作成する．"""
    d15 = ex015_modules()[0]
    t15 = ex015_modules()[2]
    root = os.path.join(t15.OUTPUT_DIR, EXPERIMENT)
    os.makedirs(root, exist_ok=True)
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        start = time.time()
        ctx = d15.load_context(seed)
        for key in MODELS:
            rows += run_model(ctx, key, seed, root)
        print("elapsed", time.time() - start, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(root, "table_results.csv"), index=False)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
