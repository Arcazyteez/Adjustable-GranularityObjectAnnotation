"""
ex016の実行スクリプト：物体の全体らしさの特徴による，物体の一部・背景の領域の棄却（order_025，提案A）．

ex015で選んだ設定（ens の文章，人の役割を除く，β = 2，τ = 0.1，加える領域 K = 100）と同じ出力について，
アノテーションのスコアに全体らしさの特徴（ex016 data）を加え，物体の一部や背景の加えた領域のスコアを下げる．
物体認識モデル・埋め込みモデルの再学習は行わず，ロジスティック回帰の重みのみを検証用の画像で求める．
- 命名の方法：greedy（提案，上位語へ集約），flat（確率が最大の1語）
- スコア：base（ex015b の calibrated と同じ），geom，wordnet，full（提案），rule（meronym の領域を棄却する規則のみ）
評価は ex015b と同じ（AP_fed，AP_strict，AP_exact，AP_named，ECE，検証用の画像で hF_strict が最大の閾値での指標，誤りの分解）．
ばらつき：元の分割（seed 42）と，検証用とテスト用の画像を合わせて seed 0〜4 で半分ずつに分け直した場合．
元のテスト用の画像では，画像単位のブートストラップで base との差の95%信頼区間を求める．
特徴の診断：テスト用の画像の加えた領域を ex015b の誤りの種類に分け，種類ごとに特徴の平均と meronym の割合を求める．
既に結果（errors.json）がある条件は読み込み，ブートストラップの結果がある重みは再計算しない．

出力: outputs/ex016_wholeness_rejection/{方法}_{スコア}/{OWLv2の重み}/{seed}/metrics.json（dev，test），config.json，errors.json，
    curve_test.csv，および直下の table_results.csv，table_seeds.csv，bootstrap_{重み}.json，diagnosis_{重み}.json
"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定論的なCUDAの行列計算に必要な設定

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import WHOLENESS_NAMES, build_records, ex015b_modules, wholeness_features  # noqa: E402
from model import FEATURE_SETS, SCORERS, fit_scorers, predict_scores  # noqa: E402

# ===== 実験条件（order_025） =====
EXPERIMENT = "ex016_wholeness_rejection"
ORIGINAL_SEED = 42
RESPLIT_SEEDS = [0, 1, 2, 3, 4]
MODELS = ["st", "st_large", "ens"]
METHODS = ["greedy", "flat"]
SPLITS = ["dev", "test"]
N_BOOT = 1000
BOOT_MODELS = ["st", "ens"]
# ブートストラップで比べる組（a − b）．（方法，スコア）
BOOT_PAIRS = [(("greedy", "full"), ("greedy", "base")), (("greedy", "geom"), ("greedy", "base")),
              (("greedy", "wordnet"), ("greedy", "base")), (("greedy", "rule"), ("greedy", "base")),
              (("flat", "full"), ("flat", "base")), (("greedy", "full"), ("flat", "full"))]


def modules():
    """
    ex015a，ex015b のモジュールをまとめて返す．

    Returns:
        tuple: (ex015a data，ex015a model，ex015a train，ex015b model，ex015b train)．
    """
    db, mb, tb = ex015b_modules()
    da, ma, ta = db.ex015a_modules()
    return da, ma, ta, mb, tb


def split_wholeness(images, records, cache, method):
    """
    分割の全出力の全体らしさの特徴を，平坦化した順に並べる（画像ごとの値は cache に保持する）．

    Args:
        images (list[dict]): 画像．
        records (list[list[dict]]): 画像ごとの記録．
        cache (dict): （方法，画像ID）→ 特徴．
        method (str): 命名の方法．

    Returns:
        np.ndarray: 特徴（全出力数×6）．
    """
    rows = []
    for img, recs in zip(images, records):
        key = (method, img["image_id"])
        if key not in cache:
            cache[key] = wholeness_features(img, recs)
        rows.append(cache[key])
    return np.concatenate(rows)


def all_scores(images, records, cache, method):
    """
    検証用の画像で全ての特徴の組の重みを求め，全分割のスコアを返す．

    Args:
        images (dict): 分割 → 画像．
        records (dict): 分割 → 画像ごとの記録．
        cache (dict): split_wholeness の cache．
        method (str): 命名の方法．

    Returns:
        tuple[dict, dict]: スコア → 分割 → 全出力のスコア，特徴の組 → 求めたモデル．
    """
    da, ma, ta, _, _ = modules()
    flat = {s: ta.flatten(records[s]) for s in SPLITS}
    whole = {s: split_wholeness(images[s], records[s], cache, method) for s in SPLITS}
    dev_det = ma.collect(images["dev"], records["dev"], ta.split_scores(records["dev"], ma.raw_score(flat["dev"], "det")))
    cals = fit_scorers(da.features(flat["dev"]), whole["dev"], dev_det["label"])
    per_split = {s: predict_scores(cals, da.features(flat[s]), whole[s]) for s in SPLITS}
    return {k: {s: per_split[s][k] for s in SPLITS} for k in SCORERS}, cals


def run_seed(images, records, seed, model_key, settings, root, cache, test_parts, need_boot):
    """
    1つの分割の seed について，全ての方法とスコアを評価する．結果が既にある場合は読み込む．

    Args:
        images (dict): 元の分割 → 画像．
        records (dict): 方法 → 元の分割 → 画像ごとの記録．
        seed (int): 分割の seed．
        model_key (str): OWLv2の重みの名前．
        settings (dict): ex015で選んだ設定．
        root (str): 出力ディレクトリ．
        cache (dict): split_wholeness の cache．
        test_parts (dict): （方法，スコア）→ 元のテスト用の画像の評価の量（追記する）．
        need_boot (bool): ブートストラップのために評価の量を保持するか．

    Returns:
        list[dict]: 結果の表の行．
    """
    da, ma, ta, mb, tb = modules()
    imgs, recs = tb.split_sets(images, records, seed)
    rows = []
    for method in METHODS:
        dirs = {k: os.path.join(root, f"{method}_{k}", model_key, str(seed)) for k in SCORERS}
        keep = need_boot and seed == ORIGINAL_SEED
        if not keep and all(os.path.exists(os.path.join(d, "errors.json")) for d in dirs.values()):
            for k, d in dirs.items():
                m, e = (json.load(open(os.path.join(d, f))) for f in ("metrics.json", "errors.json"))
                rows.append(tb.result_row(method, k, model_key, seed, m, e))
            continue
        scores, cals = all_scores(imgs, recs[method], cache, method)
        for kind in SCORERS:
            metrics, curves, parts = tb.evaluate(imgs, recs[method], scores[kind])
            sl = ta.split_scores(recs[method]["test"], scores[kind]["test"])
            errors = {"all": mb.error_breakdown(imgs["test"], recs[method]["test"], sl, -np.inf),
                      "strict_hF": mb.error_breakdown(imgs["test"], recs[method]["test"], sl,
                                                      metrics["test"]["threshold@strict_hF"])}
            cal = cals["base" if kind == "rule" else kind]
            names = ["base"] * 8 + [WHOLENESS_NAMES[c] for c in FEATURE_SETS["base" if kind == "rule" else kind]]
            config = settings | {"method": method, "score": kind, "model": model_key, "k_add": da.K_ADD, "seed": seed,
                                 "split": "original" if seed == ORIGINAL_SEED else "resplit",
                                 "feature_names": names, "calibrator_w": cal.w.tolist(), "calibrator_b": cal.b}
            tb.save_config(dirs[kind], metrics, config, errors, curves["test"] if seed == ORIGINAL_SEED else None)
            rows.append(tb.result_row(method, kind, model_key, seed, metrics, errors))
            if seed == ORIGINAL_SEED:
                if keep:
                    test_parts[(method, kind)] = parts["test"]
                print(model_key, method, kind, {k: round(metrics["test"][k], 4) for k in
                                                ["AP_fed", "AP_strict", "AP_exact", "ECE"]},
                      {k: round(v, 3) for k, v in errors["strict_hF"].items()}, flush=True)
    return rows


def diagnosis(images, records, cache, method="greedy"):
    """
    テスト用の画像の全ての加えた領域を ex015b の誤りの種類に分け，種類ごとに全体らしさの特徴の平均を求める．

    Args:
        images (list[dict]): テスト用の画像．
        records (list[list[dict]]): 画像ごとの記録．
        cache (dict): split_wholeness の cache．
        method (str): 命名の方法．

    Returns:
        dict: 種類 → {"n"，特徴名 → 平均}．
    """
    _, _, _, mb, _ = modules()
    kinds, feats = [], []
    for img, recs in zip(images, records):
        det = np.array([r["det"] for r in recs])
        kinds += mb.classify_added(img, recs, det, -np.inf)
        f = split_wholeness([img], [recs], cache, method)
        feats.append(f[[j for j, r in enumerate(recs) if r["source"] == "owlv2"]])
    feats, kinds = np.concatenate(feats), np.array(kinds)
    out = {}
    for k in mb.ERROR_KINDS:
        sel = kinds == k
        out[k] = {"n": int(sel.sum())} | {n: float(feats[sel, c].mean()) if sel.any() else float("nan")
                                          for c, n in enumerate(WHOLENESS_NAMES)}
    flagged = feats[:, 4] > 0
    out["meronym_flagged"] = {"n": int(flagged.sum())} | {k: float((kinds[flagged] == k).mean()) if flagged.any()
                                                          else float("nan") for k in mb.ERROR_KINDS}
    return out


def run_model(ctx, model_key, root):
    """
    1つのOWLv2の重みについて，全ての方法・スコア・分割の seed を評価する．

    Args:
        ctx (dict): ex015の load_context の戻り値．
        model_key (str): OWLv2の重みの名前．
        root (str): 出力ディレクトリ．

    Returns:
        list[dict]: 結果の表の行．
    """
    da, _, _, mb, _ = modules()
    db = ex015b_modules()[0]
    t15 = da.ex015_modules()[2]
    settings = da.selected_settings()
    owl = t15.OWLv2Regions(model_key)
    namers = db.make_namers(owl, settings)
    images = {s: da.load_images(ctx, s, model_key) for s in SPLITS}
    by_split = {s: build_records(images[s], namers, settings) for s in SPLITS}
    records = {m: {s: by_split[s][m] for s in SPLITS} for m in METHODS}
    del owl, namers
    torch.cuda.empty_cache()
    cache = {}
    with open(os.path.join(root, f"diagnosis_{model_key}.json"), "w") as f:
        json.dump(diagnosis(images["test"], records["greedy"]["test"], cache), f, indent=4)
    rows, test_parts = [], {}
    need_boot = model_key in BOOT_MODELS and not os.path.exists(os.path.join(root, f"bootstrap_{model_key}.json"))
    for seed in tqdm([ORIGINAL_SEED] + RESPLIT_SEEDS, desc=f"{model_key} seeds", leave=False):
        rows += run_seed(images, records, seed, model_key, settings, root, cache, test_parts, need_boot)
    if need_boot:
        totals_rows = mb.image_totals(images["test"])
        boot = {}
        for a, b in tqdm(BOOT_PAIRS, desc="bootstrap", leave=False):
            boot[f"{a[0]}_{a[1]} - {b[0]}_{b[1]}"] = mb.bootstrap_difference(test_parts[a], test_parts[b], totals_rows,
                                                                             N_BOOT, ORIGINAL_SEED)
        with open(os.path.join(root, f"bootstrap_{model_key}.json"), "w") as f:
            json.dump(boot, f, indent=4)
    return rows


def summarize(rows, root):
    """
    元の分割の値と，分け直した5通りの平均・標準偏差の表を作る．

    Args:
        rows (list[dict]): 結果の行．
        root (str): 出力ディレクトリ．

    Returns:
        None
    """
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "table_results.csv"), index=False)
    keys = ["AP_fed", "AP_strict", "AP_exact", "AP_named", "ECE", "strict_hF@strict_hF", "noncoco_named@strict_hF",
            "added_precision@strict_hF", "err_correct", "err_part", "err_background"]
    re = df[df["seed"] != ORIGINAL_SEED].groupby(["model", "method", "score"])[keys].agg(["mean", "std"])
    re.columns = [f"{a}_{b}" for a, b in re.columns]
    orig = df[df["seed"] == ORIGINAL_SEED].set_index(["model", "method", "score"])[keys]
    orig.columns = [f"{c}_seed42" for c in orig.columns]
    orig.join(re).reset_index().to_csv(os.path.join(root, "table_seeds.csv"), index=False)


def main():
    """全てのOWLv2の重みについて評価し，集計表を作成する．"""
    da = modules()[0]
    d15, _, t15 = da.ex015_modules()
    root = os.path.join(t15.OUTPUT_DIR, EXPERIMENT)
    os.makedirs(root, exist_ok=True)
    torch.manual_seed(ORIGINAL_SEED)
    start = time.time()
    ctx = d15.load_context(ORIGINAL_SEED)
    rows = []
    for key in MODELS:
        rows += run_model(ctx, key, root)
        summarize(rows, root)
    print("elapsed", time.time() - start, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
