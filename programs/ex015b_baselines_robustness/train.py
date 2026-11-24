"""
ex015bの実行スクリプト：命名の方法の比較，ラベルの正誤による AP，分割と校正のばらつき，誤りの分解（order_021）．

ex015で選んだ設定（ens の文章，人の役割を除く，β = 2，τ = 0.1，加える領域 K = 100）で，加えた領域の命名の方法を以下の4通りとし，
全てに ex015a と同じ形の校正したスコア（検証用の画像のみで9個の重みを求める）を付けて，スコアの順の曲線で比較する．
物体認識モデル・埋め込みモデルの再学習は行わない．YOLO11xのボックスの出力は全ての方法で同じである．
- greedy（提案）：キャプションの語彙（7806語）の確率をWordNetの上位語へ集約し，確信度 τ 以上の間だけ下位語へ進む
- flat：キャプションの語彙から確率が最大の1語を出力する（階層を用いない）
- greedy_lvis，flat_lvis：OWLv2にLVISの1203カテゴリ名を与えた場合（評価の語彙を事前に知る，参考）
スコアは det（検出のスコア）と calibrated（校正したスコア）を比べる．
評価：AP_fed，AP_strict（階層的な適合率・再現率），AP_exact，AP_named（ラベルの正誤のみ，カテゴリを区別しない COCO 形式の AP@0.5），ECE，
検証用の画像で hF_strict が最大となる閾値での指標．
ばらつき：元の分割（seed 42）に加え，検証用とテスト用の画像を合わせて seed 0〜4 で半分ずつに分け直し，重みを求め直す．
元のテスト用の画像では，画像単位のブートストラップで差の95%信頼区間を求める．
誤りの分解：加えた領域の出力を，正しい命名，誤った命名，重複，物体の一部，位置ずれ，背景・注釈の無い物体に分ける．
上限の見積もり：greedy，flat について，GTとIoU 0.5以上で重なる加えた領域のみを残した場合（{方法}_oracle，GTを用いるため分析のみ）．
既に結果（errors.json）がある条件は読み込み，ブートストラップの結果がある重みは再計算しない．

出力: outputs/ex015b_baselines_robustness/{方法}_{スコア}/{OWLv2の重み}/{seed}/metrics.json（dev，test），config.json，errors.json，
    curve_test.csv，および直下の table_results.csv，table_seeds.csv，bootstrap_{重み}.json，paired_naming_{重み}.json
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
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import METHODS, build_all_records, ex015a_modules, make_namers  # noqa: E402
from model import bootstrap_difference, error_breakdown, image_totals, label_ap, per_image_outcomes  # noqa: E402

# ===== 実験条件（order_021） =====
EXPERIMENT = "ex015b_baselines_robustness"
ORIGINAL_SEED = 42                 # ex015と同じ分割（load_context の seed）
RESPLIT_SEEDS = [0, 1, 2, 3, 4]    # 検証用とテスト用の画像を合わせて分け直す seed
MODELS = ["st", "st_large", "ens"]
SCORE_KINDS = ["det", "calibrated"]
SPLITS = ["dev", "test"]
N_BOOT = 1000
BOOT_MODELS = ["st", "ens"]
ORACLE_METHODS = ["greedy", "flat"]   # 物体でない領域を除いた場合の上限の見積もり
# ブートストラップで比べる組（a − b）
BOOT_PAIRS = [(("greedy", "calibrated"), ("greedy", "det")), (("greedy", "calibrated"), ("flat", "calibrated")),
              (("greedy", "det"), ("flat", "det")), (("greedy_lvis", "calibrated"), ("flat_lvis", "calibrated")),
              (("greedy", "calibrated"), ("flat_lvis", "calibrated"))]
THRESHOLD_KEYS = ["strict_hF", "fed_hF", "noncoco_named", "noncoco_fine", "coco_named", "added_precision", "n_kept",
                  "strict_hP", "fed_hP", "hR"]


def calibrated_scores(images, records):
    """
    検証用の画像で，検出のスコアの順に対応付けた場合の正誤を教師としてロジスティック回帰の重みを求め，全分割のスコアを返す（ex015aと同じ）．

    Args:
        images (dict): 分割 → 画像．
        records (dict): 分割 → 画像ごとの記録．

    Returns:
        tuple[dict, LogisticCalibrator]: 分割 → 全出力のスコア，求めたモデル．
    """
    da, ma, ta = ex015a_modules()
    flat_dev = ta.flatten(records["dev"])
    dev_det = ma.collect(images["dev"], records["dev"], ta.split_scores(records["dev"], ma.raw_score(flat_dev, "det")))
    x, y = da.features(flat_dev), dev_det["label"]
    cal = ma.LogisticCalibrator().fit(x[y >= 0], y[y >= 0])
    return {s: cal.predict(da.features(ta.flatten(records[s]))) for s in SPLITS}, cal


def evaluate(images, records, scores):
    """
    1つの構成（方法とスコア）について，分割ごとの指標を求める．

    Args:
        images (dict): 分割 → 画像．
        records (dict): 分割 → 画像ごとの記録．
        scores (dict): 分割 → 全出力のスコア．

    Returns:
        tuple[dict, dict, dict]: 分割 → 指標，分割 → 曲線，分割 → 画像ごとの評価の量（per_image_outcomes）．
    """
    _, ma, ta = ex015a_modules()
    metrics, curves, parts = {}, {}, {}
    for s in SPLITS:
        sl = ta.split_scores(records[s], scores[s])
        parts[s] = per_image_outcomes(images[s], records[s], sl)
        flat = {k: np.concatenate([p[k] for p in parts[s]]) for k in parts[s][0]}
        totals = ma.gt_totals(images[s])
        cur = ma.curve(flat, totals)
        ece, _ = ma.expected_calibration_error(flat["score"], flat["label"])
        metrics[s] = {"AP_fed": ma.average_precision(cur["hR"], cur["fed_hP"]),
                      "AP_strict": ma.average_precision(cur["hR"], cur["strict_hP"]), "ECE": ece,
                      "max_fed_hF": float(cur["fed_hF"].max()), "max_strict_hF": float(cur["strict_hF"].max())}
        metrics[s] |= label_ap(flat, totals["noncoco_n"] + totals["coco_n"])
        curves[s] = cur
    thr = ma.best_threshold(curves["dev"], "strict_hF")
    for s in SPLITS:
        at = ma.at_threshold(curves[s], thr)
        metrics[s] |= {f"{k}@strict_hF": at[k] for k in THRESHOLD_KEYS} | {"threshold@strict_hF": thr}
    return metrics, curves, parts


def split_sets(images, records, seed):
    """
    分割を返す．seed が ORIGINAL_SEED の場合は元の分割，それ以外は検証用とテスト用の画像を合わせて半分ずつに分け直す．

    Args:
        images (dict): 元の分割 → 画像．
        records (dict): 方法 → 元の分割 → 画像ごとの記録．
        seed (int): シード．

    Returns:
        tuple[dict, dict]: 分割 → 画像，方法 → 分割 → 画像ごとの記録．
    """
    if seed == ORIGINAL_SEED:
        return images, records
    pool = images["dev"] + images["test"]
    perm = np.random.default_rng(seed).permutation(len(pool))
    half = len(pool) // 2
    idx = {"dev": perm[:half], "test": perm[half:]}
    imgs = {s: [pool[i] for i in idx[s]] for s in SPLITS}
    recs = {}
    for m, r in records.items():
        rp = r["dev"] + r["test"]
        recs[m] = {s: [rp[i] for i in idx[s]] for s in SPLITS}
    return imgs, recs


def save_config(d, metrics, config, errors=None, curve=None):
    """
    1構成の結果を保存する．

    Args:
        d (str): 保存先．
        metrics (dict): 分割 → 指標．
        config (dict): 設定．
        errors (dict | None): 誤りの分解．
        curve (dict | None): テスト用の曲線．

    Returns:
        None
    """
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    if errors is not None:
        with open(os.path.join(d, "errors.json"), "w") as f:
            json.dump(errors, f, indent=4)
    if curve is not None:
        ex015a_modules()[2].save_curve(curve, os.path.join(d, "curve_test.csv"))


def paired_naming(images, records, n_examples=30):
    """
    テスト用の画像で，検出のスコアの順にCOCO外の物体と対応した加えた領域について，greedy と flat の命名の正誤を対にして数える．

    Args:
        images (list[dict]): テスト用の画像．
        records (dict): 方法 → テスト用の画像ごとの記録．
        n_examples (int): 保存する例の数（種類ごと）．

    Returns:
        dict: 正誤の組の数と例（GTの語，flat の語，greedy の経路）．
    """
    da, ma, _ = ex015a_modules()
    word = lambda s: wn.synset(s).lemma_names()[0].replace("_", " ")  # noqa: E731
    counts = {"both": 0, "greedy_only": 0, "flat_only": 0, "neither": 0}
    examples = {k: [] for k in counts}
    for img, rg, rf in zip(images, records["greedy"], records["flat"]):
        matched = ma.match_greedy(img["iou"], np.array([r["det"] for r in rg]))
        og = ma.outcomes(img, rg, matched)
        of = ma.outcomes(img, rf, matched)
        for j, g in enumerate(matched):
            if rg[j]["source"] != "owlv2" or g < 0 or img["gts"][g]["kind"] != "noncoco":
                continue
            key = {(1, 1): "both", (1, 0): "greedy_only", (0, 1): "flat_only", (0, 0): "neither"}[
                (int(og["named"][j]), int(of["named"][j]))]
            counts[key] += 1
            if len(examples[key]) < n_examples:
                examples[key].append({"image_id": img["image_id"], "gt": word(img["gts"][g]["synset"]),
                                      "flat": word(rf[j]["synset"]), "greedy": [word(n) for n, _ in rg[j]["chain"]]})
    n = sum(counts.values())
    return {"n": n, "counts": counts, "greedy_named": (counts["both"] + counts["greedy_only"]) / max(n, 1),
            "flat_named": (counts["both"] + counts["flat_only"]) / max(n, 1), "examples": examples}


def run_model(ctx, model_key, root):
    """
    1つのOWLv2の重みについて，全ての方法・スコア・分割の seed を評価する．

    Args:
        ctx (dict): ex015の load_context の戻り値．
        model_key (str): OWLv2の重みの名前．
        root (str): 実験の出力ディレクトリ．

    Returns:
        list[dict]: 結果の表の行．
    """
    da, _, _ = ex015a_modules()
    t15 = da.ex015_modules()[2]
    settings = da.selected_settings()
    owl = t15.OWLv2Regions(model_key)
    namers = make_namers(owl, settings)
    images = {s: da.load_images(ctx, s, model_key) for s in SPLITS}
    by_split = {s: build_all_records(images[s], namers, settings) for s in SPLITS}
    records = {m: {s: by_split[s][m] for s in SPLITS} for m in METHODS}
    del owl, namers
    torch.cuda.empty_cache()
    with open(os.path.join(root, f"paired_naming_{model_key}.json"), "w") as f:
        json.dump(paired_naming(images["test"], {m: records[m]["test"] for m in ["greedy", "flat"]}), f, indent=1)
    variants = {m: (images, records[m]) for m in METHODS}
    # 上限の見積もり：GTとIoU 0.5以上で重なる加えた領域のみを残す（GTを用いるため，手法ではなく誤りの要因の分析に用いる）
    for m in ORACLE_METHODS:
        variants[f"{m}_oracle"] = oracle_regions(images, records[m])
    rows, test_parts = [], {}
    need_boot = model_key in BOOT_MODELS and not os.path.exists(os.path.join(root, f"bootstrap_{model_key}.json"))
    for seed in tqdm([ORIGINAL_SEED] + RESPLIT_SEEDS, desc=f"{model_key} seeds", leave=False):
        for name, (v_images, v_records) in variants.items():
            rows += run_variant(name, v_images, v_records, seed, model_key, settings, root, test_parts, need_boot)
    if need_boot:
        totals_rows = image_totals(images["test"])
        boot = {}
        for a, b in tqdm(BOOT_PAIRS, desc="bootstrap", leave=False):
            boot[f"{a[0]}_{a[1]} - {b[0]}_{b[1]}"] = bootstrap_difference(test_parts[a], test_parts[b], totals_rows, N_BOOT,
                                                                          ORIGINAL_SEED)
        with open(os.path.join(root, f"bootstrap_{model_key}.json"), "w") as f:
            json.dump(boot, f, indent=4)
    return rows


def oracle_regions(images, records):
    """
    加えた領域のうち，いずれかのGTとIoU 0.5以上で重なるもののみを残した画像と記録を返す．

    Args:
        images (dict): 分割 → 画像．
        records (dict): 分割 → 画像ごとの記録．

    Returns:
        tuple[dict, dict]: 分割 → 画像，分割 → 画像ごとの記録．
    """
    out_i, out_r = {}, {}
    for s in SPLITS:
        out_i[s], out_r[s] = [], []
        for img, recs in zip(images[s], records[s]):
            keep = [j for j in range(len(recs)) if j < img["n_yolo"]
                    or (img["iou"].size and img["iou"][j].max() >= 0.5)]
            out_i[s].append(img | {"boxes": [img["boxes"][j] for j in keep], "iou": img["iou"][keep],
                                   "feats": img["feats"][keep]})
            out_r[s].append([recs[j] for j in keep])
    return out_i, out_r


def run_variant(name, images, records, seed, model_key, settings, root, test_parts, need_boot):
    """
    1つの方法について，1つの分割の seed で全てのスコアを評価する．結果が既にある場合は読み込む．

    Args:
        name (str): 方法の名前．
        images (dict): 元の分割 → 画像．
        records (dict): 元の分割 → 画像ごとの記録．
        seed (int): 分割の seed．
        model_key (str): OWLv2の重みの名前．
        settings (dict): ex015で選んだ設定．
        root (str): 実験の出力ディレクトリ．
        test_parts (dict): （方法，スコア）→ 元のテスト用の画像の評価の量（ブートストラップ用，追記する）．
        need_boot (bool): ブートストラップのために評価の量を保持するか．

    Returns:
        list[dict]: 結果の表の行．
    """
    da, ma, ta = ex015a_modules()
    base = name.replace("_oracle", "")
    rows = []
    dirs = {k: os.path.join(root, f"{name}_{k}", model_key, str(seed)) for k in SCORE_KINDS}
    keep_parts = need_boot and seed == ORIGINAL_SEED and any((name, k) in p for p in BOOT_PAIRS for k in SCORE_KINDS)
    if not keep_parts and all(os.path.exists(os.path.join(d, "errors.json")) for d in dirs.values()):
        for kind, d in dirs.items():
            metrics, errors = json.load(open(os.path.join(d, "metrics.json"))), json.load(open(os.path.join(d, "errors.json")))
            rows.append(result_row(name, kind, model_key, seed, metrics, errors))
        return rows
    imgs, recs = split_sets(images, {name: records}, seed)
    recs = recs[name]
    cal_scores, cal = calibrated_scores(imgs, recs)
    for kind in SCORE_KINDS:
        flat_recs = {s: ta.flatten(recs[s]) for s in SPLITS}
        scores = cal_scores if kind == "calibrated" else {s: ma.raw_score(flat_recs[s], "det") for s in SPLITS}
        metrics, curves, parts = evaluate(imgs, recs, scores)
        sl = ta.split_scores(recs["test"], scores["test"])
        errors = {"all": error_breakdown(imgs["test"], recs["test"], sl, -np.inf),
                  "strict_hF": error_breakdown(imgs["test"], recs["test"], sl, metrics["test"]["threshold@strict_hF"])}
        config = settings | {"method": name, "vocab": METHODS[base][0], "rule": METHODS[base][1], "score": kind,
                             "oracle_regions": name.endswith("_oracle"), "model": model_key, "k_add": da.K_ADD, "seed": seed,
                             "split": "original" if seed == ORIGINAL_SEED else "resplit"}
        if kind == "calibrated":
            config |= {"calibrator_w": cal.w.tolist(), "calibrator_b": cal.b}
        save_config(dirs[kind], metrics, config, errors, curves["test"] if seed == ORIGINAL_SEED else None)
        rows.append(result_row(name, kind, model_key, seed, metrics, errors))
        if seed == ORIGINAL_SEED:
            if keep_parts:
                test_parts[(name, kind)] = parts["test"]
            print(model_key, name, kind, {k: round(metrics["test"][k], 4) for k in
                                          ["AP_fed", "AP_strict", "AP_exact", "AP_named", "ECE"]}, flush=True)
    return rows


def result_row(name, kind, model_key, seed, metrics, errors):
    """
    結果の表の1行を作る．

    Args:
        name (str): 方法．
        kind (str): スコア．
        model_key (str): OWLv2の重みの名前．
        seed (int): 分割の seed．
        metrics (dict): 分割 → 指標．
        errors (dict): 誤りの分解．

    Returns:
        dict: 行．
    """
    return ({"method": name, "score": kind, "model": model_key, "seed": seed} | metrics["test"]
            | {f"err_{k}": v for k, v in errors["strict_hF"].items()} | {f"err_all_{k}": v for k, v in errors["all"].items()})


def summarize(rows, root):
    """
    分割の seed ごとの結果から，元の分割の値と分け直した5通りの平均・標準偏差の表を作る．

    Args:
        rows (list[dict]): 結果の行．
        root (str): 実験の出力ディレクトリ．

    Returns:
        None
    """
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "table_results.csv"), index=False)
    keys = ["AP_fed", "AP_strict", "AP_exact", "AP_named", "ECE", "strict_hF@strict_hF", "noncoco_named@strict_hF",
            "added_precision@strict_hF"]
    re = df[df["seed"] != ORIGINAL_SEED].groupby(["model", "method", "score"])[keys].agg(["mean", "std"])
    re.columns = [f"{a}_{b}" for a, b in re.columns]
    orig = df[df["seed"] == ORIGINAL_SEED].set_index(["model", "method", "score"])[keys]
    orig.columns = [f"{c}_seed42" for c in orig.columns]
    orig.join(re).reset_index().to_csv(os.path.join(root, "table_seeds.csv"), index=False)


def main():
    """全てのOWLv2の重みについて評価し，集計表を作成する．"""
    da = ex015a_modules()[0]
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
