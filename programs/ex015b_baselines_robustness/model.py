"""
ex015bのモデル：命名の方法とスコアの比較に用いる評価．
- ラベルの正誤による AP（AP_exact：GTのSynset自身か下位語，AP_named：GTの上位語・自身・下位語）
- 画像単位のブートストラップによる差の信頼区間
- 加えた領域の出力の誤りの分解
"""
import numpy as np

from data import ex015a_modules

IOU_MATCH = 0.5
IOU_LOC = 0.1          # これ以上 IOU_MATCH 未満を位置ずれとする
CONTAIN = 0.8          # 領域の面積のうちGTのボックスに含まれる割合がこれ以上であれば物体の一部とする
ERROR_KINDS = ["correct", "misnamed", "duplicate", "part", "localization", "background"]


def per_image_outcomes(images, records, score_list):
    """
    画像ごとに，スコアの順に出力をGTへ対応付けた評価の量を求める（ブートストラップで画像を選び直すため，画像ごとに保持する）．

    Args:
        images (list[dict]): 画像．
        records (list[list[dict]]): 画像ごとの出力の記録．
        score_list (list[np.ndarray]): 画像ごとのスコア．

    Returns:
        list[dict[str, np.ndarray]]: 画像ごとの ex015a の outcomes の戻り値に "score" を加えたもの．
    """
    ma = ex015a_modules()[1]
    parts = []
    for img, recs, s in zip(images, records, score_list):
        o = ma.outcomes(img, recs, ma.match_greedy(img["iou"], s))
        o["score"] = np.asarray(s, dtype=np.float64)
        parts.append(o)
    return parts


def image_totals(images):
    """
    画像ごとのGTの量（ex015a の gt_totals を1枚ずつ計算したもの）を返す．

    Args:
        images (list[dict]): 画像．

    Returns:
        np.ndarray: 画像ごとの [gt_size，noncoco_n，coco_n]（画像数×3）．
    """
    ma = ex015a_modules()[1]
    rows = []
    for img in images:
        t = ma.gt_totals([img])
        rows.append([t["gt_size"], t["noncoco_n"], t["coco_n"]])
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def label_ap(flat, n_gt):
    """
    ラベルの正誤のみで判定する AP を求める．IoU 0.5 でGTと対応し，出力した語が正しい出力を正解とし，
    適合率は正解の数を残した出力の数で，再現率は正解の数をGTの物体の数で割る（カテゴリを区別しない COCO 形式の AP@0.5）．

    Args:
        flat (dict): 全出力の outcomes（"fine"，"named"，"pred"，"score"）．
        n_gt (float): GTの物体の数．

    Returns:
        dict[str, float]: AP_exact（GTのSynset自身または下位語），AP_named（GTの上位語・自身・下位語）．
    """
    ma = ex015a_modules()[1]
    order = np.argsort(-flat["score"], kind="stable")
    n = np.arange(1, len(order) + 1)
    out = {}
    for key, tp in [("AP_exact", flat["fine"]), ("AP_named", flat["named"] * (flat["pred"] > 0))]:
        c = np.cumsum(tp[order])
        out[key] = ma.average_precision(c / n_gt, c / n)
    return out


def metrics_from_parts(parts, totals_rows):
    """
    画像ごとの評価の量から，AP_fed，AP_strict，AP_exact，AP_named を求める．

    Args:
        parts (list[dict]): per_image_outcomes の戻り値（の部分集合，重複可）．
        totals_rows (np.ndarray): 対応する画像の image_totals の行（画像数×3）．

    Returns:
        dict[str, float]: 指標．
    """
    ma = ex015a_modules()[1]
    flat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    s = totals_rows.sum(axis=0)
    totals = {"gt_size": s[0], "noncoco_n": s[1], "coco_n": s[2]}
    cur = ma.curve(flat, totals)
    return {"AP_fed": ma.average_precision(cur["hR"], cur["fed_hP"]),
            "AP_strict": ma.average_precision(cur["hR"], cur["strict_hP"])} | label_ap(flat, s[1] + s[2])


def bootstrap_difference(parts_a, parts_b, totals_rows, n_boot, seed):
    """
    画像単位のブートストラップで，2つの構成（同じ画像の集合）の指標の差（a − b）の分布を求める．
    両者に同じ画像の選び直しを用いる（対応のある比較）．

    Args:
        parts_a (list[dict]): 構成aの per_image_outcomes の戻り値．
        parts_b (list[dict]): 構成bの per_image_outcomes の戻り値．
        totals_rows (np.ndarray): image_totals の戻り値．
        n_boot (int): 選び直しの回数．
        seed (int): 乱数のシード．

    Returns:
        dict[str, dict]: 指標 → {"diff"（元の画像での差），"mean"，"lo"，"hi"（95%区間），"p_le_0"（差が0以下の割合）}．
    """
    rng = np.random.default_rng(seed)
    base_a, base_b = metrics_from_parts(parts_a, totals_rows), metrics_from_parts(parts_b, totals_rows)
    diffs = {k: [] for k in base_a}
    n = len(parts_a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma_ = metrics_from_parts([parts_a[i] for i in idx], totals_rows[idx])
        mb_ = metrics_from_parts([parts_b[i] for i in idx], totals_rows[idx])
        for k in diffs:
            diffs[k].append(ma_[k] - mb_[k])
    out = {}
    for k, v in diffs.items():
        v = np.array(v)
        out[k] = {"diff": base_a[k] - base_b[k], "mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5)), "p_le_0": float((v <= 0).mean())}
    return out


def _containment(box, gts):
    """
    領域の面積のうち，各GTのボックスに含まれる割合を返す．

    Args:
        box (list[float]): 領域（x0，y0，x1，y1）．
        gts (list[dict]): GT（"box"）．

    Returns:
        np.ndarray: GTごとの割合．
    """
    if not gts:
        return np.zeros(0)
    g = np.asarray([x["box"] for x in gts], dtype=np.float64).reshape(-1, 4)
    w = np.clip(np.minimum(box[2], g[:, 2]) - np.maximum(box[0], g[:, 0]), 0, None)
    h = np.clip(np.minimum(box[3], g[:, 3]) - np.maximum(box[1], g[:, 1]), 0, None)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    return w * h / area


def classify_added(img, recs, scores, threshold):
    """
    1枚の画像の加えた領域のうちスコアが閾値以上の出力を，誤りの種類に分ける．

    Args:
        img (dict): 画像（"boxes"，"gts"，"iou"）．
        recs (list[dict]): 出力の記録．
        scores (np.ndarray): 出力のスコア（全出力について対応付けに用いる）．
        threshold (float): 残す出力のスコアの下限．

    Returns:
        list[str]: 残した加えた領域ごとの種類（ERROR_KINDS のいずれか）．
            correct：GTと対応し命名が正しい，misnamed：GTと対応したが命名が誤り，duplicate：IoU 0.5以上のGTが他の出力に対応済み，
            part：GTのボックスに含まれる，localization：GTとのIoUが0.1以上0.5未満，background：それ以外（背景または注釈の無い物体）．
    """
    ma = ex015a_modules()[1]
    da = ex015a_modules()[0]
    keep = np.asarray(scores) >= threshold
    # 閾値未満の出力は取り除いた上で対応付ける
    s = np.where(keep, scores, -np.inf)
    iou = np.where(keep[:, None], img["iou"], 0.0) if img["iou"].size else img["iou"]
    matched = ma.match_greedy(iou, s)
    out = []
    for j, r in enumerate(recs):
        if r["source"] != "owlv2" or not keep[j]:
            continue
        g = matched[j]
        if g >= 0:
            # ex015a の outcomes と同じ命名の判定
            t, syn = img["gts"][g]["synset"], r["synset"]
            out_set = da.evaluated_set(t) if t in da.ancestors_of(syn) else da.evaluated_set(syn)
            ok = bool(out_set) and (syn in da.ancestors_of(t) or t in da.ancestors_of(syn))
            out.append("correct" if ok else "misnamed")
            continue
        best = float(img["iou"][j].max()) if img["iou"].size else 0.0
        if best >= IOU_MATCH:
            out.append("duplicate")
        elif (_containment(img["boxes"][j]["box"], img["gts"]) >= CONTAIN).any():
            out.append("part")
        elif best >= IOU_LOC:
            out.append("localization")
        else:
            out.append("background")
    return out


def error_breakdown(images, records, score_list, threshold):
    """
    加えた領域の出力の誤りの種類の割合を求める．

    Args:
        images (list[dict]): 画像．
        records (list[list[dict]]): 画像ごとの記録．
        score_list (list[np.ndarray]): 画像ごとのスコア．
        threshold (float): 残す出力のスコアの下限．

    Returns:
        dict: n（残した加えた領域の数）と種類ごとの割合．
    """
    kinds = [k for img, recs, s in zip(images, records, score_list) for k in classify_added(img, recs, s, threshold)]
    n = len(kinds)
    return {"n": n} | {k: (sum(x == k for x in kinds) / n if n else float("nan")) for k in ERROR_KINDS}
