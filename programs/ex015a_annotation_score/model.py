"""
ex015aのモデル：アノテーションのスコアの計算（検出のスコア，命名の確率，両者の積，校正したスコア）と，
スコアの順に出力を並べた評価（階層的な適合率・再現率の曲線，AP，校正誤差）．
"""
import numpy as np
import torch

from data import ancestors_of, evaluated_set

SCORES = ["det", "name", "det_x_name", "det_x_name_added", "calibrated"]
IOU_THRESHOLD = 0.5


# ===================== スコア =====================

def raw_score(recs, kind):
    """
    学習を伴わないスコアを計算する．

    Args:
        recs (list[dict]): 出力の記録（平坦化したもの）．
        kind (str): "det"（検出のスコア），"name"（出力した単語の集約した確率），"det_x_name"（両者の積），
            "det_x_name_added"（加えた領域のみ両者の積とし，YOLO11xのボックスは検出のスコア）．

    Returns:
        np.ndarray: スコア（N）．
    """
    det = np.array([r["det"] for r in recs], dtype=np.float64)
    q = np.nan_to_num(np.array([r["q"] for r in recs], dtype=np.float64), nan=1.0)
    added = np.array([r["source"] == "owlv2" for r in recs])
    return {"det": det, "name": q, "det_x_name": det * q, "det_x_name_added": np.where(added, det * q, det)}[kind]


class LogisticCalibrator:
    """検出・命名の特徴からアノテーションが正しい確率を推定するロジスティック回帰（検証用の画像のみで重みを求める）．"""

    def __init__(self, l2=1e-3):
        """
        Args:
            l2 (float): 重みのL2正則化の係数．
        """
        self.l2 = l2
        self.w = self.b = self.mean = self.std = None

    def fit(self, x, y):
        """
        重みを求める．

        Args:
            x (np.ndarray): 特徴（N×F）．
            y (np.ndarray): 正解のラベル（N，0または1）．

        Returns:
            LogisticCalibrator: 自身．
        """
        self.mean, self.std = x.mean(axis=0), x.std(axis=0) + 1e-6
        with torch.inference_mode(False), torch.enable_grad():  # 推論モードの中から呼ばれても勾配を計算できるようにする
            self._optimize(torch.tensor((x - self.mean) / self.std, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
        return self

    def _optimize(self, xt, yt):
        """
        L-BFGSで交差エントロピーとL2正則化の和を最小化する．

        Args:
            xt (torch.Tensor): 標準化した特徴（N×F）．
            yt (torch.Tensor): ラベル（N）．

        Returns:
            None
        """
        w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + self.l2 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.w, self.b = w.detach().numpy(), float(b.detach())

    def predict(self, x):
        """
        正しい確率を推定する．

        Args:
            x (np.ndarray): 特徴（N×F）．

        Returns:
            np.ndarray: 確率（N）．
        """
        z = ((x - self.mean) / self.std) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


# ===================== 評価 =====================

def match_greedy(iou, scores):
    """
    スコアの高い出力から順に，IoUが最大の未対応のGT（IoU 0.5以上）に対応付ける（COCOのAPと同じ方法）．

    Args:
        iou (np.ndarray): 出力×GTのIoU．
        scores (np.ndarray): 出力のスコア．

    Returns:
        np.ndarray: 出力ごとの対応するGTの番号（対応しない場合は−1）．
    """
    matched = np.full(len(scores), -1)
    if iou.size == 0:
        return matched
    used = np.zeros(iou.shape[1], dtype=bool)
    for j in np.argsort(-scores, kind="stable"):
        cand = np.where(used, -1.0, iou[j])
        g = int(cand.argmax())
        if cand[g] >= IOU_THRESHOLD:
            matched[j], used[g] = g, True
    return matched


def outcomes(img, recs, matched):
    """
    1枚の画像の出力ごとに，評価に用いる量を求める．

    Args:
        img (dict): 画像（"gts"，"checked_synsets"）．
        recs (list[dict]): 出力の記録．
        matched (np.ndarray): 対応するGTの番号．

    Returns:
        dict[str, np.ndarray]: inter（GTと共通のSynset数），pred（出力のSynset数），fed_extra，strict_extra（誤りとして加えるSynset数），
            named，fine，noncoco（対応したGTがCOCO外か），added（加えた領域か），label（正しい1，誤り0，不明−1）．
    """
    n = len(recs)
    out = {k: np.zeros(n) for k in ["inter", "pred", "fed_extra", "strict_extra", "named", "fine", "noncoco", "added"]}
    out["label"] = np.full(n, -1.0)
    for j, (r, g) in enumerate(zip(recs, matched)):
        syn = r["synset"]
        out["added"][j] = r["source"] == "owlv2"
        if g < 0:
            size = len(evaluated_set(syn))
            out["strict_extra"][j] = size
            if ancestors_of(syn) & img["checked_synsets"]:
                out["fed_extra"][j], out["label"][j] = size, 0.0
            continue
        t = img["gts"][g]["synset"]
        out_set = evaluated_set(t) if t in ancestors_of(syn) else evaluated_set(syn)
        out["inter"][j] = len(out_set & evaluated_set(t))
        out["pred"][j] = len(out_set)
        named = bool(out_set) and (syn in ancestors_of(t) or t in ancestors_of(syn))
        out["named"][j], out["fine"][j] = named, t in ancestors_of(syn)
        out["noncoco"][j] = img["gts"][g]["kind"] == "noncoco"
        out["label"][j] = float(named)
    return out


def gt_totals(images):
    """
    GTの評価用のSynset数の合計と，COCO外・COCOの物体の数を返す．

    Args:
        images (list[dict]): 画像．

    Returns:
        dict: gt_size，noncoco_n，coco_n．
    """
    gts = [g for img in images for g in img["gts"]]
    return {"gt_size": sum(len(evaluated_set(g["synset"])) for g in gts),
            "noncoco_n": sum(g["kind"] == "noncoco" for g in gts), "coco_n": sum(g["kind"] == "coco" for g in gts)}


def collect(images, records, score_list):
    """
    スコアに従って全画像の出力を対応付け，評価に用いる量を平坦化する．

    Args:
        images (list[dict]): 画像．
        records (list[list[dict]]): 画像ごとの出力の記録．
        score_list (list[np.ndarray]): 画像ごとの出力のスコア．

    Returns:
        dict[str, np.ndarray]: outcomes の各量と "score"（全出力を連結したもの）．
    """
    parts = []
    for img, recs, s in zip(images, records, score_list):
        o = outcomes(img, recs, match_greedy(img["iou"], s))
        o["score"] = s
        parts.append(o)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def curve(flat, totals):
    """
    スコアの閾値を下げながら出力を加えた場合の指標の推移を求める．

    Args:
        flat (dict): collect の戻り値．
        totals (dict): gt_totals の戻り値．

    Returns:
        dict[str, np.ndarray]: score（閾値），hP，fed_hP，strict_hP，hR，fed_hF，strict_hF，noncoco_named，noncoco_fine，
            coco_named，added_precision，n_kept．
    """
    order = np.argsort(-flat["score"], kind="stable")
    c = {k: np.cumsum(v[order]) for k, v in flat.items() if k != "score"}
    eps = 1e-12
    inter, pred = c["inter"], c["pred"]
    h_r = inter / totals["gt_size"]
    res = {"score": flat["score"][order], "hP": inter / (pred + eps), "fed_hP": inter / (pred + c["fed_extra"] + eps),
           "strict_hP": inter / (pred + c["strict_extra"] + eps), "hR": h_r, "n_kept": np.arange(1, len(order) + 1)}
    for k in ["fed", "strict"]:
        p = res[f"{k}_hP"]
        res[f"{k}_hF"] = 2 * p * h_r / (p + h_r + eps)
    named_noncoco = np.cumsum((flat["named"] * flat["noncoco"])[order])
    named_coco = np.cumsum((flat["named"] * (1 - flat["noncoco"]) * (flat["pred"] > 0))[order])
    res["noncoco_named"] = named_noncoco / totals["noncoco_n"]
    res["noncoco_fine"] = np.cumsum((flat["fine"] * flat["noncoco"])[order]) / totals["noncoco_n"]
    res["coco_named"] = named_coco / totals["coco_n"]
    res["added_precision"] = np.cumsum((flat["named"] * flat["added"])[order]) / np.maximum(c["added"], 1)
    return res


def average_precision(recall, precision):
    """
    適合率を右側の最大値で単調にした上で，再現率の増分で積分したAPを返す．

    Args:
        recall (np.ndarray): 再現率（非減少）．
        precision (np.ndarray): 適合率．

    Returns:
        float: AP．
    """
    p = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate([[0.0], recall])
    return float(np.sum((r[1:] - r[:-1]) * p))


def at_threshold(cur, thr):
    """
    スコアが閾値以上の出力のみを残した場合の指標を返す．

    Args:
        cur (dict): curve の戻り値．
        thr (float): 閾値．

    Returns:
        dict: 指標．
    """
    idx = int(np.searchsorted(-cur["score"], -thr, side="right")) - 1
    idx = max(idx, 0)
    keys = ["hP", "fed_hP", "strict_hP", "hR", "fed_hF", "strict_hF", "noncoco_named", "noncoco_fine", "coco_named",
            "added_precision", "n_kept"]
    return {k: float(cur[k][idx]) for k in keys} | {"threshold": float(thr)}


def best_threshold(cur, key):
    """
    指標が最大となるスコアの閾値を返す．

    Args:
        cur (dict): curve の戻り値．
        key (str): 指標名．

    Returns:
        float: 閾値．
    """
    return float(cur["score"][int(np.argmax(cur[key]))])


def expected_calibration_error(scores, labels, bins=10):
    """
    スコアを確率とみなした場合の校正誤差（ECE，等幅のビン）を返す．ラベルが不明（−1）の出力は除く．

    Args:
        scores (np.ndarray): スコア（0〜1）．
        labels (np.ndarray): ラベル．
        bins (int): ビンの数．

    Returns:
        tuple[float, list[dict]]: ECE と，ビンごとの {"lo"，"hi"，"n"，"score"，"accuracy"}．
    """
    keep = labels >= 0
    s, y = np.clip(scores[keep], 0, 1), labels[keep]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(s[m].mean() - y[m].mean())
        table.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()), "score": float(s[m].mean()),
                      "accuracy": float(y[m].mean())})
    return float(ece), table
