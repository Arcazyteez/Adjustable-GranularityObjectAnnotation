"""
ex017：main.pdf と 進捗報告_2026B.pdf の実験の実行（order_027）．

学習は行わず，学習済みの D と F による推論と評価のみを行う．各条件の結果は
    outputs/ex017_coco_granular_annotation/{実験}/{手法}/{条件}/{seed}/
に保存し，metrics.json（または annotations.json）が既にある条件はスキップする．

実験：
- E1_vlm（論文4.2節）：F の比較．画像全体に対するタスクA（COCO 80クラス）とタスクB（キャプションの名詞・動詞）
- E2_detector（論文4.3節）：D の比較．クラス非依存の Recall@IoU0.5，平均最良IoU，提案数，検出時間
- E3_quantitative（論文4.4節，進捗報告の定量評価）：初期の語彙をCOCO 80クラス，下位語なしとした提案手法と
  Close Vocabularyモデル（YOLO11x，DETR）の比較．θ は dev の F値で選び，test で報告する
- E4_depth（論文4.5節）：WordNet の深さごとのインスタンス数・候補数・平均スコア
- E5_open_vocab（論文4.6節）：COCO外の語を含む画像のアノテーション
- E6_qualitative（論文4.7節，進捗報告の定性評価）：初期の語彙をCOCO 80クラスまたはWordNetの上位語とした階層的アノテーション

並列化について：すべての処理は1枚のGPU（12GB）上のモデルの推論であり，並列数1での使用量（VRAM 約3GB）でも
推論自体がGPUを占有するため，マルチプロセス化はせず，バッチ化とキャッシュで高速化する．

order_027：COCO val2017 から seed 42 で1000枚（dev 500，test 500）．D ∈ {DETR(0.5), YOLO11x(0.25)}，
F ∈ {ResNet50-IN1k, CLIP-RN50, CLIP-ViT-B32, SigLIP-ViT-B16}，スコアは cosine（進捗報告）と softmax（論文）．切り抜き画像は黒で正方形にパディングする（data.make_crop）．
"""
import itertools
import json
import os
import textwrap
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import (DATASET_DIR, OUTPUT_DIR, caption_words, gt_instances, make_crop, load_coco, load_image,  # noqa: E402
                  prepare_open_vocab_images, root_vocab, split_image_ids)
from model import granular_annotation, load_detector, load_embedder, set_seed, timed_detect  # noqa: E402

SEED = 0
SPLITS = ["dev", "test"]
IOU_THRESHOLD = 0.5
N_BOOTSTRAP = 1000

E1_EMBEDDERS = ["ResNet50-IN1k", "CLIP-RN50", "CLIP-ViT-B32", "SigLIP-ViT-B16"]
E2_DETECTORS = ["DETR", "YOLO11x", "OWL-ViT", "Grounding DINO"]
E3_DETECTORS = ["DETR", "YOLO11x"]
E3_EMBEDDERS = ["CLIP-ViT-B32", "SigLIP-ViT-B16"]
E3_MODES = ["cosine", "softmax"]
THETA_QUANTILES = np.linspace(0.05, 0.95, 19)   # θ の候補（dev の最大スコアの分位点）．θ = −∞ も加える
E4_MAX_DEPTH = None                              # None：下位語が無くなるまで進む
N_QUALITATIVE = 6


# ---------------------------------------------------------------------------
# 共通の関数
# ---------------------------------------------------------------------------
def save_json(obj, path):
    """
    辞書をJSONで保存する（ディレクトリは自動で作る）．

    Args:
        obj (dict | list): 保存する内容．
        path (str): 保存先のパス．
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def result_dir(experiment, method, condition, seed=SEED):
    """
    結果の保存先ディレクトリを返す．

    Args:
        experiment (str): 実験名（例: "E3_quantitative"）．
        method (str): 比較する手法．
        condition (str): 条件（分割や語彙）．
        seed (int): シード．

    Returns:
        str: outputs/ex017_coco_granular_annotation/{experiment}/{method}/{condition}/{seed}
    """
    return os.path.join(OUTPUT_DIR, experiment, method.replace(" ", "_"), condition, str(seed))


def box_iou(a, b):
    """
    2組のボックスのIoU行列を計算する．

    Args:
        a (np.ndarray): xyxy形式のボックス（N×4）．
        b (np.ndarray): xyxy形式のボックス（M×4）．

    Returns:
        np.ndarray: IoU（N×M）．
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    import torchvision
    return torchvision.ops.box_iou(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)).numpy()


def greedy_match(iou, pred_order, allowed=None, threshold=IOU_THRESHOLD):
    """
    スコアの高い予測から順に，未対応のGTのうちIoUが最大のものに対応付ける．

    Args:
        iou (np.ndarray): IoU（予測数P×GT数G）．
        pred_order (list[int]): 予測を処理する順（スコアの降順）．
        allowed (np.ndarray | None): 対応付けてよい組（P×G，bool）．None なら全組．
        threshold (float): 対応とみなすIoUの下限．

    Returns:
        dict[int, int]: 予測のインデックス → GTのインデックス．
    """
    used, pairs = set(), {}
    for p in pred_order:
        cand = iou[p].copy()
        if allowed is not None:
            cand[~allowed[p]] = -1
        cand[list(used)] = -1
        g = int(np.argmax(cand)) if cand.size else -1
        if g >= 0 and cand[g] >= threshold:
            pairs[p] = g
            used.add(g)
    return pairs


def bootstrap_ci(per_image, ratio, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    画像単位のブートストラップで比の指標の95%信頼区間を求める．

    Args:
        per_image (np.ndarray): 画像ごとの [分子, 分母]（N×2）．
        ratio (callable): 合計した [分子, 分母] から指標を求める関数．
        n_bootstrap (int): リサンプリング回数．
        seed (int): 乱数シード．

    Returns:
        list[float]: [下限, 上限]．
    """
    rng = np.random.default_rng(seed)
    stats = [ratio(per_image[rng.integers(0, len(per_image), len(per_image))].sum(axis=0)) for _ in range(n_bootstrap)]
    return [float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))]


def safe_div(a, b):
    """
    0除算のときNaNを返す割り算．

    Args:
        a (float): 分子．
        b (float): 分母．

    Returns:
        float: a / b（b = 0 なら NaN）．
    """
    return float(a) / float(b) if b else float("nan")


# ---------------------------------------------------------------------------
# キャッシュ（検出，切り抜き画像の特徴）
# ---------------------------------------------------------------------------
def cached_detections(coco, detector_name, image_ids, tag, queries=None):
    """
    検出結果をキャッシュから読み込み，無ければ検出して保存する．

    Args:
        coco (dict): load_coco の戻り値．
        detector_name (str): 検出器の名前．
        image_ids (list[int]): 画像IDのリスト．
        tag (str): キャッシュのファイル名の識別子（例: "test"）．
        queries (list[str] | None): テキストクエリ（OWL-ViT，Grounding DINO）．

    Returns:
        dict[int, dict]: image_id → detect の戻り値に "time"（秒）を加えた辞書．
    """
    path = os.path.join(DATASET_DIR, f"detections_{detector_name.replace(' ', '_')}_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    detector = load_detector(detector_name, queries)
    timed_detect(detector, load_image(coco, image_ids[0]))  # 時間の計測を安定させるためのウォームアップ
    out = {}
    for image_id in tqdm(image_ids, desc=f"detect {detector_name} {tag}", leave=False):
        det, sec = timed_detect(detector, load_image(coco, image_id))
        out[image_id] = det | {"time": sec}
    os.makedirs(DATASET_DIR, exist_ok=True)
    torch.save(out, path)
    del detector
    torch.cuda.empty_cache()
    return out


def cached_crop_features(coco, embedder, detector_name, detections, tag):
    """
    検出したボックスの切り抜き画像の特徴をキャッシュから読み込み，無ければ計算して保存する．

    Args:
        coco (dict): load_coco の戻り値．
        embedder: 埋め込みモデル F．
        detector_name (str): 検出器の名前．
        detections (dict[int, dict]): cached_detections の戻り値．
        tag (str): 分割の名前．

    Returns:
        dict[int, dict]: image_id → {"feats": 特徴（K×D），"time": 特徴計算の時間（秒）}．
    """
    path = os.path.join(DATASET_DIR, f"cropfeat_{detector_name}_{embedder.name}_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    out = {}
    for image_id, det in tqdm(detections.items(), desc=f"crop feat {embedder.name} {detector_name} {tag}", leave=False):
        image = load_image(coco, image_id)
        start = time.perf_counter()
        feats = embedder.image_features([make_crop(image, b) for b in det["boxes"]])
        out[image_id] = {"feats": feats, "time": time.perf_counter() - start}
    torch.save(out, path)
    return out


# ---------------------------------------------------------------------------
# E1：VLM f の比較（論文4.2節）
# ---------------------------------------------------------------------------
def topk_precision_recall(scores, vocab, gt_words, ks=(1, 5)):
    """
    画像ごとの Top-k Precision / Recall を計算する．

    Args:
        scores (np.ndarray): スコア（N×V）．
        vocab (list[str]): 候補語彙（長さV）．
        gt_words (list[set[str]]): 画像ごとの正解の単語の集合（長さN）．
        ks (tuple[int]): k の値．

    Returns:
        pd.DataFrame: 列 P@k，R@k を持つ画像ごとの指標（N行）．
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    rows = []
    for i, gt in enumerate(gt_words):
        row = {}
        for k in ks:
            hits = len({vocab[j] for j in order[i, :k]} & gt)
            row[f"P@{k}"], row[f"R@{k}"] = hits / k, safe_div(hits, len(gt))
        rows.append(row)
    return pd.DataFrame(rows)


def run_e1(coco, splits):
    """
    E1：画像全体に対する F の比較（タスクA，タスクB）と Wilcoxon 符号付き順位検定．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): split_image_ids の戻り値．
    """
    ids = splits["test"]
    gt_a = [set(gt_instances(coco, i)[1]) for i in ids]
    cap = caption_words(coco, ids)
    gt_b = [set(cap[i]) for i in ids]
    tasks = {"taskA": (coco["class_names"], gt_a), "taskB": (sorted(set().union(*gt_b)), gt_b)}
    per_image = {}
    for name in E1_EMBEDDERS:
        todo = [t for t in tasks if not os.path.exists(os.path.join(result_dir("E1_vlm", name, t), "metrics.json"))]
        if todo:
            embedder = load_embedder(name, SEED)
            feats = embedder.image_features([load_image(coco, i) for i in tqdm(ids, desc=f"E1 load {name}", leave=False)])
        for task, (vocab, gt) in tasks.items():
            d = result_dir("E1_vlm", name, task)
            if task in todo:
                mode = "softmax" if name != "SigLIP-ViT-B16" else "sigmoid"
                df = topk_precision_recall(embedder.scores(feats, vocab, mode), vocab, gt)
                df.insert(0, "image_id", ids)
                os.makedirs(d, exist_ok=True)
                df.to_csv(os.path.join(d, "per_image.csv"), index=False)
                metrics = {"n_images": len(ids), "vocab_size": len(vocab), "score": mode}
                for col in ["P@1", "R@1", "P@5", "R@5"]:
                    v = df[col].dropna().to_numpy()
                    rng = np.random.default_rng(SEED)
                    boots = [rng.choice(v, len(v)).mean() for _ in range(N_BOOTSTRAP)]
                    metrics[col] = float(v.mean())
                    metrics[f"{col}_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                save_json(metrics, os.path.join(d, "metrics.json"))
                save_json({"embedder": name, "task": task, "image_ids": "test split", "seed": SEED}, os.path.join(d, "config.json"))
                print("E1", name, task, {k: round(v, 3) for k, v in metrics.items() if k in ("P@1", "R@1", "P@5", "R@5")})
            per_image[(name, task)] = pd.read_csv(os.path.join(d, "per_image.csv"))
        if todo:
            del embedder
            torch.cuda.empty_cache()
    rows = []
    for task, (a, b), col in itertools.product(tasks, itertools.combinations(E1_EMBEDDERS, 2), ["P@1", "R@1", "P@5", "R@5"]):
        x, y = per_image[(a, task)][col].to_numpy(), per_image[(b, task)][col].to_numpy()
        ok = ~(np.isnan(x) | np.isnan(y))
        p = float(wilcoxon(x[ok], y[ok]).pvalue) if np.any(x[ok] != y[ok]) else 1.0
        rows.append({"task": task, "metric": col, "method_a": a, "method_b": b,
                     "mean_a": x[ok].mean(), "mean_b": y[ok].mean(), "p_wilcoxon": p})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "E1_vlm", "wilcoxon.csv"), index=False)


# ---------------------------------------------------------------------------
# E2：物体領域検出モデルの比較（論文4.3節）
# ---------------------------------------------------------------------------
def localization_metrics(coco, detections, image_ids):
    """
    クラス非依存の位置検出精度を求める．

    Args:
        coco (dict): load_coco の戻り値．
        detections (dict[int, dict]): cached_detections の戻り値．
        image_ids (list[int]): 評価する画像ID．

    Returns:
        dict: n_images，n_gt，mean_proposals，recall_iou50，mean_best_iou，mean_time_sec．
    """
    best, n_props, times = [], [], []
    for i in image_ids:
        gt_boxes, _ = gt_instances(coco, i)
        iou = box_iou(detections[i]["boxes"], gt_boxes)
        best.extend(iou.max(axis=0) if len(iou) else np.zeros(len(gt_boxes)))
        n_props.append(len(detections[i]["boxes"]))
        times.append(detections[i]["time"])
    best = np.array(best)
    return {"n_images": len(image_ids), "n_gt": len(best), "mean_proposals": float(np.mean(n_props)),
            "recall_iou50": float((best >= IOU_THRESHOLD).mean()), "mean_best_iou": float(best.mean()),
            "mean_time_sec": float(np.mean(times))}


def small_subset(coco, image_ids, n=4, max_gt=3):
    """
    論文4.3節の少数画像の評価に用いる，GTが max_gt 件以下の先頭 n 枚を選ぶ．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 候補の画像ID（テスト用）．
        n (int): 枚数．
        max_gt (int): GTの最大件数．

    Returns:
        list[int]: 選んだ画像ID．
    """
    return [i for i in image_ids if 1 <= len(gt_instances(coco, i)[1]) <= max_gt][:n]


def run_e2(coco, splits):
    """
    E2：DETR，YOLO11x，OWL-ViT，Grounding DINO のクラス非依存の位置検出精度を test 500枚と少数の4枚で比較する．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): split_image_ids の戻り値．
    """
    subsets = {"test500": splits["test"], "test4": small_subset(coco, splits["test"])}
    for name in E2_DETECTORS:
        dets = cached_detections(coco, name, splits["test"], "test", queries=coco["class_names"])
        for cond, ids in subsets.items():
            d = result_dir("E2_detector", name, cond)
            if os.path.exists(os.path.join(d, "metrics.json")):
                continue
            metrics = localization_metrics(coco, dets, ids)
            save_json(metrics, os.path.join(d, "metrics.json"))
            save_json({"detector": name, "image_ids": ids, "queries": "COCO 80 classes" if name in ("OWL-ViT", "Grounding DINO") else None},
                      os.path.join(d, "config.json"))
            print("E2", name, cond, {k: round(v, 3) for k, v in metrics.items()})


# ---------------------------------------------------------------------------
# E3：定量評価（論文4.4節，進捗報告）
# ---------------------------------------------------------------------------
def build_records(coco, detections, image_ids, labels_fn):
    """
    評価用に，画像ごとのGTと予測（ボックス，スコア，上位5語）をまとめる．

    Args:
        coco (dict): load_coco の戻り値．
        detections (dict[int, dict]): cached_detections の戻り値．
        image_ids (list[int]): 画像ID．
        labels_fn (callable): image_id → (予測の上位5語のリスト（各要素は長さ5以下のリスト），予測スコア（K）)．

    Returns:
        list[dict]: 画像ごとの "image_id"，"gt_boxes"，"gt_labels"，"boxes"，"top"，"scores"．
    """
    records = []
    for i in image_ids:
        gt_boxes, gt_labels = gt_instances(coco, i)
        top, scores = labels_fn(i)
        records.append({"image_id": i, "gt_boxes": gt_boxes, "gt_labels": gt_labels,
                        "boxes": detections[i]["boxes"], "top": top, "scores": np.asarray(scores, dtype=np.float32),
                        "det_scores": detections[i]["scores"]})
    return records


def metrics_func(records, theta=-np.inf):
    """
    予測と正解の一致度を評価する（進捗報告の「ラベルの一致率やバウンディングボックスのズレの大きさ」）．

    スコアが θ 未満の予測は棄却する（進捗報告の Alg. の break）．
    - ラベルの一致率：クラスを問わず IoU 0.5 以上で対応したボックスのうち，Top-1（Top-5）のクラスがGTと一致した割合
    - ボックスのずれ：対応したボックスの IoU，中心のずれ（GTの対角線長で正規化），座標の平均絶対誤差（ピクセル）
    - クラスを含めた適合率・再現率・F値（IoU 0.5 以上かつクラスが一致した場合を正解とする）

    Args:
        records (list[dict]): build_records の戻り値．
        theta (float): 類似と判断する下限値 θ．

    Returns:
        tuple[dict, np.ndarray]: 指標の辞書と，ブートストラップ用の画像ごとの計数
                                 （N×6：対応数，Top-1正解数，Top-5正解数，TP，予測数，GT数）．
    """
    counts, ious, centers, maes = [], [], [], []
    for r in records:
        keep = np.where(r["scores"] >= theta)[0]
        boxes, top = r["boxes"][keep], [r["top"][k] for k in keep]
        order = list(np.argsort(-r["det_scores"][keep], kind="stable"))
        iou = box_iou(boxes, r["gt_boxes"])
        pairs = greedy_match(iou, order)
        c1 = sum(top[p][0] == r["gt_labels"][g] for p, g in pairs.items())
        c5 = sum(r["gt_labels"][g] in top[p][:5] for p, g in pairs.items())
        same = np.zeros((len(top), len(r["gt_labels"])), dtype=bool)
        for p, t in enumerate(top):
            same[p] = [t[0] == gl for gl in r["gt_labels"]]
        tp = len(greedy_match(iou, order, allowed=same))
        counts.append([len(pairs), c1, c5, tp, len(keep), len(r["gt_labels"])])
        for p, g in pairs.items():
            b, gb = boxes[p], r["gt_boxes"][g]
            ious.append(iou[p, g])
            diag = np.hypot(gb[2] - gb[0], gb[3] - gb[1])
            centers.append(np.hypot((b[0] + b[2] - gb[0] - gb[2]) / 2, (b[1] + b[3] - gb[1] - gb[3]) / 2) / max(diag, 1e-6))
            maes.append(np.abs(b - gb).mean())
    counts = np.array(counts, dtype=np.float64)
    s = counts.sum(axis=0)
    precision, recall = safe_div(s[3], s[4]), safe_div(s[3], s[5])
    f1 = safe_div(2 * precision * recall, precision + recall) if s[3] else 0.0
    return {"n_images": len(records), "n_gt": int(s[5]), "n_pred": int(s[4]), "n_matched": int(s[0]),
            "top1_acc": safe_div(s[1], s[0]), "top5_acc": safe_div(s[2], s[0]),
            "precision": precision, "recall": recall, "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
            "mean_center_offset": float(np.mean(centers)) if centers else float("nan"),
            "mean_coord_mae_px": float(np.mean(maes)) if maes else float("nan")}, counts


def metrics_with_ci(records, theta=-np.inf):
    """
    metrics_func の指標に，Top-1正解率とF値の画像単位のブートストラップ95%信頼区間を加える．

    Args:
        records (list[dict]): build_records の戻り値．
        theta (float): θ．

    Returns:
        dict: 指標の辞書．
    """
    metrics, counts = metrics_func(records, theta)
    metrics["top1_acc_ci95"] = bootstrap_ci(counts[:, [1, 0]], lambda s: safe_div(s[0], s[1]))

    def f1_from(s):
        """合計した [TP, 予測数, GT数] からF値を求める．"""
        p, r = safe_div(s[0], s[1]), safe_div(s[0], s[2])
        return safe_div(2 * p * r, p + r) if s[0] else 0.0
    metrics["f1_ci95"] = bootstrap_ci(counts[:, [3, 4, 5]], f1_from)
    return metrics


def proposed_labels_fn(embedder, feats, class_names, mode):
    """
    提案手法（初期の語彙をCOCO 80クラス，下位語なし）の予測を返す関数を作る．

    Args:
        embedder: 埋め込みモデル F．
        feats (dict[int, dict]): cached_crop_features の戻り値．
        class_names (list[str]): COCOの80クラス名．
        mode (str): スコアの種類．

    Returns:
        callable: image_id → (上位5語のリスト，最大スコア（K）)．
    """
    def fn(image_id):
        """画像1枚の予測を返す．"""
        f = feats[image_id]["feats"]
        if len(f) == 0:
            return [], np.zeros(0)
        s = embedder.scores(f, class_names, mode)
        order = np.argsort(-s, axis=1)[:, :5]
        return [[class_names[j] for j in row] for row in order], s.max(axis=1)
    return fn


def closed_labels_fn(detections):
    """
    Close Vocabularyモデル（検出器自身のクラス出力）の予測を返す関数を作る．

    Args:
        detections (dict[int, dict]): cached_detections の戻り値．

    Returns:
        callable: image_id → (クラスを1語だけ含むリストのリスト，検出スコア（K）)．
    """
    return lambda i: ([[label] for label in detections[i]["labels"]], detections[i]["scores"])


def select_theta(records, out_dir):
    """
    dev の F値が最大となる θ を選ぶ（候補は −∞ と dev の最大スコアの分位点）．theta_curve.csv を保存する．

    Args:
        records (list[dict]): dev の build_records の戻り値．
        out_dir (str): theta_curve.csv の保存先．

    Returns:
        float: 選んだ θ．
    """
    all_scores = np.concatenate([r["scores"] for r in records])
    candidates = [-np.inf] + [float(np.quantile(all_scores, q)) for q in THETA_QUANTILES]
    curve = pd.DataFrame([{"theta": t} | metrics_func(records, t)[0] for t in candidates])
    os.makedirs(out_dir, exist_ok=True)
    curve.to_csv(os.path.join(out_dir, "theta_curve.csv"), index=False)
    return float(curve.loc[curve["f1"].idxmax(), "theta"])


def save_e3(method, split, records, theta, config):
    """
    E3 の1条件の結果（metrics.json，config.json，test は predictions.json）を保存する．

    Args:
        method (str): 手法名．
        split (str): "dev" または "test"．
        records (list[dict]): build_records の戻り値．
        theta (float): θ．
        config (dict): 条件のメタデータ．

    Returns:
        dict: 指標の辞書．
    """
    d = result_dir("E3_quantitative", method, split)
    metrics = metrics_with_ci(records, theta)
    save_json(metrics, os.path.join(d, "metrics.json"))
    save_json(config | {"theta": theta, "split": split}, os.path.join(d, "config.json"))
    if split == "test":
        preds = [{"image_id": r["image_id"], "boxes": r["boxes"].round(1).tolist(), "top5": r["top"],
                  "score": r["scores"].round(4).tolist()} for r in records]
        save_json(preds, os.path.join(d, "predictions.json"))
    print("E3", method, split, f"θ={theta:.4f}", {k: round(metrics[k], 3) for k in ("top1_acc", "top5_acc", "precision", "recall", "f1", "mean_iou")})
    return metrics


def run_e3(coco, splits):
    """
    E3：初期の語彙をCOCO 80クラスとした提案手法（下位語なし）と，Close Vocabularyモデルの比較．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): split_image_ids の戻り値．

    Returns:
        dict[tuple[str, str, str], float]: (D, F, mode) → 選んだ θ（E4〜E6で用いる）．
    """
    thetas, rows = {}, []
    dets = {(d, s): cached_detections(coco, d, splits[s], s) for d, s in itertools.product(E3_DETECTORS, SPLITS)}
    for det_name, split in itertools.product(E3_DETECTORS, SPLITS):
        d = result_dir("E3_quantitative", f"closed_{det_name}", split)
        if not os.path.exists(os.path.join(d, "metrics.json")):
            records = build_records(coco, dets[(det_name, split)], splits[split], closed_labels_fn(dets[(det_name, split)]))
            save_e3(f"closed_{det_name}", split, records, -np.inf, {"method": "closed", "detector": det_name})
    for emb_name in E3_EMBEDDERS:
        embedder = load_embedder(emb_name, SEED)
        for det_name, mode in itertools.product(E3_DETECTORS, E3_MODES):
            feats = {s: cached_crop_features(coco, embedder, det_name, dets[(det_name, s)], s) for s in SPLITS}
            records = {s: build_records(coco, dets[(det_name, s)], splits[s],
                                        proposed_labels_fn(embedder, feats[s], coco["class_names"], mode)) for s in SPLITS}
            base = f"proposed_{det_name}_{emb_name}_{mode}"
            theta_path = os.path.join(result_dir("E3_quantitative", base + "_theta", "dev"), "config.json")
            if os.path.exists(theta_path):
                theta = json.load(open(theta_path))["theta"]
            else:
                theta = select_theta(records["dev"], os.path.dirname(theta_path))
            thetas[(det_name, emb_name, mode)] = theta
            for split, (suffix, t) in itertools.product(SPLITS, [("_all", -np.inf), ("_theta", theta)]):
                if os.path.exists(os.path.join(result_dir("E3_quantitative", base + suffix, split), "metrics.json")):
                    continue
                save_e3(base + suffix, split, records[split], t,
                        {"method": "proposed", "detector": det_name, "embedder": emb_name, "mode": mode,
                         "vocabulary": "coco80", "hyponyms": False,
                         "theta_selection": "dev F1" if suffix == "_theta" else "none (θ = −∞)"})
            rows.append({"detector": det_name, "embedder": emb_name, "mode": mode, "theta": theta,
                         "detect_sec_per_image": np.mean([v["time"] for v in dets[(det_name, "test")].values()]),
                         "annotate_sec_per_image": np.mean([v["time"] for v in feats["test"].values()])})
        del embedder
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "E3_quantitative", "thetas_and_timing.csv"), index=False)
    return thetas


# ---------------------------------------------------------------------------
# E4：WordNet の深さごとのスコア（論文4.5節）
# ---------------------------------------------------------------------------
def run_e4(coco, splits, thetas):
    """
    E4：test で深さ0の推定がGTと一致したインスタンス（DETRのボックス）について，下位語をたどり深さごとに集計する．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): split_image_ids の戻り値．
        thetas (dict): run_e3 の戻り値．
    """
    det_name, ids = "DETR", splits["test"]
    dets = cached_detections(coco, det_name, ids, "test")
    vocab = root_vocab("coco80")
    for emb_name in E3_EMBEDDERS:
        conds = [(m, t) for m, t in [("softmax", -np.inf), ("cosine", thetas[(det_name, emb_name, "cosine")])]
                 if not os.path.exists(os.path.join(result_dir("E4_depth", f"{det_name}_{emb_name}_{m}", "test"), "metrics.json"))]
        if not conds:
            continue
        embedder = load_embedder(emb_name, SEED)
        feats = cached_crop_features(coco, embedder, det_name, dets, "test")
        for mode, theta in conds:
            chains = []
            for i in tqdm(ids, desc=f"E4 {emb_name} {mode}", leave=False):
                gt_boxes, gt_labels = gt_instances(coco, i)
                iou = box_iou(dets[i]["boxes"], gt_boxes)
                for p, g in greedy_match(iou, list(range(len(dets[i]["boxes"])))).items():
                    chain = granular_annotation(embedder, feats[i]["feats"][p:p + 1], vocab, theta, mode, E4_MAX_DEPTH)
                    if chain and chain[0]["word"] == gt_labels[g]:
                        chains.append({"image_id": i, "box": dets[i]["boxes"][p].tolist(), "gt": gt_labels[g], "chain": chain})
            steps = pd.DataFrame([s | {"instance": k} for k, c in enumerate(chains) for s in c["chain"]])
            table = steps.groupby("depth").agg(n_instances=("instance", "count"), mean_candidates=("n_candidates", "mean"),
                                               mean_score=("score", "mean")).reset_index()
            d = result_dir("E4_depth", f"{det_name}_{emb_name}_{mode}", "test")
            os.makedirs(d, exist_ok=True)
            table.to_csv(os.path.join(d, "depth_table.csv"), index=False)
            save_json({"n_correct_depth0": len(chains), "theta": theta, "depth_table": table.to_dict(orient="records"),
                       "final_depth_hist": pd.Series([len(c["chain"]) - 1 for c in chains]).value_counts().sort_index().to_dict()},
                      os.path.join(d, "metrics.json"))
            save_json(chains, os.path.join(d, "chains.json"))
            save_json({"detector": det_name, "embedder": emb_name, "mode": mode, "theta": theta, "max_depth": E4_MAX_DEPTH},
                      os.path.join(d, "config.json"))
            print("E4", emb_name, mode, table.round(3).to_dict(orient="records"))
        del embedder
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# E5，E6：COCO外の画像と定性評価（論文4.6節，4.7節，進捗報告の定性評価）
# ---------------------------------------------------------------------------
def chain_text(chain, width=34):
    """
    1つのボックスの出力の列を図のラベルの文字列にする．

    Args:
        chain (list[dict]): granular_annotation の出力．
        width (int): 折り返す文字数．

    Returns:
        str: 例 "dog (0.91) → hound (0.40)"．出力が無ければ "(rejected)"．
    """
    if not chain:
        return "(rejected)"
    return "\n".join(textwrap.wrap(" → ".join(f"{s['word']} ({s['score']:.2f})" for s in chain), width))


def draw_grid(images, annotations, titles, path, max_boxes=6):
    """
    画像ごとにボックスと粗い語から細かい語までの出力を描いた図を保存する．

    Args:
        images (list[PIL.Image.Image]): 画像のリスト．
        annotations (list[list[dict]]): 画像ごとの annotate の結果（"box"，"chain"）．
        titles (list[str]): 画像ごとのタイトル．
        path (str): 保存先．
        max_boxes (int): 1枚に描く最大のボックス数（検出スコアの上位）．
    """
    n = len(images)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows), squeeze=False)
    colors = plt.cm.tab10.colors
    for ax in axes.ravel():
        ax.axis("off")
    for k, (im, ann, title) in enumerate(zip(images, annotations, titles)):
        ax = axes.ravel()[k]
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        for b_i, a in enumerate(ann[:max_boxes]):
            x0, y0, x1, y1 = a["box"]
            c = colors[b_i % 10]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=c, lw=2))
            ax.text(x0, y0, chain_text(a["chain"]), fontsize=6.5, color="white", va="bottom",
                    bbox={"facecolor": c, "alpha": 0.8, "pad": 1, "lw": 0})
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def annotate_cached(embedder, image, det, vocab, theta, mode):
    """
    検出済みのボックス（D の出力）に対して Alg. の While ループを適用する（検出は再実行しない）．

    Args:
        embedder: 埋め込みモデル F．
        image (PIL.Image.Image): 画像．
        det (dict): detect の戻り値．
        vocab (list[tuple[str, str]]): 初期の語彙．
        theta (float): θ．
        mode (str): スコアの種類．

    Returns:
        list[dict]: "box"，"det_score"，"chain" のリスト（検出スコアの降順）．
    """
    if len(det["boxes"]) == 0:
        return []
    feats = embedder.image_features([make_crop(image, b) for b in det["boxes"]])
    return [{"box": [float(v) for v in b], "det_score": float(s),
             "chain": granular_annotation(embedder, feats[k:k + 1], vocab, theta, mode)}
            for k, (b, s) in enumerate(zip(det["boxes"], det["scores"]))]


def run_qualitative(experiment, condition, images, dets, vocab_name, thetas, titles):
    """
    E5，E6 の共通処理：DETR の検出結果に，(F, mode) ごとに粒度可変アノテーションを行い，結果と図を保存する．

    Args:
        experiment (str): "E5_open_vocab" または "E6_qualitative"．
        condition (str): 条件名（語彙の名前など）．
        images (list[PIL.Image.Image]): 画像．
        dets (list[dict]): 画像ごとの検出結果．
        vocab_name (str): root_vocab の名前．
        thetas (dict): run_e3 の戻り値．
        titles (list[str]): 画像ごとのタイトル．
    """
    vocab = root_vocab(vocab_name)
    for emb_name in E3_EMBEDDERS:
        conds = [(m, t) for m, t in [("softmax", -np.inf), ("cosine", thetas[("DETR", emb_name, "cosine")])]
                 if not os.path.exists(os.path.join(result_dir(experiment, f"DETR_{emb_name}_{m}", condition), "annotations.json"))]
        if not conds:
            continue
        embedder = load_embedder(emb_name, SEED)
        for mode, theta in conds:
            anns = [annotate_cached(embedder, im, det, vocab, theta, mode) for im, det in zip(images, dets)]
            d = result_dir(experiment, f"DETR_{emb_name}_{mode}", condition)
            save_json({"theta": theta, "mode": mode, "vocabulary": vocab_name, "n_root_words": len(vocab),
                       "results": [{"title": t, "annotations": a} for t, a in zip(titles, anns)]},
                      os.path.join(d, "annotations.json"))
            save_json({"detector": "DETR", "embedder": emb_name, "mode": mode, "theta": theta, "vocabulary": vocab_name},
                      os.path.join(d, "config.json"))
            th = "−∞" if np.isinf(theta) else f"{theta:.3f}"
            draw_grid(images, anns, [f"{t}  [DETR + {emb_name}, {mode}, θ={th}]" for t in titles], os.path.join(d, "grid.png"))
        del embedder
        torch.cuda.empty_cache()


def run_e5(thetas):
    """
    E5：COCO外の語を主要な被写体とする画像に，COCO 80 + 22語の語彙でアノテーションする（論文4.6節）．

    Args:
        thetas (dict): run_e3 の戻り値．
    """
    images = prepare_open_vocab_images()
    detector = load_detector("DETR", seed=SEED)
    dets = [detector.detect(im) for im in images.values()]
    del detector
    run_qualitative("E5_open_vocab", "coco80_extra", list(images.values()), dets, "coco80_extra", thetas, list(images))


def run_e6(coco, splits, thetas):
    """
    E6：COCOの画像に，初期の語彙をCOCO 80クラスまたはWordNetの上位語として階層的にアノテーションする．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): split_image_ids の戻り値．
        thetas (dict): run_e3 の戻り値．
    """
    dets_all = cached_detections(coco, "DETR", splits["test"], "test")
    ids = [i for i in splits["test"] if 2 <= len(dets_all[i]["boxes"]) <= 4][:N_QUALITATIVE]
    images = [load_image(coco, i) for i in ids]
    for vocab_name in ["coco80", "wordnet_tops"]:
        run_qualitative("E6_qualitative", vocab_name, images, [dets_all[i] for i in ids], vocab_name, thetas,
                        [f"COCO {i}" for i in ids])


def train():
    """
    E1〜E6 をすべて実行する（終了済みの条件はスキップする）．
    """
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coco = load_coco()
    splits = split_image_ids(coco)
    save_json(splits, os.path.join(DATASET_DIR, "splits.json"))
    run_e1(coco, splits)
    run_e2(coco, splits)
    thetas = run_e3(coco, splits)
    run_e4(coco, splits, thetas)
    run_e5(thetas)
    run_e6(coco, splits, thetas)


if __name__ == "__main__":
    train()
