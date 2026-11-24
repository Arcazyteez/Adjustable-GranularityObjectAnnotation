"""
ex010の実行スクリプト：COCO外の物体の検出から粒度変更まで（order_011の要求2，3）．

学習は行わない．lvis_test（2148枚）で以下を評価する．
1. 領域の候補の再現率: 4つの検出器（model.DETECTORS）について，1画像あたり上位 K 件のボックスで，COCOの物体と
   COCO外の物体（LVIS）がIoU 0.5以上で検出される割合（table_recall.csv）
2. 検出から命名・粒度変更まで（table_end2end.csv）:
   - yolo:            YOLO11x（閾値0.25）の語のみ（比較対象）
   - hybrid:          YOLO11xの語を深さ0とし，SigLIPでWordNetの下位語をたどる
   - proposed:        DETR（閾値0.5）のボックスをSigLIPで命名（語彙 V）し，WordNetの下位語をたどる
   - hybrid+{検出器}: hybrid のボックスに，YOLO11xのいずれのボックスともIoUが0.5未満の候補を上位 K 件加え，
                      加えたボックスをSigLIPで命名（語彙 V）し，WordNetの下位語をたどる
   出力ボックスとGT（COCOの物体とCOCO外の物体）をIoU 0.5以上で1対1に対応付け，COCO外の物体について
   検出率，深さ0の語がLVISのSynsetの上位語または自身である割合（命名率），下位語の展開がLVISのSynsetに到達した割合（到達率）を求める．
3. ex004の実画像の定性評価

出力: outputs/ex010_lvis_open_world_detection/{手法}/{語彙}/lvis_test_K{K}/{seed}/metrics.json，records.json
    および直下の table_recall.csv，table_end2end.csv，qualitative/
"""
import itertools
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image, make_crop  # noqa: E402
from common.metrics import box_iou, match_boxes  # noqa: E402
from common.paths import DATASET_DIR, OUTPUT_DIR  # noqa: E402
from common.vlm import vlm_logits  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET  # noqa: E402
from data import (EVAL_SPLIT, cached_detections, ex008_data, ex009_data, ground_truth, import_experiment_module,  # noqa: E402
                  lvis_vocab, wordnet_vocab)
from model import DETECTORS, ex009_model, load_detector  # noqa: E402

# ===== 実験条件（order_011） =====
EXPERIMENT = "ex010_lvis_open_world_detection"
SEEDS = [42]
CROP_MODE = "pad"
IOU_THRESHOLD = 0.5
RECALL_KS = [1, 5, 10, 20, 50, 100]
SUPPLEMENT_KS = [5, 10, 20]       # YOLO11xに加える候補の数（1画像あたり）
MAIN_K = 10
VOCABS = {"wn": wordnet_vocab, "lvis": lvis_vocab}
NORMALIZATION = "sigmoid"
QUALITATIVE_K = 5


def methods():
    """
    評価する（手法，加える候補の検出器，K，語彙）の組を返す．

    Returns:
        list[tuple[str, str | None, int, str]]: 条件のリスト（VLMで命名しない手法の語彙は "none"）．
    """
    out = [("yolo", None, 0, "none"), ("hybrid", None, 0, "none")]
    out += [("proposed", None, 0, v) for v in VOCABS]
    out += [(f"hybrid+{d}", d, k, v) for d, k, v in itertools.product(DETECTORS, SUPPLEMENT_KS, VOCABS)]
    return out


def target_dir(method, vocab, k, seed):
    """
    1条件の保存先を返す．

    Args:
        method (str): 手法名．
        vocab (str): 語彙名．
        k (int): 加える候補の数．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, vocab, f"{EVAL_SPLIT}_K{k}", str(seed))


# ===================== 領域の候補の再現率 =====================

def recall_at_k(gts_per_image, boxes_per_image, ks):
    """
    1画像あたり上位 K 件のボックスで，GTがIoU 0.5以上で検出される割合を計算する（1対1の対応付けは行わない）．

    Args:
        gts_per_image (dict[str, list[dict]]): 画像ID → GT（ground_truth の戻り値）．
        boxes_per_image (dict[str, list]): 画像ID → スコアの降順のボックス（K×4）．
        ks (list[int]): K のリスト．

    Returns:
        dict: "{kind}_recall@{K}" と "boxes@{K}"（1画像あたりの平均ボックス数）．
    """
    hits = {(kind, k): [] for kind in ["coco", "noncoco"] for k in ks}
    n_boxes = {k: [] for k in ks}
    for image_id, gts in gts_per_image.items():
        boxes = np.asarray(boxes_per_image.get(image_id, []), dtype=np.float32).reshape(-1, 4)
        gt_boxes = np.asarray([g["box"] for g in gts], dtype=np.float32).reshape(-1, 4)
        iou = box_iou(gt_boxes, boxes) if len(boxes) and len(gt_boxes) else np.zeros((len(gt_boxes), len(boxes)))
        for k in ks:
            n_boxes[k].append(min(k, len(boxes)))
            best = iou[:, :k].max(axis=1) if iou.shape[1] else np.zeros(len(gts))
            for g, b in zip(gts, best):
                hits[(g["kind"], k)].append(b >= IOU_THRESHOLD)
    out = {f"{kind}_recall@{k}": float(np.mean(v)) for (kind, k), v in hits.items()}
    return out | {f"boxes@{k}": float(np.mean(v)) for k, v in n_boxes.items()}


def recall_table(gts_per_image, proposals, base):
    """
    4つの検出器の再現率と，YOLO11x（閾値0.25）・DETR（閾値0.5）・YOLO11xに候補を加えた場合の再現率をまとめる．

    Args:
        gts_per_image (dict): 画像ID → GT．
        proposals (dict[str, dict]): 検出器名 → cached_detections の戻り値．
        base (dict[str, dict]): "yolo"，"detr" → 画像ID → 出力ボックス（output_boxes の戻り値）．

    Returns:
        pd.DataFrame: 行が検出器，列が再現率と平均ボックス数．
    """
    rows = [{"detector": name} | recall_at_k(gts_per_image, {i: d["boxes"] for i, d in det.items()}, RECALL_KS)
            for name, det in proposals.items()]
    for name, boxes in base.items():
        row = recall_at_k(gts_per_image, {i: [b["box"] for b in v] for i, v in boxes.items()}, [10 ** 6])
        rows.append({"detector": name} | {k.replace("@1000000", "@all"): v for k, v in row.items()})
    return pd.DataFrame(rows)


# ===================== 出力ボックスと対応付け =====================

def output_boxes(method, detector, k, image_id, ctx, proposals):
    """
    手法が1枚の画像に出力するボックスを返す．

    Args:
        method (str): 手法名．
        detector (str | None): 加える候補の検出器．
        k (int): 加える候補の数．
        image_id (str): 画像ID．
        ctx (dict): ex009の load_context の戻り値．
        proposals (dict[str, dict]): 検出器名 → cached_detections の戻り値．

    Returns:
        list[dict]: {"box"，"score"，"source"（"yolo"，"detr" または検出器名），("yolo_word")}．
    """
    if method == "proposed":
        det = ctx["detections"]["detr"][image_id]
        return [{"box": b, "score": s, "source": "detr"} for b, s in zip(det["boxes"], det["scores"]) if s > 0.5]
    det = ctx["detections"]["yolo"][image_id]
    out = [{"box": b, "score": s, "source": "yolo", "yolo_word": t[0]}
           for b, s, t in zip(det["boxes"], det["scores"], det["top5"]) if s > 0.25]
    if detector is None:
        return out
    cand = proposals[detector][image_id]
    base = np.asarray([b["box"] for b in out], dtype=np.float32).reshape(-1, 4)
    boxes = np.asarray(cand["boxes"], dtype=np.float32).reshape(-1, 4)
    overlap = box_iou(boxes, base).max(axis=1) if len(base) and len(boxes) else np.zeros(len(boxes))
    extra = [i for i in range(len(boxes)) if overlap[i] < IOU_THRESHOLD][:k]
    return out + [{"box": cand["boxes"][i], "score": cand["scores"][i], "source": detector} for i in extra]


def matched_records(method, detector, k, ctx, proposals, gts_per_image):
    """
    出力ボックスとGTを1対1に対応付け，対応したボックスの記録と出力ボックスの統計を返す．

    Args:
        method (str): 手法名．
        detector (str | None): 加える候補の検出器．
        k (int): 加える候補の数．
        ctx (dict): ex009の load_context の戻り値．
        proposals (dict): 検出器名 → cached_detections の戻り値．
        gts_per_image (dict): 画像ID → GT．

    Returns:
        tuple[list[dict], dict]: 記録 {"image_id"，"box"，"source"，"gt"，("yolo_word")} と，
            n_boxes（全出力ボックス数），n_extra（加えたボックス数），n_extra_matched（加えたボックスのうちGTと対応した数）．
    """
    records, stats = [], {"n_boxes": 0, "n_extra": 0, "n_extra_matched": 0}
    for image_id, gts in gts_per_image.items():
        out = output_boxes(method, detector, k, image_id, ctx, proposals)
        stats["n_boxes"] += len(out)
        stats["n_extra"] += sum(b["source"] == detector for b in out)
        if not out or not gts:
            continue
        pairs = match_boxes(np.asarray([b["box"] for b in out], dtype=np.float32),
                            np.asarray([g["box"] for g in gts], dtype=np.float32), IOU_THRESHOLD)
        for p, g, _ in pairs:
            stats["n_extra_matched"] += out[p]["source"] == detector
            records.append({"image_id": int(image_id), "gt": gts[g]} | out[p])
    return records, stats


# ===================== 命名と粒度変更 =====================

class Annotator:
    """SigLIPによる深さ0の命名とWordNetの下位語の展開を行い，クロップの画像特徴と展開結果をキャッシュする．"""

    def __init__(self, vlm, coco):
        """
        Args:
            vlm (SigLIPModel): VLM．
            coco (COCO): instancesのCOCOオブジェクト．
        """
        self.vlm, self.coco, self.m9 = vlm, coco, ex009_model()
        self.features, self.chains = {}, {}

    @staticmethod
    def key(record):
        """
        クロップを識別するキーを返す．

        Args:
            record (dict): "image_id"，"box" を持つ記録．

        Returns:
            tuple: (画像ID, 丸めたボックス)．
        """
        return record["image_id"], tuple(round(float(v), 2) for v in record["box"])

    def prepare(self, records, batch_size=256):
        """
        未計算のクロップの画像特徴をまとめて計算する．

        Args:
            records (list[dict]): 記録．
            batch_size (int): 一度にVLMへ入力するクロップ数．

        Returns:
            None
        """
        todo = sorted({self.key(r): r for r in records if self.key(r) not in self.features}.items(),
                      key=lambda kv: kv[0][0])
        for start in tqdm(range(0, len(todo), batch_size), desc="SigLIP features", leave=False):
            chunk = todo[start:start + batch_size]
            crops = [make_crop(load_image(self.coco, k[0]), r["box"], CROP_MODE) for k, r in chunk]
            for (k, _), feat in zip(chunk, self.vlm.image_features(crops)):
                self.features[k] = feat

    def name(self, records, vocab):
        """
        VLMで命名する記録について，語彙 vocab から argmax の語を深さ0とする．

        Args:
            records (list[dict]): VLMで命名する記録（長さN）．
            vocab (list[dict]): 語彙．

        Returns:
            list[tuple[str, str, float]]: (語，Synset名，シグモイドの確信度) のリスト．
        """
        if not records:
            return []
        feats = torch.stack([self.features[self.key(r)] for r in records])
        logits = vlm_logits(self.vlm, feats, [v["word"] for v in vocab])
        best = logits.argmax(dim=-1).cpu().numpy()
        conf = logits.sigmoid().gather(1, torch.as_tensor(best, device=logits.device)[:, None])[:, 0].cpu().numpy()
        return [(vocab[b]["word"], vocab[b]["synset"], float(c)) for b, c in zip(best, conf)]

    def chain(self, record, word, synset):
        """
        深さ0の語から下位語を最後まで展開する（キャッシュする）．

        Args:
            record (dict): 記録．
            word (str): 深さ0の語．
            synset (str): 深さ0のSynset名．

        Returns:
            list[dict]: 各深さの {"depth"，"word"，"synset"，"conf"}（シグモイドの確信度）．
        """
        k = (self.key(record), synset)
        if k not in self.chains:
            self.chains[k] = [{"depth": s["depth"], "word": s["word"], "synset": s["synset"],
                               "conf": s[f"conf_{NORMALIZATION}"]}
                              for s in self.m9.hierarchical_chain(self.vlm, self.features[self.key(record)], word, synset)]
        return self.chains[k]


def annotate(annotator, method, records, vocab_name):
    """
    対応したボックスの深さ0を決め，yolo 以外では下位語を最後まで展開する．

    Args:
        annotator (Annotator): 命名と展開を行うオブジェクト．
        method (str): 手法名．
        records (list[dict]): matched_records の記録．
        vocab_name (str): 語彙名（"none" はVLMで命名するボックスが無い手法）．

    Returns:
        list[dict]: 記録に "root_word"，"root_synset"，"chain" を追記したもの．
    """
    if method != "yolo":
        annotator.prepare(records)
    vlm_records = [r for r in records if r["source"] != "yolo"]
    names = dict(zip(map(id, vlm_records), annotator.name(vlm_records, VOCABS[vocab_name]() if vlm_records else [])))
    out = []
    for r in records:
        if r["source"] == "yolo":
            word, synset = r["yolo_word"], COCO_TO_SYNSET[r["yolo_word"]]
        else:
            word, synset, _ = names[id(r)]
        chain = [{"depth": 0, "word": word, "synset": synset}] if method == "yolo" else annotator.chain(r, word, synset)
        out.append(r | {"root_word": word, "root_synset": synset, "chain": chain})
    return out


def end2end_metrics(records, gts_per_image, stats):
    """
    COCO外の物体とCOCOの物体について，検出・命名・粒度変更の指標を計算する．

    Args:
        records (list[dict]): annotate の戻り値．
        gts_per_image (dict): 画像ID → GT．
        stats (dict): matched_records の出力ボックスの統計．

    Returns:
        dict: noncoco_n，noncoco_detected，noncoco_named（深さ0がLVISのSynsetの上位語または自身），noncoco_reach（展開がLVISの
            Synsetに到達），coco_n，coco_detected，coco_named（深さ0がGTのカテゴリ），boxes_per_image，extra_per_image，extra_matched_rate．
    """
    ancestors = ex008_data().ancestors
    n = {kind: sum(g["kind"] == kind for gts in gts_per_image.values() for g in gts) for kind in ["coco", "noncoco"]}
    non = [r for r in records if r["gt"]["kind"] == "noncoco"]
    coco = [r for r in records if r["gt"]["kind"] == "coco"]
    named = [r["root_synset"] == r["gt"]["synset"] or r["root_synset"] in ancestors(r["gt"]["synset"]) for r in non]
    reach = [r["gt"]["synset"] in [s["synset"] for s in r["chain"]] for r in non]
    n_images = len(gts_per_image)
    return {"noncoco_n": n["noncoco"], "noncoco_detected": len(non) / n["noncoco"],
            "noncoco_named": sum(named) / n["noncoco"], "noncoco_reach": sum(reach) / n["noncoco"],
            "noncoco_named_given_detected": float(np.mean(named)) if non else float("nan"),
            "coco_n": n["coco"], "coco_detected": len(coco) / n["coco"],
            "coco_named": sum(r["root_synset"] == COCO_TO_SYNSET[r["gt"]["name"]] for r in coco) / n["coco"],
            "boxes_per_image": stats["n_boxes"] / n_images, "extra_per_image": stats["n_extra"] / n_images,
            "extra_matched_rate": stats["n_extra_matched"] / stats["n_extra"] if stats["n_extra"] else float("nan")}


# ===================== 定性評価 =====================

def run_qualitative(annotator, seed):
    """
    ex004の実画像に hybrid+{検出器}（K=QUALITATIVE_K，語彙 wn）を適用し，結果を保存する（完了済みならスキップ）．

    Args:
        annotator (Annotator): 命名と展開を行うオブジェクト．
        seed (int): シード．

    Returns:
        None
    """
    out_dir = os.path.join(OUTPUT_DIR, EXPERIMENT, "qualitative")
    if os.path.exists(os.path.join(out_dir, "annotations.json")):
        return
    samples = import_experiment_module("ex006_coco_val500_yolo_clip_hybrid", "data").load_qualitative_samples()["open_vocab"]
    yolo = ex009_model().load_yolo(seed, 0.25)
    base = {s["name"]: yolo.detect(s["image"]) for s in samples}
    del yolo
    results = {}
    for detector in DETECTORS:
        model = load_detector(detector, seed)
        for s in samples:
            boxes, scores = model.detect(s["image"])
            regions = [{"box": b, "score": sc, "source": "yolo", "yolo_word": t[0]}
                       for b, sc, t in zip(base[s["name"]]["boxes"], base[s["name"]]["scores"], base[s["name"]]["top5"])]
            ref = np.asarray([r["box"] for r in regions], dtype=np.float32).reshape(-1, 4)
            overlap = box_iou(np.asarray(boxes).reshape(-1, 4), ref).max(axis=1) if len(ref) and len(boxes) else np.zeros(len(boxes))
            regions += [{"box": list(map(float, b)), "score": float(sc), "source": detector}
                        for b, sc, o in zip(boxes, scores, overlap) if o < IOU_THRESHOLD][:QUALITATIVE_K]
            feats = annotator.vlm.image_features([make_crop(s["image"], r["box"], CROP_MODE) for r in regions]) if regions else []
            vocab = wordnet_vocab()
            logits = vlm_logits(annotator.vlm, feats, [v["word"] for v in vocab]) if regions else []
            for r, feat, lg in zip(regions, feats, logits):
                if r["source"] == "yolo":
                    word, synset = r["yolo_word"], COCO_TO_SYNSET[r["yolo_word"]]
                else:
                    best = int(lg.argmax())
                    word, synset = vocab[best]["word"], vocab[best]["synset"]
                r["chain"] = [{"depth": c["depth"], "word": c["word"], "conf": c[f"conf_{NORMALIZATION}"]}
                              for c in annotator.m9.hierarchical_chain(annotator.vlm, feat, word, synset)]
                r["root_conf"] = float(lg.sigmoid()[int(lg.argmax())]) if r["source"] != "yolo" else float(r["score"])
            results[f"{detector}/{s['name']}"] = [r | {"box": list(map(float, r["box"])), "score": float(r["score"])}
                                                  for r in regions]
        del model
        torch.cuda.empty_cache()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump({"samples": {s["name"]: s["meta"] for s in samples}, "results": results}, f, indent=1)
    for detector in DETECTORS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, s in zip(axes.flat, samples):
            ax.imshow(s["image"])
            ax.axis("off")
            ax.set_title(f"YOLO + {detector}: {s['name']}", fontsize=9)
            for i, r in enumerate(results[f"{detector}/{s['name']}"]):
                x0, y0, x1, y1 = r["box"]
                color = "tab:blue" if r["source"] == "yolo" else "tab:red"
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
                head = "YOLO: " if r["source"] == "yolo" else "VLM: "
                text = head + " > ".join(c["word"] for c in r["chain"][:3])
                ax.text(x0, y0, text, fontsize=6, color="white", va="top", bbox=dict(facecolor=color, alpha=0.7, pad=1))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"grid_{detector}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


# ===================== 実行 =====================

def main():
    """再現率，検出から粒度変更までの評価，定性評価を実行し，集計表を作成する（完了済みの条件はスキップ）．"""
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    for seed in SEEDS:
        ctx = ex009_data().load_context(seed, None, None)
        image_ids = ctx["splits"][EVAL_SPLIT]
        proposals = {d: cached_detections(d, ctx["coco"], image_ids, lambda d=d: load_detector(d, seed)) for d in DETECTORS}
        gts_per_image = {str(i): ground_truth(ctx, i) for i in image_ids}
        os.makedirs(root, exist_ok=True)
        base = {f"{m}": {str(i): output_boxes(m, None, 0, str(i), ctx, proposals) for i in image_ids}
                for m in ["yolo", "proposed"]}
        recall_table(gts_per_image, proposals, {"yolo0.25": base["yolo"], "detr0.5": base["proposed"]}).to_csv(
            os.path.join(root, "table_recall.csv"), index=False)
        annotator = Annotator(ex009_model().load_model(seed), ctx["coco"])
        rows = []
        for method, detector, k, vocab in methods():
            out_dir = target_dir(method, vocab, k, seed)
            if not os.path.exists(os.path.join(out_dir, "metrics.json")):
                start = time.time()
                records, stats = matched_records(method, detector, k, ctx, proposals, gts_per_image)
                records = annotate(annotator, method, records, vocab)
                metrics = end2end_metrics(records, gts_per_image, stats)
                config = {"method": method, "detector": detector, "k": k, "vocab": vocab, "split": EVAL_SPLIT,
                          "n_images": len(image_ids), "vocab_size": len(VOCABS[vocab]()) if vocab != "none" else 0,
                          "normalization": NORMALIZATION, "iou_threshold": IOU_THRESHOLD, "seed": seed,
                          "elapsed_sec": time.time() - start}
                os.makedirs(out_dir, exist_ok=True)
                for name, obj in [("records.json", records), ("metrics.json", metrics), ("config.json", config)]:
                    with open(os.path.join(out_dir, name), "w") as f:
                        json.dump(obj, f, indent=None if name == "records.json" else 4)
                print(method, vocab, k, {x: round(metrics[x], 4) for x in ["noncoco_detected", "noncoco_named",
                                                                         "noncoco_reach", "coco_named"]})
            with open(os.path.join(out_dir, "metrics.json")) as f:
                rows.append({"method": method, "detector": detector, "k": k, "vocab": vocab} | json.load(f))
        pd.DataFrame(rows).to_csv(os.path.join(root, "table_end2end.csv"), index=False)
        run_qualitative(annotator, seed)


if __name__ == "__main__":
    main()
