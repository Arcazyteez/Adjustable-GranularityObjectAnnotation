"""
ex006の実行スクリプト：YOLOとCLIPを組み合わせた粒度可変アノテーション（ハイブリッド構成）の評価．

学習は行わない．深さ0のカテゴリをYOLO11x（COCOの80クラス）の出力とし，そのSynsetから式(3)に従って
WordNetの下位語をたどり，深さ1以降はYOLOのボックスのクロップ（pad）をCLIP-ViT-B/32で分類する．
- hybrid: 深さ0は常にYOLOの出力とする
- hybrid_switch: CLIPがCOCOの80クラスとCOCO外の22語（計102語）の中でCOCO外の語を最上位に選んだ場合のみ，
  深さ0をCLIPの語に切り替える
1. coco500: COCO val2017の500枚で，深さ0の正解率，正命名率，深さごとの確信度を集計し，提案手法（ex003）と比較する
2. qualitative: ex004の実画像6枚とCOCO画像（論文の図4，図7）で定性評価を行う

出力: outputs/ex006_coco_val500_yolo_clip_hybrid/{手法}/{画像集合}_conf{閾値}/{seed}/
"""
import itertools
import json
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, load_coco, make_crop  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.wordnet_utils import COCO_TO_SYNSET, EXTRA_TO_SYNSET, hierarchical_annotation  # noqa: E402
from data import load_images, load_n_gt, load_proposed_chains, load_qualitative_samples, load_yolo_instances  # noqa: E402
from model import load_model, load_vlm_model  # noqa: E402

# ===== 実験条件（order_006） =====
EXPERIMENT = "ex006_coco_val500_yolo_clip_hybrid"
SEEDS = [42]
N_IMAGES = 500
YOLO_NAME = "yolo11x"
CONFS = [0.25, 0.5, 0.7]  # ex005と同じ検出スコアの閾値
MODES = ["hybrid", "hybrid_switch", "hybrid_switch_c0.5"]  # hybrid_switch_c0.5 は1回目の実行結果を受けて追加した
VLM_NAME = "CLIP-ViT-B32"
CROP_MODE = "pad"
PROPOSED_REFS = ["detr_threshold0.5_crop_pad", "detr_threshold0.97_crop_pad"]  # 比較する提案手法（ex003）の条件
QUALITATIVE_CONF = 0.25
QUALITATIVE_MODES = ["hybrid_switch", "hybrid_switch_c0.5"]
WORD_TO_SYNSET = COCO_TO_SYNSET | EXTRA_TO_SYNSET


def target_dir(method, condition, seed):
    """
    結果の保存先ディレクトリを返す．

    Args:
        method (str): 手法名．
        condition (str): 条件名．
        seed (int): シード．

    Returns:
        str: ディレクトリのパス．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, condition, str(seed))


def save_json(path, obj):
    """
    JSONファイルを保存する（親ディレクトリも作成する）．

    Args:
        path (str): 保存先．
        obj: JSONに変換可能なオブジェクト．

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def decide_root(yolo_word, yolo_conf, clip_scores, vocab, extra_words, mode):
    """
    深さ0のカテゴリを決定する．

    hybrid ではYOLOの出力をそのまま用いる．hybrid_switch では，CLIPの最上位の語がCOCO外の語である場合のみ
    CLIPの語に切り替える．hybrid_switch_c{γ}（例: hybrid_switch_c0.5）では，さらにその語のCLIPの確信度が γ 以上の
    場合のみ切り替える．

    Args:
        yolo_word (str): YOLOが出力したクラス名．
        yolo_conf (float): YOLOの検出スコア．
        clip_scores (np.ndarray | None): 候補語彙 vocab 上のCLIPのスコア（長さV）．hybrid では None でよい．
        vocab (list[str]): CLIPの候補語彙（COCOの80クラス + COCO外の語）．
        extra_words (set[str]): COCO外の語の集合．
        mode (str): "hybrid"，"hybrid_switch"，"hybrid_switch_c{γ}" のいずれか．

    Returns:
        tuple[str, float, str]: 深さ0の単語，確信度，決定した手法（"yolo" または "clip"）．
    """
    if mode == "hybrid":
        return yolo_word, float(yolo_conf), "yolo"
    min_conf = float(mode.split("_c")[1]) if "_c" in mode else 0.0
    best = int(np.argmax(clip_scores))
    if vocab[best] in extra_words and clip_scores[best] >= min_conf:
        return vocab[best], float(clip_scores[best]), "clip"
    return yolo_word, float(yolo_conf), "yolo"


def depth_statistics(records):
    """
    深さ1以降のインスタンス数・平均候補数・平均確信度を集計する（ex003の表4と同じ集計）．

    深さ0の確信度はYOLOの検出スコアとCLIPのソフトマックスが混在し，深さ1以降と比較できないため，件数のみを集計する．

    Args:
        records (list[dict]): "chain" を含むインスタンスの結果．

    Returns:
        pd.DataFrame: 列 depth, n_instances, mean_candidates, mean_conf, std_conf．
    """
    steps = [step for r in records for step in (r.get("chain") or [])]
    df = pd.DataFrame(steps)
    stats = (df.groupby("depth")
               .agg(n_instances=("conf", "size"), mean_candidates=("n_candidates", "mean"),
                    mean_conf=("conf", "mean"), std_conf=("conf", "std"))
               .reset_index())
    stats.loc[stats["depth"] == 0, ["mean_candidates", "mean_conf", "std_conf"]] = np.nan
    return stats


def annotate_instances(vlm, instances, images, vocab, mode):
    """
    COCOの評価インスタンスの深さ0を決定し，深さ0が正解のインスタンスについてWordNet階層をたどる．

    Args:
        vlm (CLIPModel): VLM．
        instances (list[dict]): load_yolo_instances の戻り値．
        images (dict): 画像ID → PIL.Image.Image．
        vocab (list[str]): CLIPの候補語彙（102語）．
        mode (str): "hybrid"，"hybrid_switch"，"hybrid_switch_c{γ}" のいずれか．

    Returns:
        list[dict]: インスタンスごとの {"image_id", "gt_index", "gt_name", "yolo_word", "root_word", "root_source",
            "top1_correct", "chain"（深さ0が正解のときのみ）}．
    """
    crops = [make_crop(images[i["image_id"]], i["box"], CROP_MODE) for i in instances]
    # 切替を行う構成（hybrid_switch，hybrid_switch_c{γ}）ではCLIPのスコアを計算する（order_008で hybrid_switch_c{γ} の計算漏れを修正）
    clip_scores = vlm.score(crops, vocab) if mode != "hybrid" else [None] * len(instances)
    extra_words = set(EXTRA_TO_SYNSET)
    records = []
    for ins, crop, scores in tqdm(list(zip(instances, crops, clip_scores)), desc=mode, leave=False):
        word, conf, source = decide_root(ins["top5"][0], ins["det_score"], scores, vocab, extra_words, mode)
        record = {"image_id": ins["image_id"], "gt_index": ins["gt_index"], "gt_name": ins["gt_name"],
                  "yolo_word": ins["top5"][0], "root_word": word, "root_source": source,
                  "top1_correct": word == ins["gt_name"]}
        if record["top1_correct"]:
            record["chain"] = hierarchical_annotation(vlm, crop, word, WORD_TO_SYNSET[word], conf)
            record["chain"][0]["n_candidates"] = len(vocab) if source == "clip" else 80
        records.append(record)
    return records


def compare_chains(records, proposed):
    """
    両手法の深さ0が正解となった共通のGTインスタンスで，深さ1以降の推定単語の一致率を求める．

    Args:
        records (list[dict]): annotate_instances の戻り値．
        proposed (dict): load_proposed_chains の戻り値．

    Returns:
        dict: 共通インスタンス数，深さ1の単語の一致率，系列全体の一致率，各手法の平均の深さ．
    """
    pairs = [(r["chain"], proposed[(r["image_id"], r["gt_index"])]["chain"]) for r in records
             if r.get("chain") and proposed.get((r["image_id"], r["gt_index"]), {}).get("chain")]
    words = [([s["word"] for s in a], [s["word"] for s in b]) for a, b in pairs]
    with_depth1 = [(a, b) for a, b in words if len(a) > 1 and len(b) > 1]
    return {"n_common": len(pairs), "n_common_with_depth1": len(with_depth1),
            "depth1_agreement": float(np.mean([a[1] == b[1] for a, b in with_depth1])) if with_depth1 else None,
            "chain_agreement": float(np.mean([a == b for a, b in words])) if words else None,
            "hybrid_mean_depth": float(np.mean([len(a) - 1 for a, _ in words])) if words else None,
            "proposed_mean_depth": float(np.mean([len(b) - 1 for _, b in words])) if words else None}


def summarize(records, n_gt):
    """
    深さ0の正解率，正命名率，階層アノテーションの対象となったインスタンス数を計算する．

    Args:
        records (list[dict]): annotate_instances の戻り値．
        n_gt (int): GTインスタンス数の合計．

    Returns:
        dict: 評価指標．
    """
    correct = np.array([r["top1_correct"] for r in records])
    switched = np.array([r["root_source"] == "clip" for r in records])
    return {"n_instances": len(records), "n_gt": n_gt, "top1_accuracy": float(correct.mean()),
            "annotation_recall": float(correct.sum() / n_gt), "n_hierarchy": int(correct.sum()),
            "n_switched": int(switched.sum()), "n_switched_correct": int((switched & correct).sum()),
            "n_switched_from_correct_yolo": int(sum(r["root_source"] == "clip" and r["yolo_word"] == r["gt_name"]
                                                    for r in records)),
            "mean_depth": float(np.mean([len(r["chain"]) - 1 for r in records if r.get("chain")]))}


def run_coco500(vlm, vocab, n_gt, proposed_refs, seed):
    """
    全閾値・全構成でCOCO 500枚の評価を行う（完了済みの条件はスキップ）．

    Args:
        vlm (CLIPModel): VLM．
        vocab (list[str]): CLIPの候補語彙（102語）．
        n_gt (int): GTインスタンス数の合計．
        proposed_refs (dict): 条件名 → load_proposed_chains の戻り値．
        seed (int): シード．

    Returns:
        None
    """
    images = None
    for conf, mode in itertools.product(CONFS, MODES):
        out_dir = target_dir(mode, f"coco500_conf{conf}", seed)
        if os.path.exists(os.path.join(out_dir, "metrics.json")):
            continue
        instances = load_yolo_instances(YOLO_NAME, conf, seed)
        images = images or load_images(i["image_id"] for i in load_yolo_instances(YOLO_NAME, min(CONFS), seed))
        start = time.time()
        records = annotate_instances(vlm, instances, images, vocab, mode)
        elapsed = time.time() - start
        metrics = summarize(records, n_gt)
        save_json(os.path.join(out_dir, "instances.json"), records)
        save_json(os.path.join(out_dir, "metrics.json"), metrics)
        depth_statistics(records).to_csv(os.path.join(out_dir, "depth_stats.csv"), index=False)
        for ref, proposed in proposed_refs.items():
            save_json(os.path.join(out_dir, f"chain_agreement_vs_{ref}.json"), compare_chains(records, proposed))
        save_json(os.path.join(out_dir, "config.json"), {"yolo": YOLO_NAME, "conf": conf, "mode": mode, "vlm": VLM_NAME,
                                                         "crop_mode": CROP_MODE, "vocab_size": len(vocab), "seed": seed,
                                                         "n_images": N_IMAGES, "elapsed_sec": elapsed})
        print(mode, conf, metrics)


def annotate_image(detector, vlm, image, vocab, mode):
    """
    1枚の画像をYOLOで検出し，各領域の深さ0を決定してWordNet階層をたどる．

    Args:
        detector (YOLODetector): 検出器．
        vlm (CLIPModel): VLM．
        image (PIL.Image.Image): 入力画像．
        vocab (list[str]): CLIPの候補語彙（102語）．
        mode (str): "hybrid" または "hybrid_switch"．

    Returns:
        list[dict]: 領域ごとの {"box", "det_score", "yolo_word", "root_source", "chain"}．
    """
    det = detector.detect(image)
    if not det["boxes"]:
        return []
    crops = [make_crop(image, box, CROP_MODE) for box in det["boxes"]]
    clip_scores = vlm.score(crops, vocab)
    regions = []
    for box, score, top5, crop, scores in zip(det["boxes"], det["scores"], det["top5"], crops, clip_scores):
        word, conf, source = decide_root(top5[0], score, scores, vocab, set(EXTRA_TO_SYNSET), mode)
        chain = hierarchical_annotation(vlm, crop, word, WORD_TO_SYNSET[word], conf)
        regions.append({"box": box, "det_score": score, "yolo_word": top5[0], "root_source": source, "chain": chain})
    return regions


def draw_regions(ax, image, regions, title):
    """
    画像に検出ボックスと各段階の推定単語・確信度を描画する（深さ0の決定がCLIPのときは「CLIP:」を付す）．

    Args:
        ax (matplotlib.axes.Axes): 描画先．
        image (PIL.Image.Image): 入力画像．
        regions (list[dict]): annotate_image の戻り値．
        title (str): タイトル．

    Returns:
        None
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region["box"]
        color = plt.cm.tab10.colors[i % 10]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
        head = "CLIP: " if region["root_source"] == "clip" else "YOLO: "
        lines = [head + f"{s['word']} ({s['conf']:.2f})" if s["depth"] == 0 else f"> {s['word']} ({s['conf']:.2f})"
                 for s in region["chain"]]
        ax.text(x0, y0, "\n".join(lines), fontsize=6, color="white", va="top", bbox=dict(facecolor=color, alpha=0.75, pad=1))


def run_qualitative(vlm, vocab, seed):
    """
    ex004の実画像とCOCO画像でハイブリッド構成の定性評価を行う（完了済みの条件はスキップ）．

    Args:
        vlm (CLIPModel): VLM．
        vocab (list[str]): CLIPの候補語彙（102語）．
        seed (int): シード．

    Returns:
        None
    """
    image_sets = None
    for mode, set_name in itertools.product(QUALITATIVE_MODES, ["open_vocab", "coco_qualitative"]):
        out_dir = target_dir(mode, f"{set_name}_conf{QUALITATIVE_CONF}", seed)
        if os.path.exists(os.path.join(out_dir, "annotations.json")):
            continue
        image_sets = image_sets or load_qualitative_samples()
        detector = load_model(YOLO_NAME, QUALITATIVE_CONF, seed)
        samples = image_sets[set_name]
        results = {s["name"]: annotate_image(detector, vlm, s["image"], vocab, mode) for s in samples}
        save_json(os.path.join(out_dir, "annotations.json"),
                  {s["name"]: {"meta": s["meta"], "regions": results[s["name"]]} for s in samples})
        n_cols = 3 if len(samples) <= 6 else 4
        n_rows = (len(samples) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        for ax in axes.flat:
            ax.axis("off")
        for ax, s in zip(axes.flat, samples):
            draw_regions(ax, s["image"], results[s["name"]], f"{s['name']} ({len(results[s['name']])} boxes)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grid.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        for name, regions in results.items():
            print(mode, set_name, name, [" > ".join(f"{s['word']}({s['conf']:.2f})" for s in r["chain"]) for r in regions])
        del detector
        torch.cuda.empty_cache()


def main():
    """全条件の評価を実行する（完了済みの条件はスキップ）．"""
    for seed in SEEDS:
        vocab = category_names(load_coco("instances")) + list(EXTRA_TO_SYNSET)
        n_gt = load_n_gt(YOLO_NAME, min(CONFS), seed)
        proposed_refs = {ref: load_proposed_chains(ref, N_IMAGES, seed) for ref in PROPOSED_REFS}
        vlm = load_vlm_model(VLM_NAME, seed)
        run_coco500(vlm, vocab, n_gt, proposed_refs, seed)
        run_qualitative(vlm, vocab, seed)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
