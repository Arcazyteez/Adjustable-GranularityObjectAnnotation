"""
ex015cの実行スクリプト：COCO外の物体に対して，提案手法（greedy）の経路が正解の枝から外れる段階の分析（order_026）．

ex015で選んだ設定（ens の文章，人の役割を除く，β = 2，τ = 0.1，K = 100）で，テスト用の画像のうちCOCO外のGTと対応した加えた領域について，
- 経路が正解の枝から外れたか（fine：GT自身か下位語，coarse：GTの上位語で停止，diverged：外れた）
- 外れた段階（分岐元が physical_entity，object，whole の場合を深さ0とする），正しい子と選んだ子の確率，正しい子の順位
- 外れた後に進んだ語数，確率が最大の1語（flat）なら正しかったか
を求め，集計と例の図を保存する．再学習や新しい特徴の計算は行わない（ex015の領域のキャッシュを用いる）．

出力: outputs/ex015c_divergence_analysis/{OWLv2の重み}/42/regions.csv（領域ごと），summary.json，fig_examples.png，config.json
"""
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定論的なCUDAの行列計算に必要な設定

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import ex015b_data, matched_noncoco, modules, region_probs  # noqa: E402
from model import divergence, flat_named, step_table, summarize, word  # noqa: E402

# ===== 実験条件（order_026） =====
EXPERIMENT = "ex015c_divergence_analysis"
SEED = 42
MODELS = ["st", "ens"]
N_EXAMPLES = 8
PREFERRED = ["kangaroo.n.01"]   # 例の図に必ず含めるGT（存在する場合）
OPEN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "ex004_open_vocab_qualitative")
OPEN_IMAGES = ["kangaroo", "camel", "penguin", "helicopter", "lighthouse_1", "lighthouse_2"]  # ex004のCOCO外の実画像
YOLO_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "ex012_wordnet_granular_annotation", "weights",
                            "yolo11x.pt")
N_TRACE = 5      # 実画像で経路を記録する加えた領域の数（物体らしさの上位）


def analyze(namer, images, records, beta):
    """
    COCO外のGTと対応した加えた領域ごとに，経路が外れる段階を求める．

    Args:
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        images (list[dict]): テスト用の画像．
        records (list[list[dict]]): 画像ごとの greedy の記録．
        beta (float): 温度の倍率．

    Returns:
        list[dict]: 領域ごとの結果．
    """
    d15 = modules()[0]
    hier = namer.hier
    items = matched_noncoco(images, records)
    p, node_p = region_probs(namer, images, items, beta)
    p, node_p = p.cpu().numpy(), node_p.cpu().numpy()
    size = hier.member.sum(dim=0).cpu().numpy()   # Synsetごとの，その下にある語彙の語数
    rows = []
    for k, it in enumerate(tqdm(items, desc="divergence", leave=False)):
        gt = it["gt_synset"]
        path = [d15.ROOT] + [s for s, _ in it["rec"]["chain"]]
        r = divergence(hier, node_p[k], path, gt)
        r |= {"image_id": images[it["img"]]["image_id"], "img": it["img"], "j": it["j"], "gt_index": it["gt"], "gt": gt,
              "gt_word": word(gt), "output": it["rec"]["synset"], "chain": " > ".join(word(s) for s in path[1:]),
              "chain_p": " > ".join(f"{q:.2f}" for _, q in it["rec"]["chain"]),
              "flat_named": flat_named(hier, p[k], gt), "in_vocab": gt in hier.index,
              "flat_word": word(hier.nodes[int(hier.node_of_word[int(p[k].argmax())])])}
        if r["status"] == "diverged":
            r["n_chosen"] = int(size[hier.index[r["chosen"]]])
            r["n_correct"] = int(size[hier.index[r["correct"]]]) if r["correct"] else 0
        rows.append(r)
    return rows


def pick_examples(rows, n):
    """
    例の図に用いる，深さ0で外れた領域を選ぶ（PREFERRED のGTを優先し，GTの語が重複しないようにする）．

    Args:
        rows (list[dict]): analyze の戻り値．
        n (int): 選ぶ数．

    Returns:
        list[dict]: 選んだ領域．
    """
    top = [r for r in rows if r["status"] == "diverged" and r["top"]]
    top.sort(key=lambda r: (r["gt"] not in PREFERRED, -r["p_chosen"]))
    out, seen = [], set()
    for r in top:
        if r["gt"] in seen:
            continue
        seen.add(r["gt"])
        out.append(r)
        if len(out) == n:
            break
    return out


def draw_examples(examples, images, path):
    """
    例ごとに画像，GTのボックス（緑），加えた領域（赤），経路と分岐の確率を描いて保存する．

    Args:
        examples (list[dict]): pick_examples の戻り値．
        images (list[dict]): テスト用の画像．
        path (str): 保存先．

    Returns:
        None
    """
    from common.paths import COCO_IMAGE_DIR

    cols = 4
    rows_n = max(1, (len(examples) + cols - 1) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 4.6 * rows_n))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, examples):
        img = images[r["img"]]
        ax.imshow(Image.open(os.path.join(COCO_IMAGE_DIR, f"{int(r['image_id']):012d}.jpg")).convert("RGB"))
        for box, color in [(img["gts"][r["gt_index"]]["box"], "lime"), (img["boxes"][r["j"]]["box"], "red")]:
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color=color, lw=2))
        correct = f"{word(r['correct'])} {r['p_correct']:.2f}" if r["correct"] else "(none)"
        ax.set_title(f"GT: {r['gt_word']}\noutput: {_short(r['chain'])}\nat '{word(r['parent'])}': "
                     f"{word(r['chosen'])} {r['p_chosen']:.2f} vs {correct}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _short(chain):
    """
    図の経路から全ての経路に共通する object，whole を除く．

    Args:
        chain (str): " > " で区切った経路．

    Returns:
        str: 短くした経路．
    """
    return " > ".join(w for w in chain.split(" > ") if w not in ("object", "whole"))


def run_model(ctx, model_key, root):
    """
    1つのOWLv2の重みについて分析し，保存する．

    Args:
        ctx (dict): ex015の load_context の戻り値．
        model_key (str): OWLv2の重みの名前．
        root (str): 出力ディレクトリ．

    Returns:
        dict: 集計．
    """
    _, t15, da, _ = modules()
    db = ex015b_data()
    settings = da.selected_settings()
    owl = t15.OWLv2Regions(model_key)
    namer = db.make_namers(owl, settings)["caption"]
    images = da.load_images(ctx, "test", model_key)
    records = da.build_records(images, namer, settings)
    rows = analyze(namer, images, records, settings["beta"])
    d = os.path.join(root, model_key, str(SEED))
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "regions.csv"), index=False)
    summary = summarize(rows)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(settings | {"model": model_key, "split": "test", "seed": SEED, "k_add": da.K_ADD}, f, indent=2)
    draw_examples(pick_examples(rows, N_EXAMPLES), images, os.path.join(d, "fig_examples.png"))
    trace_open_images(owl, namer, settings, d)
    del owl, namer
    torch.cuda.empty_cache()
    return summary


def trace_region(namer, feat, beta, tau):
    """
    1つの領域の特徴から，提案手法の経路と各段階の上位の下位語を求める．

    Args:
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        feat (torch.Tensor): 領域の特徴（1×D）．
        beta (float): 温度の倍率．
        tau (float): 閾値．

    Returns:
        dict: {"chain"（[(語，確率)]），"steps"（step_table の戻り値），"flat"（確率が最大の語と確率）}．
    """
    d15, _, da, _ = modules()
    p = namer.probs(feat, beta)
    chain = da._chains(namer, p, tau)[0]
    node_p = (p @ namer.hier.member)[0].cpu().numpy()
    path = [d15.ROOT] + [s for s, _ in chain]
    w = int(p[0].argmax())
    return {"chain": [(word(s), q) for s, q in chain], "steps": step_table(namer.hier, node_p, path),
            "flat": (word(namer.hier.nodes[int(namer.hier.node_of_word[w])]), float(p[0, w]))}


def trace_open_images(owl, namer, settings, d):
    """
    ex004のCOCO外の実画像（kangaroo 等）に提案手法を適用し，YOLO11xのボックスと加えた領域の経路と各段階の確率を保存する．
    YOLO11xのボックスはクラスを出力するが，参考として同じ領域の特徴から求めた経路も記録する．

    Args:
        owl (ex015 model.OWLv2Regions): OWLv2．
        namer (ex015 train.Namer): 命名を行うオブジェクト．
        settings (dict): ex015で選んだ設定．
        d (str): 保存先．

    Returns:
        dict: 画像名 → {"yolo"，"added"}．
    """
    from torchvision.ops import box_iou
    from ultralytics import YOLO

    d15, _, _, _ = modules()
    yolo = YOLO(YOLO_WEIGHTS)
    out = {}
    for name in tqdm(OPEN_IMAGES, desc="open images", leave=False):
        image = Image.open(os.path.join(OPEN_DIR, f"{name}.jpg")).convert("RGB")
        res = yolo(image, verbose=False)[0]
        keep_y = res.boxes.conf.cpu() >= d15.YOLO_SCORE
        yb = res.boxes.xyxy.cpu()[keep_y].float()
        labels = [res.names[int(c)] for c in res.boxes.cls.cpu()[keep_y]]
        boxes, scores, feats, keep = owl.forward(image)
        rec = {"yolo": [], "added": []}
        if len(yb):
            iou, best = box_iou(yb.to(boxes.device), boxes.float()).max(dim=1)
            for b, lab, k in zip(yb.tolist(), labels, best.tolist()):
                rec["yolo"].append({"box": b, "label": lab} | trace_region(namer, feats[k:k + 1], settings["beta"],
                                                                            settings["tau"]))
            over = box_iou(boxes[keep].float(), yb.to(boxes.device)).max(dim=1).values
            keep = keep[over < 0.5]
        for k in keep[:N_TRACE].tolist():
            rec["added"].append({"box": boxes[k].tolist(), "objectness": float(scores[k])}
                                | trace_region(namer, feats[k:k + 1], settings["beta"], settings["tau"]))
        out[name] = rec
    with open(os.path.join(d, "open_images_trace.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    draw_open(out, os.path.join(d, "fig_open_images.png"))
    return out


def draw_open(trace, path):
    """
    実画像ごとに，YOLO11xのボックス（青，クラス）と加えた領域（赤，番号と出力した語）を描いて保存する．

    Args:
        trace (dict): trace_open_images の戻り値．
        path (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, rec) in zip(axes.flat, trace.items()):
        ax.imshow(Image.open(os.path.join(OPEN_DIR, f"{name}.jpg")).convert("RGB"))
        ax.axis("off")
        for color, items, fmt in [("blue", rec["yolo"], lambda i, r: f"YOLO {r['label']}"),
                                  ("red", rec["added"], lambda i, r: f"{i + 1}: {r['chain'][-1][0]} {r['chain'][-1][1]:.2f}")]:
            for i, r in enumerate(items):
                b = r["box"]
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=color, lw=2))
                ax.text(b[0], b[1], fmt(i, r), color="white", fontsize=7, va="bottom", bbox={"facecolor": color, "pad": 1})
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """全てのOWLv2の重みについて分析する．"""
    d15, t15, _, _ = modules()
    root = os.path.join(t15.OUTPUT_DIR, EXPERIMENT)
    os.makedirs(root, exist_ok=True)
    torch.manual_seed(SEED)
    ctx = d15.load_context(SEED)
    for key in MODELS:
        s = run_model(ctx, key, root)
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)},
              flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
