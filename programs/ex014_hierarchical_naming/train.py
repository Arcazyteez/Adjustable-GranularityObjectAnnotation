"""
ex014の実行スクリプト：COCO外の物体の命名の改善（order_018）．

学習は行わない．YOLO11xのボックスは全構成でex013の BDC_look と同じ出力とし，加えたOWLv2の領域の命名を比較する．
- C_look:          ex013の BDC_look（絞り込んだWordNetを上位語から先読みでたどり，ソフトマックスで停止）
- P1_flat:         キャプションの語彙（7806語）から，SigLIPで最も類似した語を1つ選ぶ（集約なし）
- P1_agg:          提案1．キャプションの語彙の確率をWordNetの上位語へ足し上げ，確率が τ 以上の最も深い語を選ぶ
- P2_agg_owl:      提案2．P1_agg の確率を，SigLIPの代わりにOWLv2の領域の特徴から求める
- P3_agg_ens:      提案3．P1_agg の文章の特徴を，複数のテンプレートと定義文の平均にする
- P23_agg_owl_ens: P2 と P3 の確率の平均
τ は検証用の画像で hF が最大となる値を選ぶ．

上限の測定（upper_bound.json）: テスト用の画像で，加えた領域のうちCOCO外の物体とIoU 0.5以上で重なるものについて，
LVISのカテゴリ（1202語，正解の語を必ず含む）から1語を選ぶ場合の正解率を，SigLIPとOWLv2で求める．

出力: outputs/ex014_hierarchical_naming/{構成}/{分割}/{seed}/metrics.json，(tau_curve.csv)
    upper_bound.json，qualitative/，および直下の table_results.csv
"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定論的なCUDAの行列計算に必要な設定

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import load_image  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from data import DATA_DIR, caption_vocabulary, ex013_modules, lvis_vocabulary  # noqa: E402
from model import Hierarchy, OWLv2Regions, text_features  # noqa: E402

# ===== 実験条件（order_018） =====
EXPERIMENT = "ex014_hierarchical_naming"
SEEDS = [42]
SPLITS = ["dev", "test"]
BASE_CONFIG = "BDC_look"  # YOLO11xのボックスと比較の基準に用いるex013の構成
TAUS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METHODS = ["C_look", "P1_flat", "P1_agg", "P2_agg_owl", "P3_agg_ens", "P23_agg_owl_ens"]
QUALITATIVE_METHODS = ["C_look", "P23_agg_owl_ens"]


def target_dir(method, split, seed):
    """
    1構成の保存先を返す．

    Args:
        method (str): 構成名．
        split (str): 分割名．
        seed (int): シード．

    Returns:
        str: 保存先のディレクトリ．
    """
    return os.path.join(OUTPUT_DIR, EXPERIMENT, method, split, str(seed))


# ===================== 前計算 =====================

def base_outputs(t13, embedder, ctx, images, tau):
    """
    ex013の BDC_look で全ての出力ボックスの候補の下位語を展開し，停止した語を返す．

    Args:
        t13 (module): ex013の train．
        embedder (ex013 の Embedder): SigLIP-B16．
        ctx (dict): ex013の load_context の戻り値．
        images (list[dict]): ex013の prepare の戻り値．
        tau (float): BDC_look の τ．

    Returns:
        tuple[list[list[dict]], list[torch.Tensor]]: 画像ごと・ボックスごとの {"synset"，"words"} と，切り抜き画像の特徴．
    """
    cfg = t13.CONFIGS[BASE_CONFIG]
    feats = t13.features(embedder, ctx["coco"], images, cfg["crop"])
    chains = t13.chains(embedder, images, feats, cfg)
    outputs = []
    for per in chains:
        row = []
        for c in per:
            synset, depth = t13.output_synset(c, True, tau)
            row.append({"synset": synset, "words": [s["word"] for s in c[:depth + 1]]})
        outputs.append(row)
    return outputs, feats


def owl_region_features(owl, ctx, images, split):
    """
    OWLv2で領域を検出し直し，加えた領域（ex013の出力ボックスの候補）の領域の特徴を返す（キャッシュする）．

    Args:
        owl (OWLv2Regions): OWLv2．
        ctx (dict): ex013の load_context の戻り値．
        images (list[dict]): ex013の prepare の戻り値．
        split (str): 分割名．

    Returns:
        list[torch.Tensor]: 画像ごとの加えた領域の特徴（加えた領域の数×768）．
    """
    path = os.path.join(DATA_DIR, f"owl_region_feats_{split}.pt")
    if os.path.exists(path):
        return [f.to("cuda") for f in torch.load(path)]
    out, max_diff = [], 0.0
    for img in tqdm(images, desc=f"owlv2 regions {split}", leave=False):
        boxes, feats = owl.regions(load_image(ctx["coco"], int(img["image_id"])))
        added = [b["box"] for b in img["boxes"] if b["source"] == "owlv2"]
        rows = []
        for box in added:
            diff = np.abs(np.asarray(boxes) - np.asarray(box)).sum(axis=1)
            k = int(diff.argmin())
            max_diff = max(max_diff, float(diff[k]))
            rows.append(k)
        out.append(feats[rows] if rows else torch.zeros(0, feats.shape[1], device=feats.device))
    print(f"owlv2 regions {split}: max box difference to ex013 = {max_diff:.4f} px", flush=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save([f.cpu() for f in out], path)
    return out


def added_probs(images, feats, kind, models, vocab):
    """
    加えた領域の語彙上の確率を計算する．

    Args:
        images (list[dict]): ex013の prepare の戻り値．
        feats (list[torch.Tensor]): SigLIPの場合は全ボックスの切り抜き画像の特徴，OWLv2の場合は加えた領域の特徴．
        kind (str): "siglip"，"siglip_ens"，"owl"．
        models (dict): "embedder"（SigLIP），"owl"（OWLv2Regions）．
        vocab (list[dict]): 語彙．

    Returns:
        list[torch.Tensor]: 画像ごとの加えた領域の確率（加えた領域の数×V）．
    """
    if kind == "owl":
        return [models["owl"].logits(f, vocab).softmax(dim=-1) for f in tqdm(feats, desc="owl probs", leave=False)]
    emb = models["embedder"]
    text = text_features(emb, vocab, ensemble=kind == "siglip_ens")
    out = []
    for img, f in zip(images, feats):
        idx = [j for j, b in enumerate(img["boxes"]) if b["source"] == "owlv2"]
        out.append((emb.scale * f[idx] @ text.T).softmax(dim=-1) if idx else torch.zeros(0, len(vocab), device=text.device))
    return out


# ===================== 出力と評価 =====================

def added_outputs(probs, hierarchy, vocab, method, tau):
    """
    加えた領域の出力する語を決める．

    Args:
        probs (list[torch.Tensor]): added_probs の戻り値．
        hierarchy (Hierarchy): 上位語への所属．
        vocab (list[dict]): 語彙．
        method (str): 構成名（"P1_flat" は集約しない）．
        tau (float): 閾値．

    Returns:
        list[list[dict]]: 画像ごと・加えた領域ごとの {"synset"，"words"，"p"}．
    """
    out = []
    for p in probs:
        if len(p) == 0:
            out.append([])
        elif method == "P1_flat":
            best = p.argmax(dim=1).tolist()
            out.append([{"synset": vocab[b]["synset"], "words": [vocab[b]["word"]], "p": float(p[i, b])}
                        for i, b in enumerate(best)])
        else:
            from nltk.corpus import wordnet as wn
            out.append([{"synset": s, "words": [wn.synset(s).lemma_names()[0].replace("_", " ")], "p": q}
                        for s, q in hierarchy.choose(p, tau)])
    return out


def merged_chains(images, base, added):
    """
    YOLO11xのボックスの出力（base）と加えた領域の出力（added）を，ex013の evaluate に渡す形にまとめる．

    Args:
        images (list[dict]): ex013の prepare の戻り値．
        base (list[list[dict]]): base_outputs の出力．
        added (list[list[dict]] | None): added_outputs の出力（None は base をそのまま用いる）．

    Returns:
        list[list[list[dict]]]: 画像ごと・ボックスごとの1段の展開結果．
    """
    out = []
    for i, img in enumerate(images):
        per, k = [], 0
        for j, b in enumerate(img["boxes"]):
            o = base[i][j] if (b["source"] == "yolo" or added is None) else added[i][k]
            k += b["source"] == "owlv2"
            per.append([{"synset": o["synset"], "word": " > ".join(o["words"]), "soft": 1.0, "cos": 0.0, "n_candidates": 1}])
        out.append(per)
    return out


def evaluate(t13, images, chains, keep=None):
    """
    ex013の evaluate で指標を計算し，加えた領域がCOCO外の物体に対応した場合の命名の正解率と深さを加える．

    Args:
        t13 (module): ex013の train．
        images (list[dict]): ex013の prepare の戻り値．
        chains (list): merged_chains の戻り値．
        keep (set[str] | None): 出力を記録する画像ID．

    Returns:
        dict: ex013の指標と，added_noncoco_n / named / fine / named_depth（加えた領域がIoU 0.5以上で重なるCOCO外の物体についての値）．
    """
    d13 = ex013_modules()[0]
    metrics = t13.evaluate(images, chains, {"add": t13.K_ADD, "stop": False}, 0.0, keep)
    n = named = fine = 0
    depths = []
    from nltk.corpus import wordnet as wn
    for img, per in zip(images, chains):
        for j, (b, c) in enumerate(zip(img["boxes"], per)):
            if b["source"] != "owlv2" or not len(img["gts"]):
                continue
            g = int(np.argmax(img["iou"][j]))
            if img["iou"][j][g] < 0.5 or img["gts"][g]["kind"] != "noncoco":
                continue
            t, s = img["gts"][g]["synset"], c[0]["synset"]
            n += 1
            ok = bool(t13.evaluated_set(s)) and (s in d13.ancestors(t) or t in d13.ancestors(s))
            named += ok
            fine += t in d13.ancestors(s)
            if ok:
                depths.append(wn.synset(s).min_depth())
    return metrics | {"added_noncoco_n": n, "added_noncoco_named": named / max(n, 1), "added_noncoco_fine": fine / max(n, 1),
                      "added_noncoco_named_depth": float(np.mean(depths)) if depths else float("nan")}


def upper_bound(t13, images, feats, owl_feats, models, cap_vocab):
    """
    加えた領域のうちCOCO外の物体とIoU 0.5以上で重なるものについて，LVISのカテゴリから1語を選ぶ場合の正解率を求める．

    Args:
        t13 (module): ex013の train．
        images (list[dict]): テスト用の prepare の戻り値．
        feats (list[torch.Tensor]): SigLIPの切り抜き画像の特徴．
        owl_feats (list[torch.Tensor]): OWLv2の加えた領域の特徴．
        models (dict): "embedder"，"owl"．
        cap_vocab (list[dict]): キャプションの語彙．

    Returns:
        dict: n，{siglip, owl}_{top1_exact, top5_exact, top1_on_path}，caption_vocab_has_gt，caption_vocab_has_gt_or_ancestor．
    """
    d13 = ex013_modules()[0]
    lvis = lvis_vocabulary()
    text = text_features(models["embedder"], lvis)
    cap = {v["synset"] for v in cap_vocab}
    res = {"siglip": [], "owl": []}
    targets = []
    for img, f, of in zip(images, feats, owl_feats):
        idx = [j for j, b in enumerate(img["boxes"]) if b["source"] == "owlv2"]
        if not idx or not len(img["gts"]):
            continue
        sig = (f[idx] @ text.T)
        owl = models["owl"].logits(of, lvis)
        for k, j in enumerate(idx):
            g = int(np.argmax(img["iou"][j]))
            if img["iou"][j][g] < 0.5 or img["gts"][g]["kind"] != "noncoco":
                continue
            t = img["gts"][g]["synset"]
            targets.append(t)
            for name, scores in [("siglip", sig[k]), ("owl", owl[k])]:
                top = scores.topk(5).indices.tolist()
                pred = lvis[top[0]]["synset"]
                res[name].append((pred == t, t in [lvis[i]["synset"] for i in top],
                                  pred in d13.ancestors(t) or t in d13.ancestors(pred)))
    out = {"n": len(targets), "lvis_vocab_size": len(lvis), "caption_vocab_size": len(cap_vocab),
           "caption_vocab_has_gt": float(np.mean([t in cap for t in targets])),
           "caption_vocab_has_gt_or_ancestor": float(np.mean([bool((d13.ancestors(t) & cap) - {"whole.n.02"})
                                                               for t in targets]))}
    for name, rows in res.items():
        arr = np.array(rows, dtype=float)
        out |= {f"{name}_top1_exact": float(arr[:, 0].mean()), f"{name}_top5_exact": float(arr[:, 1].mean()),
                f"{name}_top1_on_path": float(arr[:, 2].mean())}
    return out


def draw(coco, outputs, path, title):
    """
    出力ボックスと単語を画像に描画して保存する．

    Args:
        coco (COCO): COCOオブジェクト．
        outputs (dict): 画像ID → {"box"，"source"，"words"} のリスト．
        path (str): 保存先．
        title (str): 図の見出し．

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (image_id, items) in zip(axes.flat, sorted(outputs.items())):
        ax.imshow(load_image(coco, int(image_id)))
        ax.axis("off")
        ax.set_title(f"{title}: image {image_id}", fontsize=9)
        for it in items:
            x0, y0, x1, y1 = it["box"]
            color = "tab:blue" if it["source"] == "yolo" else "tab:red"
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
            ax.text(x0, y0, " > ".join(it["words"]), fontsize=6, color="white", va="top",
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===================== 実行 =====================

def main():
    """上限の測定と，全構成の τ の選択・評価を行い，集計表を作成する．"""
    d13, m13, t13 = ex013_modules()
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    for seed in SEEDS:
        start = time.time()
        ctx = d13.load_context(seed)
        data = {s: t13.prepare(ctx, d13.owlv2_proposals(ctx, s, None), s) for s in SPLITS}
        with open(os.path.join(OUTPUT_DIR, t13.EXPERIMENT, BASE_CONFIG, "test", str(seed), "metrics.json")) as f:
            base_tau = json.load(f)["tau"]
        models = {"embedder": m13.Embedder("SigLIP-B16", seed)}
        base, feats = {}, {}
        for s in SPLITS:
            base[s], feats[s] = base_outputs(t13, models["embedder"], ctx, data[s], base_tau)
        models["owl"] = OWLv2Regions()
        owl_feats = {s: owl_region_features(models["owl"], ctx, data[s], s) for s in SPLITS}
        cap_vocab = caption_vocabulary()
        hierarchy = Hierarchy(cap_vocab)
        os.makedirs(root, exist_ok=True)
        bound = upper_bound(t13, data["test"], feats["test"], owl_feats["test"], models, cap_vocab)
        with open(os.path.join(root, "upper_bound.json"), "w") as f:
            json.dump(bound, f, indent=4)
        print("upper bound", {k: round(v, 4) if isinstance(v, float) else v for k, v in bound.items()}, flush=True)
        probs = {}
        for s in SPLITS:
            probs[("siglip", s)] = added_probs(data[s], feats[s], "siglip", models, cap_vocab)
            probs[("siglip_ens", s)] = added_probs(data[s], feats[s], "siglip_ens", models, cap_vocab)
            probs[("owl", s)] = added_probs(data[s], owl_feats[s], "owl", models, cap_vocab)
            probs[("owl_ens", s)] = [(a + b) / 2 for a, b in zip(probs[("owl", s)], probs[("siglip_ens", s)])]
        kind_of = {"P1_flat": "siglip", "P1_agg": "siglip", "P2_agg_owl": "owl", "P3_agg_ens": "siglip_ens",
                   "P23_agg_owl_ens": "owl_ens"}
        keep = set()
        q_path = os.path.join(OUTPUT_DIR, t13.EXPERIMENT, "qualitative", "ex012.json")  # ex013の定性評価と同じ6枚
        if os.path.exists(q_path):
            with open(q_path) as f:
                keep = set(json.load(f))
        rows = []
        for method in METHODS:
            tau, curve = None, None
            if method == "C_look":
                chains = {s: merged_chains(data[s], base[s], None) for s in SPLITS}
            elif method == "P1_flat":
                chains = {s: merged_chains(data[s], base[s], added_outputs(probs[(kind_of[method], s)], hierarchy,
                                                                          cap_vocab, method, 0.0)) for s in SPLITS}
            else:
                curve_rows = []
                for t in TAUS:
                    ch = merged_chains(data["dev"], base["dev"],
                                       added_outputs(probs[(kind_of[method], "dev")], hierarchy, cap_vocab, method, t))
                    curve_rows.append({"tau": t} | evaluate(t13, data["dev"], ch))
                curve = pd.DataFrame(curve_rows)
                tau = float(curve.loc[curve["hF"].idxmax(), "tau"])
                chains = {s: merged_chains(data[s], base[s], added_outputs(probs[(kind_of[method], s)], hierarchy,
                                                                          cap_vocab, method, tau)) for s in SPLITS}
            for s in SPLITS:
                metrics = evaluate(t13, data[s], chains[s], keep if s == "test" else None)
                outputs = metrics.pop("outputs", None)
                d = target_dir(method, s, seed)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "metrics.json"), "w") as f:
                    json.dump(metrics | {"tau": tau}, f, indent=4)
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump({"method": method, "probability": kind_of.get(method), "tau": tau, "taus": TAUS,
                               "base_config": BASE_CONFIG, "base_tau": base_tau, "vocab_size": len(cap_vocab),
                               "seed": seed, "elapsed_sec": time.time() - start}, f, indent=4)
                if s == "dev" and curve is not None:
                    curve.to_csv(os.path.join(d, "tau_curve.csv"), index=False)
                if s == "test":
                    rows.append({"method": method, "tau": tau} | metrics)
                    if method in QUALITATIVE_METHODS and outputs:
                        q_dir = os.path.join(root, "qualitative")
                        os.makedirs(q_dir, exist_ok=True)
                        for items in outputs.values():
                            for it in items:
                                it["words"] = [w for w in " > ".join(it["words"]).split(" > ")]
                        with open(os.path.join(q_dir, f"{method}.json"), "w") as f:
                            json.dump(outputs, f, indent=1)
                        draw(ctx["coco"], outputs, os.path.join(q_dir, f"grid_{method}.png"), method)
            print(method, "tau", tau, {k: round(rows[-1][k], 4) for k in ["noncoco_detected", "noncoco_named", "noncoco_fine",
                                                                           "noncoco_hR", "coco_named", "hF", "added_precision",
                                                                           "added_noncoco_named", "added_noncoco_named_depth"]},
                  flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(root, "table_results.csv"), index=False)


if __name__ == "__main__":
    main()
