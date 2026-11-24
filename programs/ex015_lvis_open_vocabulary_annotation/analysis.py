"""
ex015の追加の分析：加える領域の数 K と物体らしさによる棄却の閾値 o のトレードオフ，およびex014相当の構成の再現．

train.py の S4 は fed_hF（GTと対応しない出力のうち，存在しないと確認されたカテゴリ等を出力したもののみ誤りとする）で K を選ぶため，
K を増やすほど誤りと判定されない出力が増える．そこで，GTと対応しない出力を全て誤りとする strict_hF と，加えた領域の適合率を併せて，
K と o の組の全てを検証用とテスト用で評価し，strict_hF で選んだ「適合率重視」の設定も求める．

出力: outputs/ex015_lvis_open_vocabulary_annotation/tradeoff_{model}.csv，operating_points.csv，fig_tradeoff.png，
    ex014_reproduction.json
"""
import itertools
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import OUTPUT_DIR  # noqa: E402
from data import DATA_DIR, SPLITS, build_images, caption_vocabulary, load_context, region_cache  # noqa: E402
from train import (EXPERIMENT, Namer, OWLv2Regions, added_outputs, assemble, attach_checked, evaluate, vocab_variant,  # noqa: E402
                   yolo_classes)

KS = [10, 20, 50, 100]
OBJECTNESS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
MODELS = ["st", "ens", "st_large"]
KEYS = ["fed_hF", "strict_hF", "hF", "noncoco_detected", "noncoco_named", "noncoco_fine", "noncoco_hR", "coco_named",
        "added_per_image", "added_precision"]


def outputs_for(ctx, key, settings):
    """
    選んだ設定で，1つのOWLv2の重みの出力（加えた領域は最大 max(KS) 件）を検証用とテスト用について求める．

    Args:
        ctx (dict): load_context の戻り値．
        key (str): OWLv2の重みの名前．
        settings (dict): train.py で選んだ設定．

    Returns:
        tuple[dict, dict]: 分割 → build_images の戻り値，分割 → 出力．
    """
    owl = OWLv2Regions(key)
    namer = Namer(owl, vocab_variant(caption_vocabulary(), settings["exclude_person"]), settings["text"])
    images, outs = {}, {}
    for s in SPLITS:
        images[s] = build_images(ctx, s, region_cache(ctx, s, key, None))
        attach_checked(images[s], ctx["lvis"])
        a = added_outputs(namer, images[s], settings["beta"], settings["rule"], [settings["tau"]], max(KS))[settings["tau"]]
        outs[s] = assemble(images[s], a, yolo_classes(images[s]))
    return images, outs, owl


def save_qualitative(images, outs, key, q_dir, n_added=8, objectness=0.1):
    """
    定性評価の画像（ex013と同じ6枚）について，YOLO11xのボックスと，物体らしさが objectness 以上の加えた領域の上位 n_added 件の出力を保存する．

    Args:
        images (list[dict]): テスト用の build_images の戻り値．
        outs (list[list[str]]): テスト用の出力．
        key (str): OWLv2の重みの名前．
        q_dir (str): 保存先．
        n_added (int): 表示する加えた領域の数．
        objectness (float): 物体らしさの下限．

    Returns:
        None
    """
    from nltk.corpus import wordnet as wn
    keep = set(json.load(open(os.path.join(OUTPUT_DIR, "ex013_open_vocabulary_granular", "qualitative", "ex012.json"))))
    res = {}
    for img, out in zip(images, outs):
        if img["image_id"] not in keep:
            continue
        items, n = [], 0
        for b, syn in zip(img["boxes"], out):
            if b["source"] == "owlv2":
                if b["score"] < objectness or n >= n_added:
                    continue
                n += 1
            items.append({"box": b["box"], "source": b["source"], "synset": syn,
                          "word": wn.synset(syn).lemma_names()[0].replace("_", " ")})
        res[img["image_id"]] = items
    os.makedirs(q_dir, exist_ok=True)
    with open(os.path.join(q_dir, f"tradeoff_{key}.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    """K と o のトレードオフを評価し，図と表を保存する．"""
    root = os.path.join(OUTPUT_DIR, EXPERIMENT)
    settings = json.load(open(os.path.join(root, "selected_settings.json")))
    ctx = load_context(42)
    rows, points = [], []
    for key in MODELS:
        if not all(os.path.exists(os.path.join(DATA_DIR, f"regions_{key}_{s}_n{len(ctx['splits'][s])}.pt")) for s in SPLITS):
            continue
        images, outs, owl = outputs_for(ctx, key, settings)
        grid = []
        for k, o in itertools.product(KS, OBJECTNESS):
            r = {"model": key, "k_add": k, "objectness": o}
            for s in SPLITS:
                m = evaluate(images[s], outs[s], k, o)
                r |= {f"{s}_{x}": m[x] for x in KEYS}
            grid.append(r)
        save_qualitative(images["test"], outs["test"], key, os.path.join(root, "qualitative"))
        df = pd.DataFrame(grid)
        df.to_csv(os.path.join(root, f"tradeoff_{key}.csv"), index=False)
        rows.append(df)
        for crit in ["fed_hF", "strict_hF"]:
            best = df.loc[df[f"dev_{crit}"].idxmax()]
            points.append({"model": key, "criterion": crit, "k_add": int(best["k_add"]), "objectness": best["objectness"]}
                          | {x: best[f"test_{x}"] for x in KEYS})
        if key == "ens":
            # ex014のP2_agg_owl相当（ensの領域の特徴，deepest，plain，β=1，τ=0.05，K=10，YOLO11xはクラスのみ）
            namer = Namer(owl, caption_vocabulary(), "plain")
            rep = {}
            for s in SPLITS:
                a = added_outputs(namer, images[s], 1.0, "deepest", [0.05], 10)[0.05]
                rep[s] = evaluate(images[s], assemble(images[s], a, yolo_classes(images[s])), 10, 0.0)
            with open(os.path.join(root, "ex014_reproduction.json"), "w") as f:
                json.dump(rep, f, indent=4)
            print("ex014相当", {x: round(rep["test"][x], 4) for x in KEYS}, flush=True)
        del owl
        torch.cuda.empty_cache()
    pts = pd.DataFrame(points)
    pts.to_csv(os.path.join(root, "operating_points.csv"), index=False)
    print(pts.round(4).to_string(), flush=True)
    # 図: COCO外の物体の命名（再現率側）と加えた領域の適合率のトレードオフ
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for df, marker in zip(rows, ["o", "s", "^"]):
        key = df["model"].iloc[0]
        d0 = df[df["objectness"] == 0.0].sort_values("k_add")
        ax.plot(d0["test_added_precision"], d0["test_noncoco_named"], marker=marker, label=f"{key} (o=0, K=10..100)")
        for _, r in d0.iterrows():
            ax.annotate(f"K={int(r['k_add'])}", (r["test_added_precision"], r["test_noncoco_named"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Precision of added regions")
    ax.set_ylabel("Non-COCO objects correctly named")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "fig_tradeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
