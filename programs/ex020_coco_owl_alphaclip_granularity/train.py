"""
ex020：提案手法の粒度の変更（深さ1以降の細分化）までの評価と，VG.pdf の語彙・命名の手順の追加（order_031，order_032）．

学習は行わず，学習済みのモデルの推論と評価のみを行う．データは MSCOCO val2017 のみ（ex017・ex019 と同じ dev 500枚，test 500枚）．
VG の注釈は語彙の構成だけに用い，COCO val2017 と重複する画像は集計から除く．
結果は outputs/ex020_coco_owl_alphaclip_granularity/{実験}/{手法}+{手順}/{条件}/{seed}/ に保存し，summary.json がある条件はスキップする．

比較する手法（order_032）：
- New：OWLv2@20 + Alpha-CLIP（context1.5）+ WordNet（新しい提案手法．棄却は joint）
- Old：DETR + CLIP-ViT-B/32 + WordNet（古い提案手法．棄却は VLM の確信度）
- YOLO11x：単体（COCO のクラスのみを出力する）
深さ0の棄却 θ0 と残すボックスは ex019 の E2 で dev の F値で選んだ設定をそのまま用いる（命名の手順だけを比べるため）．

命名の手順（VLM は SigLIP ではなく各手法の F．閾値などは dev の hF で選ぶ）：
- coarse：COCO のクラスのみ（深さ0）．YOLO11x 単体はこの手順のみ
- wn_hyponym：COCO の80クラスから WordNet の直接の下位語をたどる細分化（これまでの手順）．θ_f × D
- vg_tree：COCO の80クラスから VG の語彙の木（枠＋WordNet．data.build_tree）の子をたどる細分化．θ_f × D
- vg_flat：VG の語彙（＋COCO の80クラス）の平坦な命名．τ
- vg_subtree：部分木の確率の合計（VG.pdf の提案手法の集約）．T × τ

評価（VG.pdf の式(8)）：
- 正解：COCO のキャプションの名詞の第1語義のうち，正解クラスの真の下位語（data.py）．
- hP，hR，hF：出力と正解の Synset 自身と，深さ4以上の上位語の集合で求める．COCO の80クラスの Synset は深さによらず含める．
- その他の指標：
  - 命名の正解率：出力が正解の Synset 自身，または深さ4以上の上位語である割合
  - 完全一致率
  - 細分化の正解率：命名が正しく，COCO のクラスより細かい割合
  - クラスの正解率：全ての正解インスタンスで，出力が正解クラスまたはその下位語である割合

実験：
- E0_coverage：VLM を用いない語彙の被覆率と hR の上限（VG.pdf 5.2節）
- E1_gt：正解のボックス × F ∈ {CLIP-ViT-B32，Alpha-CLIP} × 命名の手順
- E2_pipeline：3手法 × 命名の手順．hF の差は画像単位の対応のあるブートストラップで検定する
- E3_step：正解のボックスで正解の経路を1段ずつ与えた場合の各段の正解率（WordNet の木と VG の木）

実験条件の履歴：
- order_031：初版．手順は wn_hyponym のみ．評価は，クラスの Synset から細かい正解までの経路の集合による hP，hR，hF．
  比較は6構成（YOLO11x + F，OWLv2@20 + CLIP-B/32 を含む），E1 の F に CLIP-ViT-B16 を含む．
- order_032：ハイブリッド手法（YOLO11x + F）と補助の比較を外し，3手法にした．VG.pdf の語彙・枠・命名の手順を追加し，
  評価を VG.pdf の式(8)に合わせた．
"""
import itertools
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from data import (DATASET_DIR, OUTPUT_DIR, build_tree, caption_nouns, class_synsets, descendants, eval_set, fine_labels,
                  hypernym_distances, load_module, ordered_gt_path, vg_counts, vg_vocab)
from model import TreeNamer, full_chain, truncate

E17_DATA, E17_MODEL, E17_TRAIN = load_module("ex017")
E19_DATA, E19_MODEL, E19_TRAIN = load_module("ex019")
save_json, box_iou, greedy_match, safe_div = E17_TRAIN.save_json, E17_TRAIN.box_iou, E17_TRAIN.greedy_match, E17_TRAIN.safe_div

SEED = 0
SPLITS = ["dev", "test"]
MODE = "softmax"
N_BOOTSTRAP = 1000
N_QUALITATIVE = 6
ALPHA = "AlphaCLIP-ViT-B16_context1.5"
E1_EMBEDDERS = ["CLIP-ViT-B32", ALPHA]
E2_METHODS = {"New": f"OWLv2+{ALPHA}+joint@20", "Old": "DETR+CLIP-ViT-B32+vlm", "YOLO11x": "YOLO11x-closed"}
PROCEDURES = ["coarse", "wn_hyponym", "vg_tree", "vg_flat", "vg_subtree"]
GRID_THETA = [-np.inf, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
GRID_DEPTH = [1, 2, 3, 4, 6, None]                   # None は制限なし
GRID_FLAT_TAU = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GRID_T = [1, 1.5, 2, 3, 5]                           # VG.pdf と同じ温度の候補
GRID_TAU = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]   # VG.pdf と同じく [0.01, 0.9] の12通り
GRIDS = {"coarse": [{}],
         "wn_hyponym": [{"theta_f": t, "max_depth": d} for t, d in itertools.product(GRID_THETA, GRID_DEPTH)],
         "vg_tree": [{"theta_f": t, "max_depth": d} for t, d in itertools.product(GRID_THETA, GRID_DEPTH)],
         "vg_flat": [{"tau": t} for t in GRID_FLAT_TAU],
         "vg_subtree": [{"T": T, "tau": t} for T, t in itertools.product(GRID_T, GRID_TAU)]}


def result_dir(experiment, method, condition, seed=SEED):
    """
    結果の保存先 outputs/ex020_coco_owl_alphaclip_granularity/{実験}/{手法}/{条件}/{seed}/ を返す（作成する）．

    Args:
        experiment (str): 実験名．
        method (str): 手法名．
        condition (str): 条件名．
        seed (int): seed．

    Returns:
        str: ディレクトリのパス．
    """
    d = os.path.join(OUTPUT_DIR, experiment, method, condition, str(seed))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 領域（ボックス）の集合：特徴，WordNet の細分化の出力の列，正解との対応
# ---------------------------------------------------------------------------
def cached_json(path, compute):
    """
    JSON のキャッシュを読み込み，無ければ計算して保存する．

    Args:
        path (str): キャッシュのパス．
        compute (callable): 引数なしで内容を返す関数．

    Returns:
        object: キャッシュの内容．
    """
    if os.path.exists(path):
        return json.load(open(path))
    out = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def kept_mask(det, feats, embedder, setting, thresholds, class_names):
    """
    ex019 の E2 で選んだ設定（予算と θ0）で残すボックスを求める（ex019 の make_records と同じ規則）．

    Args:
        det (dict): 1枚の検出結果（スコアの降順）．
        feats (torch.Tensor): det のボックスの特徴（K×D）．
        embedder: 埋め込みモデル．
        setting (dict): ex019 の E2 の summary.json．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．
        class_names (list[str]): COCO の80クラス名．

    Returns:
        np.ndarray: 残すボックスのインデックス．
    """
    scores = np.asarray(det["scores"], np.float32)
    n = len(scores) if setting["budget"] is None else int((scores >= thresholds[setting["budget"]]).sum())
    if n == 0:
        return np.zeros(0, int)
    conf = embedder.scores(feats[:n], class_names, MODE).max(axis=1)
    s = conf * scores[:n] if setting["score_type"] == "joint" else conf
    return np.where(s >= setting["theta"])[0]


def wn_chains(embedder, feats, path):
    """
    WordNet の直接の下位語による細分化の出力の列（葉まで．キャッシュする）．

    Args:
        embedder: 埋め込みモデル．
        feats (dict[int, torch.Tensor]): 画像ID → 領域の特徴（K×D）．
        path (str): キャッシュのパス．

    Returns:
        dict[str, dict]: 画像ID（文字列）→ {"chains"，"time"}．
    """
    def compute():
        """全ての領域の出力の列を求める．"""
        vocab, out = E19_DATA.coco80_vocab(), {}
        for i, f in tqdm(feats.items(), desc="WordNet chains", leave=False):
            start = time.perf_counter()
            out[str(i)] = {"chains": [full_chain(embedder, f[k:k + 1], vocab, MODE) for k in range(len(f))],
                           "time": time.perf_counter() - start}
        return out
    return cached_json(path, compute)


def gt_regions(pool, coco, splits, emb_name, split):
    """
    正解のボックスの領域の集合（E1_gt）．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        emb_name (str): F の名前．
        split (str): 分割の名前．

    Returns:
        dict: "feats"（画像ID → K×D），"assign"（画像ID → {正解のインデックス: 領域のインデックス}），
              "boxes"（画像ID → K×4），"wn"（WordNet の出力の列）．
    """
    gt = {i: E19_DATA.gt_instances(coco, i) for i in splits[split]}
    feats = E19_TRAIN.cached_features(pool, coco, emb_name, {i: g[0] for i, g in gt.items()}, f"gt_{split}")
    feats = {i: feats[i]["feats"] for i in splits[split]}
    wn = wn_chains(pool.get(emb_name), feats, os.path.join(DATASET_DIR, f"chains_gt_{split}_{emb_name}.json"))
    return {"feats": feats, "assign": {i: {g: g for g in range(len(gt[i][1]))} for i in splits[split]},
            "boxes": {i: np.asarray(gt[i][0]).reshape(-1, 4) for i in splits[split]}, "wn": wn}


def match(coco, image_ids, boxes, det_scores):
    """
    残したボックスを正解に対応付ける（検出スコアの降順，クラス非依存，IoU 0.5 以上．ex017 の metrics_func と同じ）．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．
        boxes (dict[int, np.ndarray]): 画像ID → ボックス（K×4）．
        det_scores (dict[int, np.ndarray]): 画像ID → 検出スコア（K）．

    Returns:
        dict[int, dict[int, int]]: 画像ID → {正解のインデックス: 領域のインデックス}．
    """
    out = {}
    for i in image_ids:
        gt_boxes, _ = E19_DATA.gt_instances(coco, i)
        order = list(np.argsort(-np.asarray(det_scores[i]), kind="stable"))
        out[i] = {g: p for p, g in greedy_match(box_iou(boxes[i], gt_boxes), order).items()}
    return out


def pipeline_regions(pool, coco, splits, dets, thresholds, ex019_method, split):
    """
    検出から命名までの領域の集合（E2_pipeline．ex019 の E2 で選んだ設定で残したボックスのみ）．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): ex019 の load_detections の戻り値．
        thresholds (dict[int, float]): 予算 → OWLv2 の閾値．
        ex019_method (str): ex019 の E2 の手法名．
        split (str): 分割の名前．

    Returns:
        dict: gt_regions と同じキー．
    """
    setting = E19_TRAIN.chosen_setting(ex019_method)
    det_name, emb_name = setting["detector"], setting["embedder"]
    d = dets[(det_name, split)]
    all_feats = E19_TRAIN.cached_features(pool, coco, emb_name, {i: x["boxes"] for i, x in d.items()}, f"{det_name}_{split}")
    embedder = pool.get(emb_name)
    feats, boxes, scores = {}, {}, {}
    for i in splits[split]:
        keep = kept_mask(d[i], all_feats[i]["feats"], embedder, setting, thresholds, coco["class_names"])
        feats[i] = all_feats[i]["feats"][keep]
        boxes[i] = np.asarray(d[i]["boxes"])[keep].reshape(-1, 4)
        scores[i] = np.asarray(d[i]["scores"])[keep]
    wn = wn_chains(embedder, feats, os.path.join(DATASET_DIR, f"chains_pipeline_{split}_{ex019_method}.json"))
    return {"feats": feats, "assign": match(coco, splits[split], boxes, scores), "boxes": boxes, "wn": wn}


def closed_regions(coco, splits, dets, split):
    """
    YOLO11x 単体の領域の集合（ex019 の YOLO11x-closed の θ 以上のボックス．出力は YOLO のクラス）．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        dets (dict): ex019 の load_detections の戻り値．
        split (str): 分割の名前．

    Returns:
        dict: "assign"，"boxes"，"closed"（画像ID → 出力の Synset 名のリスト）．
    """
    theta = E19_TRAIN.chosen_setting(E2_METHODS["YOLO11x"])["theta"]
    synsets = class_synsets()
    boxes, scores, closed = {}, {}, {}
    for i in splits[split]:
        d = dets[("YOLO11x", split)][i]
        keep = np.where(np.asarray(d["scores"]) >= theta)[0]
        boxes[i] = np.asarray(d["boxes"])[keep].reshape(-1, 4)
        scores[i] = np.asarray(d["scores"])[keep]
        closed[i] = [synsets[d["labels"][k]] for k in keep]
    return {"assign": match(coco, splits[split], boxes, scores), "boxes": boxes, "closed": closed}


# ---------------------------------------------------------------------------
# 命名の手順の出力
# ---------------------------------------------------------------------------
def prepare(regions, namer):
    """
    VG の木による手順に必要な量（コサイン類似度，VG の木の出力の列）を領域ごとに前計算する．

    Args:
        regions (dict): gt_regions または pipeline_regions の戻り値（"feats" を持つ）．
        namer (TreeNamer): VG の木の命名器．

    Returns:
        dict: regions に "cos"（画像ID → K×W）と "vg"（画像ID → 出力の列のリスト）を加えたもの．
    """
    regions["cos"] = {i: namer.cosine(f) if len(f) else np.zeros((0, len(namer.word_nodes)), np.float32)
                      for i, f in regions["feats"].items()}
    regions["vg"] = {i: [namer.tree_chain(row) for row in c] for i, c in tqdm(regions["cos"].items(), desc="VG chains", leave=False)}
    return regions


def label_of(chain):
    """
    出力の列から (最後の Synset 名, 表示の語の列) を作る．

    Args:
        chain (list[dict]): 出力の列．

    Returns:
        tuple[str | None, str]: 出力の Synset 名と，" > " でつないだ語．
    """
    return (chain[-1]["synset"], " > ".join(s["word"] for s in chain)) if chain else (None, "")


def node_outputs(namer, nodes):
    """
    VG の木のノード名を (正誤判定の Synset 名, 表示の語) にする．

    Args:
        namer (TreeNamer): VG の木の命名器．
        nodes (list[str | None]): ノード名（None は出力なし）．

    Returns:
        list[tuple[str | None, str]]: 出力．
    """
    tree = namer.tree
    return [(tree["synset"][n], tree["words"][n] or n.removeprefix("frame:")) if n else (None, "") for n in nodes]


def outputs(regions, namer, procedure, params):
    """
    命名の手順と条件で，各領域の出力（Synset 名と表示の語の列）を求める．

    Args:
        regions (dict): prepare または closed_regions の戻り値．
        namer (TreeNamer | None): VG の木の命名器．
        procedure (str): PROCEDURES のいずれか．
        params (dict): 条件（GRIDS の要素）．

    Returns:
        dict[int, list[tuple[str | None, str]]]: 画像ID → 領域ごとの (出力の Synset 名, 表示の語)．出力なしは (None, "")．
    """
    if "closed" in regions:
        return {i: [(s, s.split(".")[0]) for s in v] for i, v in regions["closed"].items()}
    out = {}
    for i in regions["feats"]:
        if procedure == "coarse":
            out[i] = [label_of(c[:1]) for c in regions["wn"][str(i)]["chains"]]
        elif procedure == "wn_hyponym":
            out[i] = [label_of(truncate(c, params["theta_f"], params["max_depth"])) for c in regions["wn"][str(i)]["chains"]]
        elif procedure == "vg_tree":
            out[i] = [label_of(truncate(c, params["theta_f"], params["max_depth"])) for c in regions["vg"][i]]
        elif len(regions["cos"][i]) == 0:
            out[i] = []
        elif procedure == "vg_flat":
            out[i] = node_outputs(namer, namer.flat(regions["cos"][i], params["tau"]))
        else:
            out[i] = node_outputs(namer, namer.subtree_mass(regions["cos"][i], params["T"], params["tau"]))
    return out


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def fine_instances(coco, image_ids, nouns):
    """
    正解インスタンスと，その細かい正解（キャプション）を列挙する．

    Args:
        coco (dict): load_coco の戻り値．
        image_ids (list[int]): 画像ID．
        nouns (dict[int, list[str]]): caption_nouns の戻り値．

    Returns:
        dict[int, list[tuple[int, str, list[str]]]]: 画像ID → (正解のインデックス, クラス名, 細かい正解の Synset 名のリスト)．
    """
    out = {}
    for i in image_ids:
        _, labels = E19_DATA.gt_instances(coco, i)
        out[i] = [(g, c, fine_labels(c, nouns[i])) for g, c in enumerate(labels)]
    return out


def instance_score(pred, fines, class_synset):
    """
    1つのインスタンスの評価（VG.pdf の式(8)．細かい正解が複数ある場合は |P∩G| が最大，同数なら |G| が最小のもの）．

    Args:
        pred (str | None): 出力の Synset 名（None は出力なし）．
        fines (list[str]): 細かい正解の Synset 名（1つ以上）．
        class_synset (str): 正解クラスの Synset 名．

    Returns:
        dict: "inter"，"n_pred"，"n_gt"，"fine"，"status"（"none"，"exact"，"finer"：正しく COCO のクラスより細かい，
              "coarse_correct"：正しいが COCO のクラス以上の粒度，"wrong"）．
    """
    P = eval_set(pred) if pred else frozenset()
    fine = max(fines, key=lambda f: (len(P & eval_set(f)), -len(eval_set(f))))
    G = eval_set(fine)
    if pred is None:
        status = "none"
    elif pred == fine:
        status = "exact"
    elif pred in G:
        status = "finer" if pred in descendants(class_synset) else "coarse_correct"
    else:
        status = "wrong"
    return {"inter": len(P & G), "n_pred": len(P), "n_gt": len(G), "fine": fine, "status": status}


def wn_depth(synset_name):
    """
    Synset の WordNet の根からの最短の深さ．

    Args:
        synset_name (str): Synset 名．

    Returns:
        int: 深さ．
    """
    return wn.synset(synset_name).min_depth()


def evaluate(instances, regions, outs, keep_rows=False):
    """
    出力の階層的評価（細かい正解を持つインスタンス）と，クラスの正解率（全ての正解インスタンス）を求める．

    Args:
        instances (dict): fine_instances の戻り値．
        regions (dict): 領域の集合（"assign" を持つ）．
        outs (dict): outputs の戻り値．
        keep_rows (bool): インスタンスごとの結果を返すかどうか．

    Returns:
        tuple[dict, np.ndarray, list[dict]]: 指標，画像ごとの計数（N×3：|P∩G|，|P|，|G|），インスタンスごとの結果．
    """
    synsets = class_synsets()
    counts, rows, status = [], [], {k: 0 for k in ("none", "exact", "finer", "coarse_correct", "wrong")}
    n_all, n_class_ok, depths = 0, 0, []
    for i, items in instances.items():
        c = np.zeros(3)
        for g, cls, fines in items:
            k = regions["assign"][i].get(g)
            pred, label = outs[i][k] if k is not None else (None, "")
            n_all += 1
            n_class_ok += bool(pred) and (pred == synsets[cls] or pred in descendants(synsets[cls]))
            if not fines:
                continue
            r = instance_score(pred, fines, synsets[cls])
            c += [r["inter"], r["n_pred"], r["n_gt"]]
            status[r["status"]] += 1
            if pred:
                depths.append(wn_depth(pred))
            if keep_rows:
                rows.append({"image_id": i, "gt_index": g, "class": cls, "fine": r["fine"], "status": r["status"],
                             "output": label, "output_synset": pred})
        counts.append(c)
    counts = np.array(counts)
    n = sum(status.values())
    m = hierarchical_metrics(counts.sum(0))
    m.update({"n_fine_instances": n, "naming_acc": safe_div(status["exact"] + status["finer"] + status["coarse_correct"], n),
              "exact_acc": safe_div(status["exact"], n), "finer_acc": safe_div(status["exact"] + status["finer"], n),
              **{f"rate_{k}": safe_div(v, n) for k, v in status.items()},
              "mean_output_wn_depth": float(np.mean(depths)) if depths else 0.0,
              "n_gt_all": n_all, "class_acc": safe_div(n_class_ok, n_all)})
    return m, counts, rows


def hierarchical_metrics(s):
    """
    計数の合計から hP，hR，hF を求める．

    Args:
        s (np.ndarray): (|P∩G|，|P|，|G|) の合計（長さ3）．

    Returns:
        dict: "hP"，"hR"，"hF"．
    """
    hp, hr = safe_div(s[0], s[1]), safe_div(s[0], s[2])
    return {"hP": hp, "hR": hr, "hF": safe_div(2 * hp * hr, hp + hr) if s[0] else 0.0}


def bootstrap_hf(counts, n=N_BOOTSTRAP, seed=SEED):
    """
    画像単位のブートストラップで hF の95%信頼区間を求める．

    Args:
        counts (np.ndarray): evaluate の画像ごとの計数（N×3）．
        n (int): 再標本化の回数．
        seed (int): 乱数シード．

    Returns:
        list[float]: [下限, 上限]．
    """
    rng = np.random.default_rng(seed)
    v = [hierarchical_metrics(counts[rng.integers(0, len(counts), len(counts))].sum(0))["hF"] for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_bootstrap_hf(counts_a, counts_b, n=N_BOOTSTRAP, seed=SEED):
    """
    画像単位の対応のあるブートストラップで，hF の差（b − a）の95%信頼区間と両側 p 値を求める．

    Args:
        counts_a (np.ndarray): 手法 a の計数（N×3）．
        counts_b (np.ndarray): 手法 b の計数（N×3，同じ画像の順序）．
        n (int): 再標本化の回数．
        seed (int): 乱数シード．

    Returns:
        dict: "diff"，"ci95"，"p_value"．
    """
    rng = np.random.default_rng(seed)
    hf = lambda c: hierarchical_metrics(c.sum(0))["hF"]   # noqa: E731
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(counts_a), len(counts_a))
        diffs.append(hf(counts_b[idx]) - hf(counts_a[idx]))
    diffs = np.array(diffs)
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": float(hf(counts_b) - hf(counts_a)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_value": float(max(p, 1 / n))}


def params_name(params):
    """
    条件の辞書を表の列の文字列にする．

    Args:
        params (dict): 条件．

    Returns:
        str: 例 "theta_f=0.2,max_depth=2"（条件が無い場合は "-"）．
    """
    return ",".join(f"{k}={v}" for k, v in params.items()) or "-"


def evaluate_procedure(experiment, method, procedure, config, instances, regions, namer):
    """
    1つの (手法, 命名の手順) について，dev で条件を選び，dev と test の結果と test の格子を保存する．

    Args:
        experiment (str): "E1_gt" または "E2_pipeline"．
        method (str): 手法名．
        procedure (str): 命名の手順．
        config (dict): 保存する条件．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        regions (dict[str, dict]): 分割 → 領域の集合．
        namer (TreeNamer | None): VG の木の命名器．

    Returns:
        dict: summary（"params"，"dev"，"test" の指標）と画像ごとの計数（"counts"：分割 → N×3）．
    """
    name = f"{method}+{procedure}"
    base = os.path.join(OUTPUT_DIR, experiment, name)
    path = os.path.join(base, "summary.json")
    if os.path.exists(path):
        summary = json.load(open(path))
        summary["counts"] = {s: np.load(os.path.join(result_dir(experiment, name, f"selected_{s}"), "counts.npy")) for s in SPLITS}
        return summary
    grids = {}
    for split in SPLITS:
        grids[split] = pd.DataFrame([{"params": params_name(p), **evaluate(instances[split], regions[split],
                                                                          outputs(regions[split], namer, procedure, p))[0]}
                                     for p in GRIDS[procedure]])
        os.makedirs(base, exist_ok=True)
        grids[split].to_csv(os.path.join(base, f"grid_{split}.csv"), index=False)
    best = GRIDS[procedure][int(grids["dev"].hF.idxmax())]
    summary = {"method": method, "procedure": procedure, **config, "params": best}
    counts = {}
    for split in SPLITS:
        m, c, rows = evaluate(instances[split], regions[split], outputs(regions[split], namer, procedure, best), keep_rows=True)
        m["hF_ci95"] = bootstrap_hf(c)
        d = result_dir(experiment, name, f"selected_{split}")
        save_json(m, os.path.join(d, "metrics.json"))
        save_json({**config, "procedure": procedure, "params": best, "selection": "dev hF", "split": split, "mode": MODE,
                   "fine_gt": "COCO captions (spaCy nouns, WordNet first sense)",
                   "eval": "VG.pdf Eq.(8): self + hypernyms with depth >= 4 (+ COCO class synsets)", "seed": SEED},
                  os.path.join(d, "config.json"))
        np.save(os.path.join(d, "counts.npy"), c)
        pd.DataFrame(rows).to_csv(os.path.join(d, "instances.csv"), index=False)
        summary[split] = m
        counts[split] = c
    save_json(summary, path)
    t = summary["test"]
    print(experiment, name, params_name(best), {k: round(t[k], 3) for k in ("hP", "hR", "hF", "naming_acc", "finer_acc", "class_acc")})
    return summary | {"counts": counts}


# ---------------------------------------------------------------------------
# 実験
# ---------------------------------------------------------------------------
def coverage_of(fines, vocab_ok):
    """
    1つのインスタンスの語彙の被覆（正しい語までの上位方向の最短の距離と，1語を選ぶ場合の hR の上限）．

    Args:
        fines (list[str]): 細かい正解の Synset 名．
        vocab_ok (callable): Synset 名 → 語彙に含まれるかどうか．

    Returns:
        tuple[int, float]: 距離（正しい語が無い場合は 99）と hR の上限．
    """
    best_r, best_hit = 99, 0.0
    for f in fines:
        G = eval_set(f)
        for a, d in hypernym_distances(f).items():
            if a in G and vocab_ok(a):
                best_r = min(best_r, d)
                best_hit = max(best_hit, len(eval_set(a) & G) / len(G))
    return best_r, best_hit


def run_e0(tree, instances, split="test"):
    """
    E0_coverage：VLM を用いない語彙の被覆率（正解の Synset から上位方向に距離 r 以内に正しい語がある割合）と，
    語彙から1語を選ぶ場合の hR の上限（VG.pdf 5.2節，表1）．

    Args:
        tree (dict): data.build_tree の戻り値．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        split (str): 分割の名前．

    Returns:
        pd.DataFrame: 語彙ごとの語数と r = 0，r ≤ 1，r ≤ 2，全て，hR の上限．
    """
    synsets = class_synsets()
    coco = set(synsets.values())
    vg = {s for s in tree["synset"].values() if s}
    wn_all = set().union(*[descendants(s) for s in coco]) | coco
    vocabs = {"COCO 80 classes": (len(coco), coco.__contains__),
              "WordNet (COCO classes + all hyponyms)": (len(wn_all), wn_all.__contains__),
              "VG vocab (n_min=10) + COCO 80": (len(vg), vg.__contains__)}
    rows = []
    for name, (size, ok) in vocabs.items():
        res = [coverage_of(f, ok) for items in instances[split].values() for _, _, f in items if f]
        r = np.array([x[0] for x in res])
        rows.append({"vocabulary": name, "n_words": size, "coverage_r0": float((r == 0).mean()), "coverage_r1": float((r <= 1).mean()),
                     "coverage_r2": float((r <= 2).mean()), "coverage_any": float((r < 99).mean()),
                     "hR_upper": float(np.mean([x[1] for x in res]))})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUTPUT_DIR, "E0_coverage"), exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "E0_coverage", f"coverage_{split}.csv"), index=False)
    return df


def run_e1(pool, coco, splits, instances, tree):
    """
    E1_gt：正解のボックス × F × 命名の手順．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        tree (dict): data.build_tree の戻り値．

    Returns:
        dict[tuple[str, str], dict]: (F, 手順) → evaluate_procedure の戻り値．
    """
    out = {}
    for emb_name in E1_EMBEDDERS:
        namer = TreeNamer(pool.get(emb_name), tree, [s for _, s in E19_DATA.coco80_vocab()])
        regions = {s: prepare(gt_regions(pool, coco, splits, emb_name, s), namer) for s in SPLITS}
        for proc in PROCEDURES:
            out[(emb_name, proc)] = evaluate_procedure("E1_gt", emb_name, proc, {"embedder": emb_name, "boxes": "COCO ground truth"},
                                                       instances, regions, namer)
    return out


def run_e2(pool, coco, splits, instances, tree):
    """
    E2_pipeline：3手法（New，Old，YOLO11x）× 命名の手順．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        tree (dict): data.build_tree の戻り値．

    Returns:
        tuple[dict, dict]: (手法, 手順) → evaluate_procedure の戻り値，手法 → (分割 → 領域の集合, 命名器)．
    """
    dets, thresholds = E19_TRAIN.load_detections(coco, splits)
    out, regions_all = {}, {}
    for method, ex019_method in E2_METHODS.items():
        if method == "YOLO11x":
            regions = {s: closed_regions(coco, splits, dets, s) for s in SPLITS}
            namer, procs, config = None, ["coarse"], {"ex019_method": ex019_method, "detector": "YOLO11x", "embedder": None}
        else:
            setting = E19_TRAIN.chosen_setting(ex019_method)
            namer = TreeNamer(pool.get(setting["embedder"]), tree, [s for _, s in E19_DATA.coco80_vocab()])
            regions = {s: prepare(pipeline_regions(pool, coco, splits, dets, thresholds, ex019_method, s), namer) for s in SPLITS}
            procs = PROCEDURES
            config = {"ex019_method": ex019_method, "detector": setting["detector"], "embedder": setting["embedder"],
                      "score_type": setting["score_type"], "budget": setting["budget"], "theta0": setting["theta"]}
        for proc in procs:
            out[(method, proc)] = evaluate_procedure("E2_pipeline", method, proc, config, instances, regions, namer)
        regions_all[method] = (regions, namer)
    return out, regions_all


def step_rows_wordnet(embedder, feat, root, fine):
    """
    WordNet の木で，正解の経路の各段の候補語彙での正解の子の順位を求める．

    Args:
        embedder: 埋め込みモデル．
        feat (torch.Tensor): 領域の特徴（1×D）．
        root (str): クラスの Synset 名．
        fine (str): 細かい正解の Synset 名．

    Returns:
        list[dict]: 段ごとの "step"，"n_candidates"，"rank"（候補に無い場合は None）．
    """
    path = ordered_gt_path(root, fine)
    rows = []
    for k, (u, v) in enumerate(zip(path[:-1], path[1:]), start=1):
        vocab = E17_DATA.hyponym_vocab(u)
        synsets = [s for _, s in vocab]
        rank = None
        if v in synsets:
            s = embedder.scores(feat, [w for w, _ in vocab], MODE)[0]
            rank = int((s > s[synsets.index(v)]).sum()) + 1
        rows.append({"step": k, "n_candidates": len(vocab), "rank": rank})
    return rows


def vg_path(tree, root, fine):
    """
    VG の木で，クラスのノードから細かい正解（またはその語彙にある最も近い上位語）までの経路を求める．

    Args:
        tree (dict): data.build_tree の戻り値．
        root (str): クラスの Synset 名．
        fine (str): 細かい正解の Synset 名．

    Returns:
        list[str] | None: 上から順のノード名（クラスのノードから到達できない場合は None）．
    """
    for _, node in sorted((d, a) for a, d in hypernym_distances(fine).items() if a in tree["parent"] and a != root):
        path = [node]
        while path[-1] not in (root, "root"):
            path.append(tree["parent"][path[-1]])
        if path[-1] == root:
            return path[::-1]
    return None


def step_rows_vg(namer, cos_row, root, fine):
    """
    VG の木で，正解の経路の各段の候補（木の子）での正解の子の順位を求める．

    Args:
        namer (TreeNamer): VG の木の命名器．
        cos_row (np.ndarray): 領域のコサイン類似度（長さW）．
        root (str): クラスの Synset 名．
        fine (str): 細かい正解の Synset 名．

    Returns:
        list[dict]: 段ごとの "step"，"n_candidates"，"rank"．経路が無い場合は "rank" が None の1行．
    """
    path = vg_path(namer.tree, root, fine)
    if path is None:
        return [{"step": 1, "n_candidates": len(namer.children.get(root, [])), "rank": None}]
    rows = []
    for k, (u, v) in enumerate(zip(path[:-1], path[1:]), start=1):
        kids = [c for c in namer.children.get(u, []) if c in namer.index]
        s = cos_row[[namer.index[c] for c in kids]]
        rows.append({"step": k, "n_candidates": len(kids), "rank": int((s > s[kids.index(v)]).sum()) + 1})
    return rows


def run_e3(pool, coco, splits, instances, tree, split="test"):
    """
    E3_step：正解のボックスで，正解の経路を1段ずつ与えた場合の各段の正解率（WordNet の木と VG の木）．

    Args:
        pool: ex019 の EmbedderPool．
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        tree (dict): data.build_tree の戻り値．
        split (str): 分割の名前．

    Returns:
        pd.DataFrame: F × 木 × 群 × 段の集計．
    """
    path = os.path.join(OUTPUT_DIR, "E3_step", f"summary_{split}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    roots = class_synsets()
    rows = []
    for emb_name in E1_EMBEDDERS:
        embedder = pool.get(emb_name)
        namer = TreeNamer(embedder, tree, [s for _, s in E19_DATA.coco80_vocab()])
        regions = gt_regions(pool, coco, splits, emb_name, split)
        for i, items in tqdm(instances[split].items(), desc=f"E3 {emb_name}", leave=False):
            for g, cls, fines in items:
                if not fines:
                    continue
                fine = min(fines, key=lambda f: len(ordered_gt_path(roots[cls], f)))
                feat = regions["feats"][i][g:g + 1]
                base = {"embedder": emb_name, "image_id": i, "gt_index": g, "class": cls, "fine": fine}
                rows += [base | {"tree": "WordNet", **r} for r in step_rows_wordnet(embedder, feat, roots[cls], fine)]
                rows += [base | {"tree": "VG", **r} for r in step_rows_vg(namer, namer.cosine(feat)[0], roots[cls], fine)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "E3_step", f"steps_{split}.csv"), index=False)
    df["group"] = np.where(df["class"] == "person", "person", "non-person")
    df["top1"], df["top5"] = df["rank"] == 1, df["rank"] <= 5
    df["missing"] = df["rank"].isna()
    df["step_c"] = df.step.clip(upper=4)
    out = df.groupby(["embedder", "tree", "group", "step_c"]).agg(
        n=("top1", "size"), top1=("top1", "mean"), top5=("top5", "mean"), missing=("missing", "mean"),
        median_rank=("rank", "median"), mean_candidates=("n_candidates", "mean")).reset_index()
    out.to_csv(path, index=False)
    return out


def summary_tables(e1, e2):
    """
    test の要約表と，hF の差の検定の表を保存する．

    Args:
        e1 (dict): run_e1 の戻り値．
        e2 (dict): run_e2 の戻り値．

    Returns:
        pd.DataFrame: 要約表．
    """
    keys = ["hP", "hR", "hF", "hF_ci95", "naming_acc", "exact_acc", "finer_acc", "rate_wrong", "rate_none", "class_acc",
            "mean_output_wn_depth"]
    rows = [{"experiment": exp, "method": m, "procedure": p, "params": params_name(r["params"]), **{k: r["test"][k] for k in keys}}
            for exp, res in (("E1_gt", e1), ("E2_pipeline", e2)) for (m, p), r in res.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "summary_test.csv"), index=False)
    tests = []
    for p in PROCEDURES:
        for ref in [("Old", p), ("YOLO11x", "coarse"), ("New", "wn_hyponym")]:
            if ref != ("New", p):
                t = paired_bootstrap_hf(e2[ref]["counts"]["test"], e2[("New", p)]["counts"]["test"])
                tests.append({"experiment": "E2_pipeline", "new": f"New+{p}", "baseline": f"{ref[0]}+{ref[1]}", **t})
        t = paired_bootstrap_hf(e1[("CLIP-ViT-B32", p)]["counts"]["test"], e1[(ALPHA, p)]["counts"]["test"])
        tests.append({"experiment": "E1_gt", "new": f"Alpha+{p}", "baseline": f"CLIP-B32+{p}", **t})
    pd.DataFrame(tests).to_csv(os.path.join(OUTPUT_DIR, "paired_bootstrap_hf_test.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def plot_tradeoff(e1, e2, filename="fig_granularity_tradeoff.png"):
    """
    条件を変えた場合の test の hR と hP（(a) 正解ボックス + Alpha-CLIP，(b) New，(c) Old）．

    Args:
        e1 (dict): run_e1 の戻り値．
        e2 (dict): run_e2 の戻り値．
        filename (str): 保存するファイル名．
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [("E1_gt", ALPHA, e1, "(a) GT boxes + Alpha-CLIP"), ("E2_pipeline", "New", e2, "(b) New: OWLv2 + Alpha-CLIP"),
              ("E2_pipeline", "Old", e2, "(c) Old: DETR + CLIP-B/32")]
    for ax, (exp, method, res, title) in zip(axes, panels):
        for proc in PROCEDURES[1:]:
            g = pd.read_csv(os.path.join(OUTPUT_DIR, exp, f"{method}+{proc}", "grid_test.csv"))
            if proc in ("wn_hyponym", "vg_tree"):
                g = g[g.params.str.endswith("max_depth=None")]
            elif proc == "vg_subtree":
                g = g[g.params.str.startswith(f"T={res[(method, proc)]['params']['T']},")]
            ax.plot(g.hR, g.hP, marker="o", ms=3, label=proc)
            sel = res[(method, proc)]["test"]
            ax.scatter([sel["hR"]], [sel["hP"]], marker="*", s=150, zorder=5)
        c = res[(method, "coarse")]["test"]
        ax.scatter([c["hR"]], [c["hP"]], marker="s", s=60, color="k", label="coarse (COCO class)", zorder=6)
        if exp == "E2_pipeline":
            y = e2[("YOLO11x", "coarse")]["test"]
            ax.scatter([y["hR"]], [y["hP"]], marker="^", s=60, color="gray", label="YOLO11x (closed)", zorder=6)
        ax.set_xlabel("Hierarchical recall hR")
        ax.set_ylabel("Hierarchical precision hP")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)


def pick_examples(splits, instances, n=N_QUALITATIVE):
    """
    比較画像に用いる test の画像を選ぶ（細かい正解を持つインスタンスが1〜4件の画像を，画像IDの順に等間隔に n 枚）．

    Args:
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        n (int): 枚数．

    Returns:
        list[int]: 画像ID．
    """
    ids = [i for i in sorted(splits["test"]) if 1 <= sum(bool(f) for _, _, f in instances["test"][i]) <= 4]
    return ids[::max(len(ids) // n, 1)][:n]


def draw_examples(coco, splits, instances, e2, regions_all, filename="fig_granularity_examples.png"):
    """
    YOLO11x 単体，Old（dev の hF が最良の手順），New（wn_hyponym，vg_tree，vg_subtree）の出力を並べた比較画像を描く．

    Args:
        coco (dict): load_coco の戻り値．
        splits (dict[str, list[int]]): 分割．
        instances (dict[str, dict]): 分割 → fine_instances の戻り値．
        e2 (dict): run_e2 の戻り値．
        regions_all (dict): run_e2 の領域の集合と命名器．
        filename (str): 保存するファイル名．
    """
    ids = pick_examples(splits, instances)
    old_best = max(PROCEDURES, key=lambda p: e2[("Old", p)]["dev"]["hF"])
    columns = [("YOLO11x (closed)", "YOLO11x", "coarse"), (f"Old: {old_best}", "Old", old_best),
               ("New: wn_hyponym", "New", "wn_hyponym"), ("New: vg_tree", "New", "vg_tree"), ("New: vg_subtree", "New", "vg_subtree")]
    outs = {(m, p): outputs(regions_all[m][0]["test"], regions_all[m][1], p, e2[(m, p)]["params"]) for _, m, p in columns}
    fig, axes = plt.subplots(len(ids), len(columns), figsize=(4.5 * len(columns), 3.8 * len(ids)))
    for r, i in enumerate(ids):
        image = E19_DATA.load_image(coco, i)
        fines = sorted({f.split(".")[0] for _, _, fs in instances["test"][i] for f in fs})
        for c, (title, m, p) in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(image)
            ax.axis("off")
            for box, (_, label) in zip(regions_all[m][0]["test"]["boxes"][i], outs[(m, p)][i]):
                x0, y0, x1, y1 = box
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", lw=1.5))
                if label:
                    ax.text(x0, y0, label, fontsize=7, bbox={"facecolor": "cyan", "alpha": 0.7, "pad": 1})
            ax.set_title(f"{title}\ncaption fine GT: {', '.join(fines)}" if c == 0 else title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=110)
    plt.close(fig)


def train():
    """
    ex020 の全ての実験（E0〜E3，要約，図）を実行する．
    """
    E17_MODEL.set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coco = E19_DATA.load_coco()
    splits = E19_DATA.load_splits(coco)
    nouns = {s: caption_nouns(coco, splits[s], s) for s in SPLITS}
    instances = {s: fine_instances(coco, splits[s], nouns[s]) for s in SPLITS}
    tree = build_tree()
    vg = vg_counts()
    stats = {"vg_images": vg["n_images"], "vg_excluded_coco_val": vg["n_excluded"], "vg_vocab": len(vg_vocab()),
             "tree_nodes": len(tree["nodes"]), "tree_frames": sum(n.startswith("frame:") for n in tree["nodes"]),
             "tree_children_person": sum(p == "person.n.01" for p in tree["parent"].values()),
             **{f"n_fine_instances_{s}": sum(bool(f) for v in instances[s].values() for _, _, f in v) for s in SPLITS},
             **{f"n_gt_{s}": sum(len(v) for v in instances[s].values()) for s in SPLITS},
             "tree_frame_sizes": pd.Series(tree["frame"]).value_counts().to_dict()}
    save_json(stats, os.path.join(OUTPUT_DIR, "stats.json"))
    print("統計", {k: v for k, v in stats.items() if not isinstance(v, dict)})
    print(run_e0(tree, instances).round(3).to_string())
    pool = E19_TRAIN.EmbedderPool()
    e1 = run_e1(pool, coco, splits, instances, tree)
    e2, regions_all = run_e2(pool, coco, splits, instances, tree)
    df = summary_tables(e1, e2)
    print(df.drop(columns=["hF_ci95"]).round(3).to_string())
    print(run_e3(pool, coco, splits, instances, tree).round(3).to_string())
    plot_tradeoff(e1, e2)
    draw_examples(coco, splits, instances, e2, regions_all)


if __name__ == "__main__":
    train()
