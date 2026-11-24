"""
ex015cの分析：greedy の経路が正解の枝（GTのSynsetの上位語・自身・下位語）から外れる段階の特定．

経路は根（physical_entity.n.01）から出力した語までのSynsetの列である．経路上で最初に正解の枝から外れたSynsetを「分岐」とし，
その1つ前の（正解の枝上の）Synsetを「分岐元」とする．分岐元の下位語のうち正解の枝にあるものを「正しい子」とし，
選んだ子との確率の差と，子の中での正しい子の順位を求める．
"""
import functools

import numpy as np
from nltk.corpus import wordnet as wn

# 全ての経路が通る上位の語（論文の図では省略する）．分岐元がこれらの場合を「最上位の分類での誤り」（深さ0）とする
TOP_COMMON = ("physical_entity.n.01", "object.n.01", "whole.n.02")
STATUSES = ["fine", "coarse", "diverged", "unreachable"]


@functools.lru_cache(maxsize=None)
def ancestors(synset_name):
    """
    Synset自身と全ての上位語の集合を返す（キャッシュする）．

    Args:
        synset_name (str): Synset名．

    Returns:
        frozenset[str]: Synset名の集合．
    """
    s = wn.synset(synset_name)
    return frozenset({synset_name} | {a.name() for a in s.closure(lambda t: t.hypernyms())})


def on_branch(node, gt):
    """
    Synsetが正解の枝（GTの上位語・自身・下位語）にあるかを返す．

    Args:
        node (str): Synset名．
        gt (str): GTのSynset名．

    Returns:
        bool: 正解の枝にあれば True．
    """
    return node in ancestors(gt) or gt in ancestors(node)


def word(synset_name):
    """
    Synsetの代表の語を返す．

    Args:
        synset_name (str): Synset名．

    Returns:
        str: 語（"_" は空白に置き換える）．
    """
    return wn.synset(synset_name).lemma_names()[0].replace("_", " ")


def divergence(hier, node_p, path, gt):
    """
    1つの領域について，経路が正解の枝から外れた段階と，その段階での確率を求める．

    Args:
        hier (ex015 model.Hierarchy): 階層．
        node_p (np.ndarray): その領域のSynsetの確率（Synset数）．
        path (list[str]): 根から出力した語までのSynset名の列．
        gt (str): GTのSynset名．

    Returns:
        dict: status（fine：GT自身か下位語，coarse：GTの上位語で停止，diverged：正解の枝から外れた），
            分岐の場合は step（根からの段数，1が根の子），parent，chosen，correct（正しい子，無ければ None），
            p_chosen，p_correct，rank（分岐元の子の中での正しい子の順位），n_children，wrong_len（外れた後の語数），top（深さ0か）．
    """
    if not on_branch(path[0], gt):
        # GTのSynsetが根（physical_entity）の下に無い（poster，tag 等はWordNetで抽象物）．語彙の階層では到達できない
        return {"status": "unreachable"}
    k = next((i for i, n in enumerate(path) if not on_branch(n, gt)), None)
    if k is None:
        return {"status": "fine" if gt in ancestors(path[-1]) else "coarse"}
    parent, chosen = path[k - 1], path[k]
    kids = np.where(hier.child[hier.index[parent]].cpu().numpy())[0]
    kid_p = node_p[kids]
    good = [c for c in kids if on_branch(hier.nodes[c], gt)]
    out = {"status": "diverged", "step": k, "parent": parent, "chosen": chosen, "p_chosen": float(node_p[hier.index[chosen]]),
           "n_children": len(kids), "wrong_len": len(path) - k, "top": parent in TOP_COMMON,
           "correct": None, "p_correct": 0.0, "rank": None}
    if good:
        c = max(good, key=lambda x: node_p[x])
        out |= {"correct": hier.nodes[c], "p_correct": float(node_p[c]), "rank": int((kid_p > node_p[c]).sum()) + 1}
    return out


def flat_named(hier, p_leaf, gt):
    """
    確率が最大の1語（flat）が正解の枝にあるかを返す．

    Args:
        hier (ex015 model.Hierarchy): 階層．
        p_leaf (np.ndarray): 語彙上の確率（V）．
        gt (str): GTのSynset名．

    Returns:
        bool: 正解の枝にあれば True．
    """
    node = hier.nodes[int(hier.node_of_word[int(p_leaf.argmax())])]
    return on_branch(node, gt)


def summarize(rows, n_pairs=30):
    """
    領域ごとの分析結果を集計する．

    Args:
        rows (list[dict]): divergence の戻り値に gt，flat_named，in_vocab，（分岐の場合）n_chosen，n_correct（枝の下の語数）を加えたもの．
        n_pairs (int): 保存する（正しい子，選んだ子）の組の数．

    Returns:
        dict: 割合，分岐元の分布，組の頻度，順位と確率の統計，選んだ枝の方が語数が多い割合，
            深さ0の分岐のうちWordNetの構造（part と whole，object と matter）による割合．
    """
    n = len(rows)
    div = [r for r in rows if r["status"] == "diverged"]
    top = [r for r in div if r["top"]]
    out = {"n": n} | {f"rate_{s}": sum(r["status"] == s for r in rows) / max(n, 1) for s in STATUSES}
    out["rate_top"] = len(top) / max(n, 1)
    out["diverged_top_share"] = len(top) / max(len(div), 1)
    out["parent_counts"] = _counts([word(r["parent"]) for r in div])
    out["pair_counts_top"] = dict(list(_counts([f"{word(r['correct']) if r['correct'] else '(無し)'} → {word(r['chosen'])}"
                                                for r in top]).items())[:n_pairs])
    out["pair_counts_all"] = dict(list(_counts([f"{word(r['correct']) if r['correct'] else '(無し)'} → {word(r['chosen'])}"
                                                for r in div]).items())[:n_pairs])
    ranked = [r for r in div if r["rank"] is not None]
    out["no_correct_child"] = sum(r["rank"] is None for r in div) / max(len(div), 1)
    out["rank_counts"] = _counts([str(min(r["rank"], 5)) if r["rank"] < 5 else "5+" for r in ranked])
    for name, sel in [("all", div), ("top", top)]:
        if sel:
            out[f"p_chosen_mean_{name}"] = float(np.mean([r["p_chosen"] for r in sel]))
            out[f"p_correct_mean_{name}"] = float(np.mean([r["p_correct"] for r in sel]))
            out[f"wrong_len_mean_{name}"] = float(np.mean([r["wrong_len"] for r in sel]))
            out[f"flat_named_{name}"] = float(np.mean([r["flat_named"] for r in sel]))
            out[f"close_{name}"] = float(np.mean([r["p_chosen"] - r["p_correct"] < 0.1 for r in sel]))
    sized = [r for r in div if r.get("correct")]
    out["chosen_subtree_larger"] = float(np.mean([r["n_chosen"] > r["n_correct"] for r in sized])) if sized else float("nan")
    out["subtree_ratio_median"] = (float(np.median([r["n_chosen"] / max(r["n_correct"], 1) for r in sized]))
                                   if sized else float("nan"))
    structural = {("part.n.02", "whole.n.02"), ("part.n.03", "whole.n.02"), ("object.n.01", "matter.n.03"),
                  ("matter.n.03", "object.n.01")}
    out["top_structural_share"] = (float(np.mean([(r["correct"], r["chosen"]) in structural for r in top]))
                                   if top else float("nan"))
    out["flat_named_all_regions"] = float(np.mean([r["flat_named"] for r in rows])) if rows else float("nan")
    out["in_vocab"] = float(np.mean([r["in_vocab"] for r in rows])) if rows else float("nan")
    return out


def _counts(values):
    """
    値の出現回数を多い順に返す．

    Args:
        values (list[str]): 値．

    Returns:
        dict[str, int]: 値 → 回数（多い順）．
    """
    u, c = np.unique(values, return_counts=True) if values else (np.array([]), np.array([]))
    order = np.argsort(-c, kind="stable")
    return {str(u[i]): int(c[i]) for i in order}


def step_table(hier, node_p, path, k=3):
    """
    経路の各段階で，その時点のSynsetの下位語のうち確率が上位 k 件のものを並べる（最後の語の次の段階も含む）．

    Args:
        hier (ex015 model.Hierarchy): 階層．
        node_p (np.ndarray): その領域のSynsetの確率（Synset数）．
        path (list[str]): 根から出力した語までのSynset名の列．
        k (int): 並べる件数．

    Returns:
        list[dict]: 段階ごとの {"parent"，"p_parent"，"top"（[(語，確率)]）}．
    """
    out = []
    for n in path:
        kids = np.where(hier.child[hier.index[n]].cpu().numpy())[0]
        order = kids[np.argsort(-node_p[kids])][:k]
        out.append({"parent": word(n), "p_parent": float(node_p[hier.index[n]]),
                    "top": [(word(hier.nodes[c]), float(node_p[c])) for c in order]})
    return out
