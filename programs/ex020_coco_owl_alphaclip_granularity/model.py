"""
ex020：粒度可変アノテーション（深さ0の命名と WordNet の下位語による細分化），粒度の変更（order_031），
VG.pdf の語彙の木による命名の手順（order_032．VLM は SigLIP ではなく各手法の F）．

1. 深さ0：COCO の80クラスの語彙から，F の確信度（softmax）が最大の語を選ぶ．
2. 細分化：選んだ語の Synset の直接の下位語の集合を語彙とし，確信度が最大の語を選ぶ処理を繰り返す（ex017 の granular_annotation）．
3. 粒度の変更：細分化の各段階の確信度が θ_f 未満になるか，最大の深さ D に達した段階で止める．

θ_f と D による停止は，θ = −∞ で葉まで求めた出力の列の先頭部分を取り出すことと等しい（各段階の選択は θ によらないため）．
そこで，各ボックスの出力の列を1回だけ求めて保存し，θ_f と D を変える評価は truncate で行う．
"""
import numpy as np
import scipy.sparse as sp

from data import load_module

MAX_DEPTH = 12          # 細分化の最大の深さ（WordNet の person から葉までの段数より十分大きい）


def full_chain(embedder, image_feat, vocab, mode="softmax", max_depth=MAX_DEPTH):
    """
    θ = −∞ で葉（または MAX_DEPTH）まで細分化した出力の列を求める（ex017 の granular_annotation）．

    Args:
        embedder: 埋め込みモデル（scores を持つ）．
        image_feat (torch.Tensor): 領域の特徴（1×D）．
        vocab (list[tuple[str, str]]): 深さ0の語彙（単語, Synset名）．
        mode (str): スコアの種類．
        max_depth (int): 最大の深さ．

    Returns:
        list[dict]: 深さの浅い順の出力（"depth"，"word"，"synset"，"score"，"n_candidates"）．
    """
    chain = load_module("ex017")[1].granular_annotation(embedder, image_feat, vocab, -np.inf, mode, max_depth, top_k=1)
    return [{k: s[k] for k in ("depth", "word", "synset", "score", "n_candidates")} for s in chain]


def truncate(chain, theta_f=-np.inf, max_depth=None):
    """
    粒度を変更する：深さ1以降の確信度が θ_f 未満になる直前，または深さ max_depth までの出力を返す．

    Args:
        chain (list[dict]): full_chain の戻り値．
        theta_f (float): 細分化を続ける確信度の下限 θ_f．
        max_depth (int | None): 最大の深さ D（None なら制限しない．0 なら COCO のクラスのみ）．

    Returns:
        list[dict]: 先頭部分の出力の列（深さ0は常に残す）．
    """
    out = chain[:1]
    for step in chain[1:]:
        if (max_depth is not None and step["depth"] > max_depth) or step["score"] < theta_f:
            break
        out.append(step)
    return out


# ---------------------------------------------------------------------------
# order_032：VG の語彙の木による命名（VLM は SigLIP ではなく各手法の F）
# ---------------------------------------------------------------------------
def logit_scale(embedder):
    """
    F の学習済みの温度（logit_scale の exp）を返す．

    Args:
        embedder: ex017 の OpenCLIPEmbedder または ex019 の AlphaCLIPEmbedder．

    Returns:
        float: 温度（CLIP では約100）．
    """
    model = embedder.base.model if hasattr(embedder, "base") else embedder.model
    return float(model.logit_scale.detach().exp().cpu())


class TreeNamer:
    """VG の語彙の木（data.build_tree）に対する命名の手順（VG の木による細分化，平坦な命名，部分木の確率の合計）．"""

    def __init__(self, embedder, tree, coco_synsets):
        """
        Args:
            embedder: F（text_features を持つ）．
            tree (dict): data.build_tree の戻り値．
            coco_synsets (list[str]): 深さ0の COCO の80クラスの Synset 名（data.coco80_vocab の順）．
        """
        self.tree = tree
        self.word_nodes = [n for n in tree["nodes"] if tree["words"][n] is not None]          # 語を持つノード（W）
        self.index = {n: k for k, n in enumerate(self.word_nodes)}
        self.text = embedder.text_features([tree["words"][n] for n in self.word_nodes])     # W×D
        self.scale = logit_scale(embedder)
        self.coco_idx = np.array([self.index[s] for s in coco_synsets])
        self.children = {}
        for n in tree["nodes"]:
            self.children.setdefault(tree["parent"][n], []).append(n)
        self.nodes = tree["nodes"]                                                           # 全てのノード（N）
        node_pos = {n: j for j, n in enumerate(self.nodes)}
        rows, cols = [], []                                                                  # 語 w が u の部分木に含まれるか
        for w in self.word_nodes:
            u = w
            while u != "root":
                rows.append(self.index[w])
                cols.append(node_pos[u])
                u = tree["parent"][u]
        self.subtree = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(self.word_nodes), len(self.nodes)))
        self.depth = np.array([tree["depth"][n] for n in self.nodes])

    def cosine(self, feats):
        """
        領域の特徴と全ての語のコサイン類似度．

        Args:
            feats (torch.Tensor): 領域の特徴（B×D）．

        Returns:
            np.ndarray: コサイン類似度（B×W，float32）．
        """
        return (feats.float() @ self.text.T).numpy().astype(np.float32)

    def softmax(self, cos, temperature=1.0):
        """
        語の確率（VG.pdf の式(6)．ロジットは F の温度 × コサイン類似度）．

        Args:
            cos (np.ndarray): コサイン類似度（B×W，または長さWの一部）．
            temperature (float): 温度 T．

        Returns:
            np.ndarray: 確率（cos と同じ形，最後の軸で和が1）．
        """
        z = cos * self.scale / temperature
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def tree_chain(self, cos_row):
        """
        VG の木による細分化：深さ0は COCO の80クラスから選び，以降は木の子の中で確信度（softmax）が最大の語を選んで葉まで下る．

        Args:
            cos_row (np.ndarray): 1つの領域のコサイン類似度（長さW）．

        Returns:
            list[dict]: 深さの浅い順の出力（"depth"，"word"，"synset"，"score"，"n_candidates"）．truncate で粒度を変える．
        """
        p = self.softmax(cos_row[self.coco_idx])
        j = int(np.argmax(p))
        node = self.word_nodes[self.coco_idx[j]]
        chain = [{"depth": 0, "word": self.tree["words"][node], "synset": node, "score": float(p[j]), "n_candidates": len(p)}]
        while True:
            kids = [c for c in self.children.get(node, []) if c in self.index]
            if not kids:
                return chain
            p = self.softmax(cos_row[[self.index[c] for c in kids]])
            j = int(np.argmax(p))
            node = kids[j]
            chain.append({"depth": len(chain), "word": self.tree["words"][node], "synset": node, "score": float(p[j]),
                          "n_candidates": len(kids)})

    def flat(self, cos, tau=0.0):
        """
        VG の語彙の平坦な命名：全ての語の確率が最大の語を出力する（最大の確率が τ 未満なら出力しない）．

        Args:
            cos (np.ndarray): コサイン類似度（B×W）．
            tau (float): 出力する確率の下限．

        Returns:
            list[str | None]: 出力するノード名（長さB）．
        """
        p = self.softmax(cos)
        j = p.argmax(axis=1)
        return [self.word_nodes[k] if p[b, k] >= tau else None for b, k in enumerate(j)]

    def subtree_mass(self, cos, temperature, tau):
        """
        部分木の確率の合計による命名（VG.pdf の式(7)）：部分木の確率の合計 m_u が τ 以上のノードのうち，
        深さが最大（同じ深さなら m_u が最大）のものを出力する．

        Args:
            cos (np.ndarray): コサイン類似度（B×W）．
            temperature (float): 温度 T．
            tau (float): 閾値 τ．

        Returns:
            list[str | None]: 出力するノード名（長さB．m_u ≥ τ のノードが無ければ None）．
        """
        mass = np.asarray((self.subtree.T @ self.softmax(cos, temperature).T).T)              # B×N
        key = np.where(mass >= tau, self.depth[None, :] + mass, -np.inf)                     # 深さを優先し，同じ深さは m_u
        j = key.argmax(axis=1)
        return [self.nodes[k] if np.isfinite(key[b, k]) else None for b, k in enumerate(j)]
