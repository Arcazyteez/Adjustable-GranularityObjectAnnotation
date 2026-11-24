"""
スライド（slides_2026C.pptx）の OWLv2 の説明用の例を探すスクリプト（order_054）．

test の画像（LVIS の注釈がある画像）について，保存済みの検出結果から次を数え，例の候補を JSON に保存する．新しい推論は行わない．
  - DETR（閾値0.5）と OWLv2（物体らしさの上位50個）の箱
  - LVIS の COCO 外の物体のうち，OWLv2 だけが囲めた物体（IoU 0.5 以上）

実行: .venv_pytorch/bin/python slides/owl_example_2026C.py
出力: slides/media_2026C/owl_examples.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "programs", "ex022b_yolo_detr_equal_threshold"))
import train as E22B  # noqa: E402

OWL_PATH = os.path.join(ROOT, "datasets", "ex019_coco_owl_alphaclip_wordnet", "detections_OWLv2_test.pt")
OUT = os.path.join(ROOT, "slides", "media_2026C", "owl_examples.json")
OWL_TOP, DETR_TH = 50, 0.5


def owl_only(noncoco, owl_boxes, detr_boxes):
    """
    COCO 外の物体のうち，OWLv2 の箱だけが囲めた物体を求める．

    Args:
        noncoco (np.ndarray): COCO 外の物体のボックス（K×4，xyxy）．
        owl_boxes (np.ndarray): OWLv2 の箱（N×4）．
        detr_boxes (np.ndarray): DETR の箱（M×4）．

    Returns:
        np.ndarray: OWLv2 だけが囲めた物体の真偽値（K）．
    """
    return E22B.covered(noncoco, owl_boxes) & ~E22B.covered(noncoco, detr_boxes)


def main():
    """
    test の全ての画像で候補を求め，OWLv2 だけが囲めた物体の多い順に JSON に保存する．

    Returns:
        None
    """
    data = E22B.load_data()
    e18 = E22B.load_module("ex018_data", os.path.join(ROOT, "programs", "ex018_owl_region_check", "data.py"))
    lvis = e18.load_lvis()
    owl = torch.load(OWL_PATH, weights_only=False)
    out = []
    for i in data["splits"]["test"]:
        if i not in data["noncoco"] or i not in owl:
            continue
        ob = np.asarray(owl[i]["boxes"], np.float32)[:OWL_TOP]
        db, ds = data["dets"][("DETR", "test")][i]
        db = db[ds >= DETR_TH]
        nc = data["noncoco"][i]
        _, kinds, names = e18.gt_arrays(lvis, i)
        nc_names = [n for n, k in zip(names, kinds) if k == "noncoco"]
        only = owl_only(nc, ob, db)
        out.append({"image_id": int(i), "n_owl_only": int(only.sum()), "n_noncoco": int(len(nc)),
                    "n_detr": int(len(db)), "owl_only_boxes": nc[only].tolist(),
                    "owl_only_names": [n for n, o in zip(nc_names, only) if o], "detr_boxes": db.tolist()})
    out.sort(key=lambda r: -r["n_owl_only"])
    json.dump(out, open(OUT, "w"), indent=0)
    print(len(out), "images ->", OUT)
    for r in out[:15]:
        print(r["image_id"], r["n_owl_only"], r["n_noncoco"], r["n_detr"])


if __name__ == "__main__":
    main()
