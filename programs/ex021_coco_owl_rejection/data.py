"""
ex021：データと他の実験のモジュールの読み込み（order_033）．

データは ex017・ex019・ex020 と同じ MSCOCO val2017 の dev 500枚，test 500枚である．
検出（OWLv2，DETR，YOLO11x），領域の特徴，VG の木，キャプションによる細かい正解は，ex017・ex019・ex020 のものをそのまま用いる．
"""
import functools
import importlib
import os
import sys

EX_NAME = "ex021_coco_owl_rejection"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", EX_NAME)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", EX_NAME)
PROGRAM_DIRS = {"ex017": "ex017_coco_granular_annotation", "ex019": "ex019_coco_owl_alphaclip_wordnet",
                "ex020": "ex020_coco_owl_alphaclip_granularity", "ex018": "ex018_owl_region_check"}


@functools.lru_cache(maxsize=None)
def load_module(experiment):
    """
    他の実験の data，model，train モジュールを読み込む．
    ex021 自身の data，model，train と名前が衝突するため，一時的に sys.modules から外して読み込み，後で戻す．

    Args:
        experiment (str): "ex017"，"ex019"，"ex020" のいずれか．

    Returns:
        tuple[module, module, module]: data，model，train．
    """
    names = ("data", "model", "train")
    path = os.path.join(PROJECT_ROOT, "programs", PROGRAM_DIRS[experiment])
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, path)
    try:
        modules = tuple(importlib.import_module(n) for n in names)
    finally:
        sys.path.remove(path)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return modules
