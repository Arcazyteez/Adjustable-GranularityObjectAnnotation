"""プロジェクト内のパス定義と，ai-dev-kitのユーティリティの読み込み．"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
COCO_DIR = os.path.join(DATASET_DIR, "coco")
COCO_IMAGE_DIR = os.path.join(COCO_DIR, "val2017")
COCO_INSTANCES_PATH = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
COCO_CAPTIONS_PATH = os.path.join(COCO_DIR, "annotations", "captions_val2017.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

sys.path.insert(0, os.path.join(PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402,F401
