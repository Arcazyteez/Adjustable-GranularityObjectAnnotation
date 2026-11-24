"""ex022（YOLO11x と DETR の検出の比較）の判定関数のテスト．"""
import importlib.util
import os

import numpy as np

PATH = os.path.join(os.path.dirname(__file__), "..", "programs", "ex022_yolo_detr_region_check", "train.py")
spec = importlib.util.spec_from_file_location("ex022_train", PATH)
ex022 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex022)


def test_box_iou_identity_and_disjoint():
    """同じボックスの IoU は1，離れたボックスの IoU は0になることを確かめる．"""
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(ex022.box_iou(a, b), [[1.0, 0.0]])


def test_image_record_status():
    """正解ごとに YOLO と DETR のどちらが検出したかを正しく判定することを確かめる．"""
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], float)
    yolo = np.array([[0, 0, 10, 10]], float)
    detr = np.array([[0, 0, 10, 10], [20, 20, 30, 31], [100, 100, 110, 110]], float)
    r = ex022.image_record(gt, ["person", "cup", "dog"], yolo, detr)
    assert r["gt_yolo"].tolist() == [True, False, False]
    assert r["gt_detr"].tolist() == [True, True, False]
    assert r["detr_hit"].tolist() == [True, True, False]
    table = ex022.gt_status_table([r])
    row = table.set_index("group").loc["all"]
    assert (row["both"], row["yolo_only"], row["detr_only"], row["neither"]) == (1, 0, 1, 1)


def test_covered_empty():
    """ボックスが無い場合に全て False を返すことを確かめる．"""
    assert ex022.covered(np.zeros((2, 4)), np.zeros((0, 4))).tolist() == [False, False]
