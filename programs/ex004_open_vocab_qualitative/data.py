"""
ex004のデータ定義（論文4.6節，4.7節）．

- open_vocab: COCOに含まれない語を主要な被写体とする実画像．Wikimedia Commonsから固定のファイル名で取得し，
  ライセンス情報とともにdatasets/以下へ保存する．
- coco: 定性評価用のCOCO画像．seed=42でサンプリングした500枚の順序で，指定カテゴリを含む最初の画像を選ぶ．
"""
import json
import os
import sys

import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.coco import category_names, load_coco, load_image, sample_image_ids  # noqa: E402
from common.paths import DATASET_DIR  # noqa: E402
from common.wordnet_utils import EXTRA_TO_SYNSET  # noqa: E402

DATA_DIR = os.path.join(DATASET_DIR, "ex004_open_vocab_qualitative")
USER_AGENT = {"User-Agent": "bong-repro/0.1 (academic reproduction of an object annotation paper)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# 主要被写体 → Wikimedia Commonsのファイル名
OPEN_VOCAB_FILES = {
    "penguin": "File:Adelie Penguins on iceberg.jpg",
    "helicopter": "File:Z-20J Helicopter flying demonstration 20241115.jpg",
    "camel": "File:Dromedary in Thar desert.jpg",
    "kangaroo": "File:Eastern Grey Kangaroo Feeding edited.jpg",
    "lighthouse_1": "File:Leuchtturm in Westerheversand crop.jpg",
    "lighthouse_2": "File:Point Reyes Lighthouse in December 2019.jpg",
}
# 定性評価に用いるCOCOカテゴリ（論文の図5の被写体に対応）
COCO_QUALITATIVE_CATEGORIES = ["elephant", "zebra", "giraffe", "motorcycle", "dog", "cake", "cat", "bear"]


def download_commons_image(title, path, width=1024):
    """
    Wikimedia Commonsから画像（幅widthのサムネイル）とライセンス情報を取得する．

    Args:
        title (str): ファイル名（"File:..." 形式）．
        path (str): 画像の保存先．
        width (int): 取得するサムネイルの幅．

    Returns:
        dict: {"title", "url", "license", "artist", "description_url"}．
    """
    params = dict(action="query", format="json", titles=title, prop="imageinfo",
                  iiprop="url|extmetadata", iiurlwidth=width)
    page = next(iter(requests.get(COMMONS_API, params=params, headers=USER_AGENT, timeout=30).json()["query"]["pages"].values()))
    info = page["imageinfo"][0]
    response = requests.get(info["thumburl"], headers=USER_AGENT, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    meta = info["extmetadata"]
    return {"title": title, "url": info["thumburl"], "description_url": info["descriptionurl"],
            "license": meta.get("LicenseShortName", {}).get("value"), "artist": meta.get("Artist", {}).get("value")}


def load_open_vocab_images():
    """
    COCO外の語を含む実画像を読み込む（未取得の場合はダウンロードする）．

    Returns:
        tuple[list[dict], list[str]]: 画像ごとの {"name", "image", "meta"} のリストと，
            第1段階の候補語彙（COCO 80クラス + 追加22語 = 102語）．
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    meta_path = os.path.join(DATA_DIR, "open_vocab_meta.json")
    metas = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    samples = []
    for name, title in OPEN_VOCAB_FILES.items():
        path = os.path.join(DATA_DIR, f"{name}.jpg")
        if name not in metas or not os.path.exists(path):
            metas[name] = download_commons_image(title, path)
        samples.append({"name": name, "image": Image.open(path).convert("RGB"), "meta": metas[name]})
    with open(meta_path, "w") as f:
        json.dump(metas, f, indent=4, ensure_ascii=False)
    vocab = category_names(load_coco("instances")) + list(EXTRA_TO_SYNSET)
    return samples, vocab


def load_coco_qualitative_images(n_images=500, seed=42):
    """
    定性評価用のCOCO画像を選ぶ．

    Args:
        n_images (int): サンプリングする画像数．
        seed (int): サンプリングの乱数シード．

    Returns:
        tuple[list[dict], list[str]]: 画像ごとの {"name", "image", "meta"} のリストと，COCO 80クラス名．
    """
    coco = load_coco("instances")
    image_ids = sample_image_ids(coco, n_images, seed)
    samples, used = [], set()
    for category in COCO_QUALITATIVE_CATEGORIES:
        cat_id = coco.getCatIds(catNms=[category])[0]
        image_id = next(i for i in image_ids if i not in used and coco.getAnnIds(imgIds=i, catIds=cat_id))
        used.add(image_id)
        samples.append({"name": f"coco_{category}_{image_id}", "image": load_image(coco, image_id),
                        "meta": {"image_id": image_id}})
    return samples, category_names(coco)
