"""
WordNetによる粒度可変アノテーション（論文3.4節）．

深さ d≥1 の候補語彙集合を V_k^(d) = Hyponyms(synset(v_k^(d-1))) とし（論文の式(3)），
同一クロップに対して式(2)の argmax を繰り返し適用する．
"""
from nltk.corpus import wordnet as wn

# COCO 80クラス → WordNet Synset の対応表（定義文を確認して手動で決定）
COCO_TO_SYNSET = {
    "person": "person.n.01", "bicycle": "bicycle.n.01", "car": "car.n.01", "motorcycle": "motorcycle.n.01",
    "airplane": "airplane.n.01", "bus": "bus.n.01", "train": "train.n.01", "truck": "truck.n.01",
    "boat": "boat.n.01", "traffic light": "traffic_light.n.01", "fire hydrant": "fireplug.n.01",
    "stop sign": "street_sign.n.01", "parking meter": "parking_meter.n.01", "bench": "bench.n.01",
    "bird": "bird.n.01", "cat": "cat.n.01", "dog": "dog.n.01", "horse": "horse.n.01", "sheep": "sheep.n.01",
    "cow": "cattle.n.01", "elephant": "elephant.n.01", "bear": "bear.n.01", "zebra": "zebra.n.01",
    "giraffe": "giraffe.n.01", "backpack": "backpack.n.01", "umbrella": "umbrella.n.01",
    "handbag": "bag.n.04", "tie": "necktie.n.01", "suitcase": "bag.n.06", "frisbee": "frisbee.n.01",
    "skis": "ski.n.01", "snowboard": "snowboard.n.01", "sports ball": "ball.n.01", "kite": "kite.n.03",
    "baseball bat": "baseball_bat.n.01", "baseball glove": "baseball_glove.n.01",
    "skateboard": "skateboard.n.01", "surfboard": "surfboard.n.01", "tennis racket": "tennis_racket.n.01",
    "bottle": "bottle.n.01", "wine glass": "wineglass.n.01", "cup": "cup.n.01", "fork": "fork.n.01",
    "knife": "knife.n.01", "spoon": "spoon.n.01", "bowl": "bowl.n.01", "banana": "banana.n.02",
    "apple": "apple.n.01", "sandwich": "sandwich.n.01", "orange": "orange.n.01", "broccoli": "broccoli.n.02",
    "carrot": "carrot.n.03", "hot dog": "hotdog.n.02", "pizza": "pizza.n.01", "donut": "doughnut.n.02",
    "cake": "cake.n.03", "chair": "chair.n.01", "couch": "sofa.n.01", "potted plant": "pot_plant.n.01",
    "bed": "bed.n.01", "dining table": "dining_table.n.01", "toilet": "toilet.n.02",
    "tv": "television_receiver.n.01", "laptop": "laptop.n.01", "mouse": "mouse.n.04",
    "remote": "remote_control.n.01", "keyboard": "computer_keyboard.n.01",
    "cell phone": "cellular_telephone.n.01", "microwave": "microwave.n.02", "oven": "oven.n.01",
    "toaster": "toaster.n.02", "sink": "sink.n.01", "refrigerator": "electric_refrigerator.n.01",
    "book": "book.n.02", "clock": "clock.n.01", "vase": "vase.n.01", "scissors": "scissors.n.01",
    "teddy bear": "teddy.n.01", "hair drier": "hand_blower.n.01", "toothbrush": "toothbrush.n.01",
}

# COCOに含まれない追加語（4.6節，22語）→ WordNet Synset
EXTRA_TO_SYNSET = {
    "penguin": "penguin.n.01", "helicopter": "helicopter.n.01", "kangaroo": "kangaroo.n.01",
    "camel": "camel.n.01", "lighthouse": "lighthouse.n.01", "dolphin": "dolphin.n.02",
    "whale": "whale.n.02", "shark": "shark.n.01", "eagle": "eagle.n.01", "owl": "owl.n.01",
    "parrot": "parrot.n.01", "flamingo": "flamingo.n.01", "tiger": "tiger.n.02", "lion": "lion.n.01",
    "monkey": "monkey.n.01", "gorilla": "gorilla.n.01", "panda": "giant_panda.n.01", "deer": "deer.n.01",
    "fox": "fox.n.01", "rabbit": "rabbit.n.01", "squirrel": "squirrel.n.01", "castle": "castle.n.02",
}


def synset_word(synset):
    """
    Synsetを候補語として表す単語（先頭のレンマ名，アンダースコアは空白）に変換する．

    Args:
        synset (nltk.corpus.reader.wordnet.Synset): Synset．

    Returns:
        str: 候補語（例: "African elephant"）．
    """
    return synset.lemma_names()[0].replace("_", " ")


def hyponym_candidates(synset):
    """
    直接の下位語集合を候補語彙として返す（論文の式(3)）．表記が重複する語は先頭のみ残す．

    Args:
        synset (Synset): 1つ浅い段階で選ばれたSynset．

    Returns:
        tuple[list[str], list[Synset]]: 候補語のリストと対応するSynsetのリスト（長さ |V^(d)|）．
    """
    words, synsets = [], []
    for hypo in sorted(synset.hyponyms(), key=lambda s: s.name()):
        word = synset_word(hypo)
        if word not in words:
            words.append(word)
            synsets.append(hypo)
    return words, synsets


def hierarchical_annotation(vlm, crop, root_word, root_synset_name, root_conf, max_depth=None):
    """
    深さ0の推定結果から出発し，下位語集合が空になるか最大深さに達するまで再帰的にアノテーションする．

    Args:
        vlm: score(images, vocab) を持つVLM．
        crop (PIL.Image.Image): 領域kのクロップ X_k．
        root_word (str): 深さ0で推定された単語．
        root_synset_name (str): 深さ0の単語に対応するSynset名．
        root_conf (float): 深さ0の確信度．
        max_depth (int | None): 最大深さ（Noneで無制限）．

    Returns:
        list[dict]: 各深さの {"depth", "word", "synset", "conf", "n_candidates"} のリスト．
    """
    synset = wn.synset(root_synset_name)
    chain = [{"depth": 0, "word": root_word, "synset": root_synset_name, "conf": float(root_conf),
              "n_candidates": None}]
    depth = 1
    while max_depth is None or depth <= max_depth:
        words, synsets = hyponym_candidates(synset)
        if not words:
            break
        scores = vlm.score([crop], words)[0]
        best = int(scores.argmax())
        synset = synsets[best]
        chain.append({"depth": depth, "word": words[best], "synset": synset.name(), "conf": float(scores[best]),
                      "n_candidates": len(words)})
        depth += 1
    return chain
