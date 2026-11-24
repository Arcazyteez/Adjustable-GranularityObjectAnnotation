# order_037

## ユーザーの指示
棄却スコアをmassに変更して再実験していいですか？その結果が欲しい

（直前のやり取り：ex019 の E2「命名のモデルをそろえた比較」（検出器 DETR / OWLv2@10 × F CLIP-B/32 / Alpha-CLIP，棄却のスコア joint = VLM の確信度 × 検出スコア，深さ0のラベルは COCO の softmax）の条件を説明した．）

## 解釈
- ex019 の E2 の検出器 × F の比較を，棄却のスコアを部分木の確率の合計（mass）に変えて再評価する．棄却の判定の比較であり問題の本質は ex021 と同じため，ex021 の train.py に実験 E2_factorial を追加する（通し番号は維持）．
- 条件（itertools.product の全ての組）：
  - 検出器：DETR（閾値0.5），OWLv2@10（ex019 で dev から選んだ予算），OWLv2@20，OWLv2@50
  - F：CLIP-ViT-B/32（切り抜き），Alpha-CLIP（context1.5），Alpha-CLIP（crop）
  - 棄却のスコア：joint（従来．比較の基準），mass，mass_joint（mass × 検出スコア）．DETR の検出スコアは DETR 自身の確信度
  - 重複・部分の除去：none（ex019 と同じ），nms_contain（ex021 の最良）
  - 深さ0のラベル：coco（ex019 と同じ），mass
- θ は ex021 と同じく dev の F値（と hF）で選び，test で報告する．vg_subtree の T，τ は F ごとに ex020 で選んだもの（CLIP-B/32 は Old，Alpha-CLIP は New の条件）を用いる．
- 主な問い：
  1. ex019 の4つの組（dedup none，coco ラベル）で，スコアを mass・mass_joint にすると結果はどう変わるか．
  2. mass を用いた場合の DETR と OWLv2，CLIP と Alpha-CLIP の差（report_035 で未実施とした「最良の条件での CLIP と Alpha-CLIP の比較」を含む）．
- 新たに計算するのは DETR のボックスの Alpha-CLIP（crop）の特徴のみで，他は ex019 のキャッシュを用いる．

## 中止
- ユーザーの指示「もう大丈夫です、実験中止」により，E2_factorial の実行を途中で止めた．144通りのうち78通り（DETR，OWLv2@10 の全ての組と OWLv2@20 の一部）の結果が `outputs/ex021_coco_owl_rejection/E2_factorial/` に残っている．表（summary_test.csv），検定，図は作成していない．
- `python train.py e2` を再実行すると，完了した条件はスキップして残りから続けられる．
