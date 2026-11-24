# Order 027

## ユーザーの指示（原文）
```
@main.pdf @進捗報告_2026B.pdf
新しいフォルダーに以上のpdfを参考して、その内容を実装してください
```

## 指示の解釈
`main.pdf`（論文「物体領域検出とカテゴリアノテーションの分離に基づく，WordNet を用いた粒度可変物体アノテーション」）と `進捗報告_2026B.pdf`（「Alg. 粒度可変アノテーション」と評価方法）の内容を，1つの新しい実験フォルダにまとめて実装する．

- **フォルダ：** `programs/ex017_coco_granular_annotation/`（`data.py`，`model.py`，`train.py`）．結果は `outputs/ex017_coco_granular_annotation/`，データ（YOLOの重み，COCO外の画像の複製，語彙のキャッシュ）は `datasets/ex017_coco_granular_annotation/`．
- **自己完結：** 「新しいフォルダー」に実装するため，`programs/common/` や他の実験（ex001〜ex016）のモジュール・結果は読み込まない．COCO の画像と注釈（`datasets/coco/`）と，Hugging Face・torchvision の学習済みの重みのキャッシュのみを用いる．
- **アルゴリズム（2つのPDFの統合）：** 進捗報告の Alg. を中心とする．物体認識モデル $`D`$（DETR，YOLO11x）でボックスを1回だけ求め，各ボックスの切り抜き画像に埋め込みモデル $`F`$（CLIP-RN50，CLIP-ViT-B/32，SigLIP-ViT-B/16．比較対象としてResNet50-IN1k）で単語を付け，スコアが $`\theta`$ 以上の間，WordNetの直接の下位語をたどる．スコアは進捗報告のコサイン類似度と，論文のソフトマックス（確信度）の両方を実装する（論文の設定は $`\theta = -\infty`$ に相当する）．
- **実験（論文の4章と進捗報告の評価）：**
  - E1（論文4.2節）：VLM $`F`$ の比較（ResNet50-IN1k，CLIP-RN50，CLIP-ViT-B/32，SigLIP）．タスクA（COCO 80クラス）・タスクB（キャプションの名詞・動詞，spaCy），P@1/R@1/P@5/R@5，Wilcoxon検定，bootstrap 95% CI．
  - E2（論文4.3節）：物体領域検出モデルの比較（DETR，OWL-ViT，Grounding DINO）．クラス非依存の Recall@IoU0.5 と平均最良IoU．論文の4枚に加えて500枚でも評価する．
  - E3（論文4.4節，進捗報告の定量評価）：初期の語彙をCOCOの80クラス，下位語なしとし，$`D \times F`$ の組み合わせでラベルの一致率（Top-1/Top-5），ボックスのずれ，クラスを含めた適合率・再現率・F値を求め，Close Vocabularyモデル（YOLO11x，DETRのクラス出力）と比較する．$`\theta`$ は検証用の500枚で選び，テスト用の500枚で報告する．
  - E4（論文4.5節）：WordNetの深さごとの確信度（深さ0が正解のインスタンス）．
  - E5（論文4.6節）：COCO外の語を含む画像（ペンギン，ヘリコプター，カンガルー，ラクダ，灯台）．語彙はCOCO 80 + 22語．
  - E6（論文4.7節，進捗報告の定性評価）：初期の語彙をCOCOの80クラスとした場合と，WordNetの上位語（noun.Tops の person，animal，plant，food，artifact，natural object）とした場合の階層的アノテーションの図．
- **データ：** COCO 2017 val から seed 42 で1000枚をサンプリングし，前半500枚を検証用（$`\theta`$ の選択），後半500枚をテスト用とする．
- **その他：** テスト（`tests/test_ex017_granular_annotation.py`），`visualize_result.ipynb`，`document.md`，`.reports/report_027.md` とPDFを作成・更新する．論文・スライドは指示が無いため作成しない．
