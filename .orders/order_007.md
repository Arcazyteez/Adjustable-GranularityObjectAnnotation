# Order 007

## ユーザーの指示（原文）
```
分かった、現在新しいのものをスライドに追加して
```

## 指示の解釈
order_006で実装・論文に追加したYOLOとCLIP・WordNetのハイブリッド構成（ex006，論文4.9節）の内容を，既存の発表スライド `slides_object_annotation.pptx` に追加する．

- 既存の生成スクリプト `slides/build_slides.py` にスライドを追加し，スライド全体を再生成する．既存のスライドの内容は維持する．
- 追加する内容は，ハイブリッド構成の仕組み（深さ0をYOLO11x，深さ1以降をCLIPとWordNet），切り替え規則と閾値 $`\gamma`$，COCOの500枚での定量結果（表6），COCO以外の語を含む画像の定性結果（図10），および深さ0の誤りがそのまま下位語へ展開されるという限界とする．
- まとめのスライドにもハイブリッド構成の結果を反映する．
- 数値・図は `outputs/ex006_coco_val500_yolo_clip_hybrid/` と `paper/figures/` の実測値・図のみを用いる．
- `.ai/ai-pptx-kit` の規約（28/24/20pt，「・」，ネイティブ数式，LibreOffice不使用）に従い，`.ai/pptx-render` でPDF化してスタイルを確認する．
