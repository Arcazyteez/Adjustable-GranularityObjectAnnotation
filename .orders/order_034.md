# order_034

## ユーザーの指示
この結果を論文とスライドに生成してください

## 解釈
- 「この結果」：report_033（ex021）の結果を中心に，新しい提案手法（OWLv2 + Alpha-CLIP + WordNet）の一連の結果をまとめる．
  - ex018：OWLv2 の領域の確認
  - ex019：古い提案手法・YOLO11x との比較
  - ex020：VG の語彙・枠・部分木の確率の合計による粒度の変更
  - ex021：棄却の判定の探索
- 比較は order_032 の3手法（New，Old，YOLO11x 単体）とする．ハイブリッド手法は載せない．SigLIP は使わない．
- 論文：
  - 新しいディレクトリ `paper_2026C/` に，`paper_2026B` と同じ ieicejsp の和文の2段組で作成する．
  - VG.pdf と同程度の約6ページとする．
  - VG.pdf（著者らの先行研究）は参考文献として引用し，そこから用いた語彙・枠・部分木の確率の合計はその旨を明記する．
  - 図は `paper_2026C/make_figures.py` で outputs から作り，latexmk で PDF を生成する．
- スライド：
  - `.ai/ai-pptx-kit/templates/research.pptx` を元に，`slides/build_slides_2026C.py` で `slides_2026C.pptx` を作成する．`slides/build_slides.py` の部品を用いる．
  - 研究発表（約12〜15分，研究室の進捗報告を想定）とする．
  - PDF は `.ai/pptx-render` で作成し，スタイルを確認する．
- 投稿先・ページ数の指定は無いため，上記を仮定とし，レポートに明記する．
