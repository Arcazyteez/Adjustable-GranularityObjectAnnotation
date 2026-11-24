# Order 014

## ユーザーの指示（原文）
```
以上の実験結果を論文に書いてください
```

## 指示の解釈
order_013（ex012，`進捗報告_2026B.pdf` のアルゴリズムと評価）の実験結果を，新しい和文の論文として作成する．

- **作り方：** order_013の「昔の論文や報告などを使わず」に従い，既存の論文（`main.pdf`，`paper/`，`paper_v2/`，`paper_siglip_comparison/`）の文章は流用しない．`.ai/ai-tex-kit/templates/ja_paper/` から新しいディレクトリ `paper_2026B/` を作成する．
- **構成：** はじめに，関連研究，提案手法（進捗報告のアルゴリズムをAlgorithmとして記述），実験（共通設定，定量評価，定性評価），考察，おわりに，参考文献とする．
- **図表：** `paper_2026B/make_figures.py` で `outputs/ex012_wordnet_granular_annotation/` の結果から作成する．
- **赤字：** 新しい文書のため用いない．
- **その他：** `latexmk` で `paper_2026B/main.pdf` を生成する．`report_014.md` とPDFを作成し，`document.md` を更新する．
