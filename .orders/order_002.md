# Order 002

## ユーザーの指示（原文）
```
以上の実験結果で論文を書いてください
```

## 指示の解釈
order_001で実施した再現実験（ex001〜ex004，`.reports/report_001.md`）の結果を用いて，和文の学術論文（TeX）を作成する．

- 研究内容（物体領域検出とカテゴリアノテーションの分離，WordNetによる粒度可変アノテーション）と章構成は `main.pdf` を踏襲し，実験結果・図表はすべて本プロジェクトの `outputs/` の実測値に置き換える．`main.pdf` の数値を本論文の結果として記述しない．
- 再現実験で新たに得られた知見（クロップの前処理（正方形パディング）の影響，DETRの検出スコア閾値の影響，giraffe → okapi や Indian elephant → white elephant のような候補が1語の分岐の問題）を論文に反映する．
- テンプレートは `.ai/ai-tex-kit/templates/ja_paper/`（`ieicejsp.cls`）を `paper/` にコピーして用いる．章ごとに `paper/sections/` に分割する．
- ページ数の指定がないため，`main.pdf` と同程度（8ページ前後）とする．
- 著者名・所属は不明のため，プレースホルダとする．
- 図は `outputs/` の結果から論文用に作り直す（`paper/make_figures.py`）．
- コンパイルは `latexmk -pdfdvi -synctex=1 main.tex` で行い，最新のPDFを `paper/main.pdf` として保持する．
