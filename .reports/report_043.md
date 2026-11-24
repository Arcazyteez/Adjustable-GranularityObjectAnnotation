# report_043：語彙の木の表記を「WordNet∩VGの木」に統一

## 1. 実施内容
語彙の木の作り方は次の通りである．
- 語（ノード）：VG の注釈に現れる WordNet の Synset と COCO の80クラス．
- 親子の関係：WordNet の上下関係．

「VGの語彙の木」と書くと，WordNet をやめて VG に替えたように読めてしまう．そのため，スライドと論文の表記を「WordNet∩VGの木」に統一した．
数値と実験は変えていない．

## 2. スライド（`slides_2026C.pptx`，17枚）
| 枚 | 変更 |
| :--- | :--- |
| 4 | 表の「現在」の欄を「WordNet∩VG[4]の木」にした．箇条書きに「語彙：VGに現れる語に絞り，親子はWordNetの上下関係を使う」を加えた |
| 6 | タイトルを「提案モデル：OWLv2＋Alpha-CLIP＋WordNet∩VGの木」にした．「語：VGに現れるWordNetのSynset，親子：WordNetの上下関係」の注を加えた |
| 14 | 表の行名を「WordNet∩VGの木の下位語」にした |
| 15 | タイトルと図の凡例を「WordNet∩VG」にした．凡例が線と重なっていたため，縦軸を1.25まで広げた |
| 17 | まとめの1行目を「再学習なしで粒度を変えられるモデルを示した」と「OWLv2＋Alpha-CLIP＋WordNet∩VGの木」の2行に分けた |

## 3. 論文（`paper_2026C/main.pdf`，7ページ）
ai-tex-kit の修正依頼の規約に従い，order_040 の赤字（`\rev{}` と題名の赤）を全て黒に戻した．今回の修正箇所だけを赤字にした．

| 箇所 | 変更 |
| :--- | :--- |
| 概要 | 「WordNetとVisual Genome（VG）を組み合わせた語彙の木」．「VGの語彙の全ての語」を「語彙の木の全ての語」にした |
| はじめに | 「WordNetとVisual Genome（VG）を組み合わせた語彙の木」．第三の貢献では「WordNet∩VGの木（3.3節）」とした |
| 3.1節，図1 | 「WordNet∩VGの木」．図のノードは「WordNet∩VGの確率」とした |
| 3.2節 | 検出器を替えても，命名は「WordNet∩VGの木」で行うと書き換えた |
| 3.3節 | 「語はVGに現れるWordNetのSynset，親子はWordNetの上下関係であるため，これをWordNet∩VGの木と呼ぶ．」を追加した |
| 4章 | 表の行名「WordNet∩VGの木の下位語」，本文の2か所，図 fig_step の凡例 |
| 考察 5.3 | 「WordNet∩VGの木は，VGの注釈に現れる語のみで木を作るため」 |
| 結論 | 「WordNetとVGを組み合わせた語彙の木」 |

- 「VGの平坦な語彙」「VGの語彙には VG v1.2 を用いた」など，VG のデータそのものを指す箇所は変えていない．
- 定義の文を追加したため，一度は8ページになった．はじめにの定義の文を3.3節への参照に置き換え，7ページに戻した．
- latexmk の Overfull はない．

## 4. 確認
- pptx-render で `slides_2026C.pdf` を作成した．変更した5枚について，文字のはみ出し，引用との重なり，単語の途中での改行がないことを目で確認した．
- 論文の1〜3ページを画像にして，赤字が今回の修正箇所だけであることを確認した．
- 変更前のバックアップ：
  - `/tmp/claude-1000/-home-user-workspace-bong/097f1326-9149-47d6-84e7-54fbda3cf0a0/scratchpad/paper_before042/`
  - `/tmp/claude-1000/-home-user-workspace-bong/097f1326-9149-47d6-84e7-54fbda3cf0a0/scratchpad/slides_2026C_before043.pptx`

## 5. 未解決の論点
- 2〜3枚目（背景，前稿のモデル）の「WordNetの下位語」は前稿の手法を指すため，そのままにした．
- 題名の「語彙の木」は変えていない．題名にも WordNet と VG を入れる場合は，さらに長くなる．
