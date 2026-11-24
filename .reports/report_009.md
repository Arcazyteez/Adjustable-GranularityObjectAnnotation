# report_009：SigLIPとシグモイド，YOLOとVLMのスコアの比較，WordNetの同じ深さの語彙（ex008）

## 1. 概要

order_009では，現在の構成（CLIP-ViT-B/32とソフトマックス）に次の3つの変更を加え，現在の構成と比較した．

1. **SigLIPとシグモイド：** VLMをSigLIP-ViT-B/16（open_clip `ViT-B-16-SigLIP`，webli）に変更し，確信度をソフトマックス $`p_v = \exp(l_v) / \sum_u \exp(l_u)`$ ではなく，語ごとに独立なシグモイド $`p_v = \sigma(l_v)`$ で計算する．ここで $`l_v = t \cos(\mathbf{x}, \mathbf{w}_v) + b`$ である．
2. **スコアの比較（`hybrid_cmp`）：** YOLOのボックスのクロップに対するVLMの最大確信度が，YOLOの検出スコアより高い場合にVLMの語を深さ0とする．
3. **WordNetの同じ深さの語彙（`*_wn{k}`）：** COCOの各カテゴリについて，最長の上位語の経路で $`k`$ 段上の祖先をとる．その祖先からちょうど $`k`$ 段下のSynset（COCOのカテゴリの「いとこ」）を，深さ0の候補とする．

実験は `programs/ex008_coco_val_siglip_wordnet_root/` で行った．
比較結果は，既存の論文（`paper/main.pdf`）には反映せず，別の文書 `paper_siglip_comparison/main.pdf`（6ページ）として出力した．

| 項目 | 内容 |
| :--- | :--- |
| VLMと正規化関数 | CLIP + ソフトマックス（現在の構成），SigLIP + ソフトマックス（モデルのみ変更），SigLIP + シグモイド（モデルと正規化関数を変更） |
| 深さ0の決め方 | `proposed`（DETR + COCO 80語），`proposed_wn{k}`，`hybrid`（YOLOの語），`hybrid_switch_c0.5`（現在のハイブリッド構成），`hybrid_cmp`（COCO 80語でスコアを比較），`hybrid_cmp_wn{k}` |
| 閾値 | DETR $`\theta \in \{0.5, 0.97\}`$，YOLO11x $`\in \{0.25, 0.7\}`$ |
| 語彙の条件 | $`k \in \{1, 2, 3\}`$．物理的な実体（physical_entity.n.01）の下位語に限る．COCOのカテゴリの上位語・下位語は除き，SemCorの出現頻度が1以上の語に限る |
| 分割・指標 | ex007と同じ分割（dev500，test500，lvis_dev，lvis_test）と指標．$`\tau`$ は lvis_dev で決め，16通りから選んだ（シグモイド用に $`10^{-6}`$〜$`10^{-3}`$ を追加） |
| COCO外の物体の評価 | 次の条件を満たす検出ボックスについて，深さ0の語がLVISのSynsetの上位語または自身である割合を求めた（lvis_test_noncoco）．<br>・LVISのアノテーションのSynsetが，COCOのカテゴリの上位語・下位語でない<br>・レンマ名がCOCOのカテゴリ名と一致しない<br>・COCOのGTと対応しない |
| 定性評価 | ex004の実画像6枚（YOLO11xの閾値0.25） |
| 確認 | CLIP + ソフトマックスの `proposed`，`hybrid` の値はex007と一致した（dev500 の $`\theta = 0.5`$ で $`n = 2343`$，$`\theta = 0.97`$ で Top-1 0.604）．テストは21件すべて合格した |
| 実行 | 300条件の評価で合計約4分（画像特徴の計算を除く）．全体では約20分 |

## 2. 結果

### 2.1 ソフトマックスとシグモイド

図1(a) は，深さ0が誤った枝と正しい枝を，最深段の確信度で区別する AUROC である（test500）．
図1(b) は，深さ0が正しいインスタンスの深さ1の正誤を，深さ1の確信度で区別する AUROC である（lvis_test）．

![図1](../outputs/ex008_coco_val_siglip_wordnet_root/fig_auroc.png)

| 手法 | VLMと正規化関数 | Top-1 | 誤枝AUROC（最深段） | 誤枝AUROC（累積） | 深さ1正解率 | 深さ1 AUROC | τ | hF（τ） | hF（深さ1まで） |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| proposed 0.5 | CLIP + softmax | 0.552 | 0.611 | 0.508 | 0.440 | 0.744 | 0.2 | 0.462 | 0.451 |
| proposed 0.5 | SigLIP + softmax | 0.527 | 0.600 | 0.634 | 0.462 | 0.843 | 0.5 | 0.480 | 0.453 |
| proposed 0.5 | SigLIP + sigmoid | 0.527 | 0.715 | 0.633 | 0.462 | 0.587 | 1e-4 | 0.450 | 0.453 |
| hybrid 0.25 | CLIP + softmax | 0.969 | 0.418 | 0.430 | 0.407 | 0.762 | 0.3 | 0.647 | 0.648 |
| hybrid 0.25 | SigLIP + softmax | 0.969 | 0.480 | 0.483 | 0.405 | 0.780 | 0.3 | 0.648 | 0.648 |
| hybrid 0.25 | SigLIP + sigmoid | 0.969 | 0.617 | 0.594 | 0.405 | 0.644 | 0.05 | 0.659 | 0.648 |

- **誤った枝の検出：** シグモイドでは，誤った枝の最深段の確信度が低くなる．AUROC はハイブリッドで0.418から0.617に，提案手法で0.611から0.715に改善した．SigLIP + ソフトマックスのハイブリッドは0.480なので，改善の主因はモデルの変更ではなく，正規化関数の変更である．
- **深さ1の正誤：** 深さ0が正しい場合の深さ1の正誤では，逆にソフトマックスの方が良い．SigLIP の場合，ソフトマックスは0.843（提案手法）・0.780（ハイブリッド），シグモイドは0.587・0.644だった．兄弟の語どうしを相対的に比べるには，ソフトマックスが向いている．
- **停止規則（図2）：** ハイブリッドでは，シグモイドの $`\tau = 0.05`$ で hF が0.659となった．CLIP の0.647や深さの上限1の0.648をわずかに上回る程度である．提案手法では，シグモイドの深さ0の確信度が小さく，τ を上げると深さ0で棄却されやすい．そのため，hP の高い領域では他の構成より劣った．
- **シグモイドの値の小ささ：** 深い段階の確信度は0.01以下になることが多い（例：bird > nonpasserine bird (0.00)）．選ばれた τ も $`10^{-4}`$ や0.05と小さい値だった．

![図2](../outputs/ex008_coco_val_siglip_wordnet_root/fig_stopping_scorers.png)

### 2.2 YOLOとVLMのスコアの比較（hybrid_cmp，test500）

| 閾値 | VLMと正規化関数 | 切替 | 改善 | 悪化 | Top-1（基準 hybrid） |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0.25 | CLIP + softmax | 103 | 7 | 92 | 0.936（0.969） |
| 0.25 | SigLIP + softmax | 331 | 20 | 304 | 0.858（0.969） |
| 0.25 | SigLIP + sigmoid | 6 | 1 | 5 | 0.967（0.969） |
| 0.7 | CLIP + softmax | 21 | 0 | 21 | 0.969（0.982） |
| 0.7 | SigLIP + softmax | 127 | 5 | 122 | 0.908（0.982） |
| 0.7 | SigLIP + sigmoid | 1 | 0 | 1 | 0.982（0.982） |

- **ソフトマックスでは精度が下がる：** 切替の大部分が，YOLO の正解を不正解に変えた．
- **切替の頻度は尺度の違いで決まる：** 閾値0.25での確信度の中央値は次のとおりである．
  - YOLO の検出スコア：0.787
  - VLM の最大確信度：CLIP + softmax は0.391，SigLIP + softmax は0.755，SigLIP + sigmoid は0.022

  VLM の方が高かったインスタンスの割合は，それぞれ24.0%，52.2%，3.0%だった．つまり切替が起きるかどうかは，予測の正しさではなく，2つのモデルのスコアの尺度の違いで決まっている．
- **候補を COCO の80語にした hybrid_cmp：** COCO 外の物体の正解率は，すべての構成で0だった．

### 2.3 WordNetの同じ深さの語彙

| k | 語彙数 | COCO外の語 | 22語のうち含まれる語 | lvis_test のCOCO外511カテゴリのうち上位語を持つもの |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 213 | 133 | 2（deer，fox） | 96 |
| 2 | 681 | 601 | 4（monkey，deer，fox，squirrel） | 232（45%） |
| 3 | 1115 | 1035 | 5 | 289 |

penguin，camel，kangaroo，helicopter などは，COCO のいとこより深い語なので語彙に含まれない．penguin はこの設計では bird と答えるのが正しい粒度になる．

図3 の横軸は COCO の Top-1 正解率（test500），縦軸は COCO 外の物体の深さ0の正解率（lvis_test_noncoco，YOLO の閾値0.25，262件）である．

![図3](../outputs/ex008_coco_val_siglip_wordnet_root/fig_tradeoff.png)

| 構成（閾値0.25 / θ=0.5） | VLMと正規化関数 | COCO Top-1 | COCO外の正解率 |
| :--- | :--- | :---: | :---: |
| hybrid / hybrid_switch_c0.5 / hybrid_cmp | 全て | 0.858〜0.969 | 0 |
| hybrid_cmp_wn2 | CLIP + softmax | 0.962 | 2.3%（6件） |
| hybrid_cmp_wn2 | SigLIP + softmax | 0.920 | 5.3%（14件） |
| hybrid_cmp_wn2 | SigLIP + sigmoid | 0.945 | 3.8%（10件） |
| proposed_wn2 | CLIP + softmax | 0.302（基準0.552） | 7.3%（521件中38件） |
| proposed_wn2 | SigLIP（両方） | 0.304（基準0.527） | 10.0%（521件中52件） |

- **COCO の物体での悪化：** SigLIP + sigmoid の hybrid_cmp_wn2 では，test500 で65件が切り替わり，そのうち62件が悪化だった．物体の部分や類義の語が選ばれている（person → leg，tv → screen，oven → stove，bottle → slit）．
- **COCO 外の物体で正しく呼べた例：** plate，can，pear，sack，pot，bucket など，容器や食品が中心である．
- **語彙の質の問題：** 深さ0の上位に slit，worsted，column のような多義の語が多く現れた．

### 2.4 定性結果（SigLIP + sigmoid，図4）

![図4](../outputs/ex008_coco_val_siglip_wordnet_root/fig_qualitative.png)

- **上段（hybrid_switch_c0.5，現在の構成）：** penguin の3領域のうち2領域（0.84，0.81）と camel（0.75）が，COCO 外の語に切り替わった．
- **下段（hybrid_cmp_wn2）：**
  - penguin は bird になった．COCO と同じ深さの語としては正しい．
  - camel と kangaroo は語彙にないため，YOLO の dog と sheep のままだった．
  - ヘリコプター画像の下の小さな車両は，military vehicle（0.80）になった．
- **参考（SigLIP + softmax の hybrid_switch_c0.5）：** penguin，helicopter，camel，kangaroo の4つすべてが正しい語（確信度1.00）になった．出力は `outputs/ex008_coco_val_siglip_wordnet_root/SigLIP-ViT-B16_softmax/hybrid_switch_c0.5/open_vocab_th0.25/42/grid.png`．

## 3. 考察

- **シグモイド：**
  - 「深さ0の誤りを深い段階の確信度で検出できない」という ex007 の課題には有効だった．
  - 一方で，深さ1の兄弟の語どうしの比較ではソフトマックスに劣った．
  - 両者は相補的なので，語の選択・兄弟の比較にはソフトマックス，誤った枝の検出・棄却にはシグモイドを使う組み合わせが考えられる．
- **スコアの比較：** 尺度の異なる2つのスコアを直接比べているため，切替の数は正規化関数だけで6件から331件まで変わった．使うには，検証用の分割で確信度を正解率に対応付ける較正が必要である．
- **WordNet の同じ深さの語彙：** 「YOLO を再学習せずに，COCO と同じ粒度で COCO 外の語に対応する」という目的には合っている．ただし，今の語彙では効果が小さく，COCO の精度が下がる．原因は次の3つである．
  - COCO 外のカテゴリの55%が，語彙に上位語を持たない．
  - WordNet のレンマに，物体の部分や多義の語が多い．
  - 同じ深さという制約により，camel のような基本的な語が除かれる．
- **論文への反映（提案）：**
  - シグモイドの結果は，深い段階の確信度の課題に対する対策として追記できる．
  - スコアの比較と WordNet の語彙は，現状では hybrid_switch_c0.5 を上回らない．提案手法としては採用せず，検討結果と今後の課題として書くのが妥当である．

## 4. 作成・変更したファイル

| ファイル | 内容 |
| :--- | :--- |
| `.orders/order_009.md` | 指示と解釈 |
| `programs/ex008_coco_val_siglip_wordnet_root/{data,model,train}.py` | ex008 のプログラム（ex007 のモジュールを importlib で読み込んで再利用する） |
| `tests/test_siglip_wordnet_root.py` | 同じ深さの語彙，正規化関数，hybrid_cmp の判定のテスト（4件） |
| `outputs/ex008_coco_val_siglip_wordnet_root/` | 全条件の結果と集計表，図 |
| `datasets/ex008_coco_val_siglip_wordnet_root/` | 深さ0の語彙（k = 1，2，3） |
| `visualize_result.ipynb` | ex008 の表と図1〜4のセルを追加して実行した |
| `paper_siglip_comparison/` | 比較用の文書（main.tex，sections/，figures/，main.pdf） |
| `document.md` | ex008 と比較文書の情報を追記した |

既存の論文 `paper/` とスライドは変更していない．

## 5. 未解決の論点と次の候補

- **確信度の組み合わせ：** ソフトマックスとシグモイドを組み合わせる（語の選択はソフトマックス，停止・棄却はシグモイド）．効果は未検証である．
- **スコアの較正：** YOLO の検出スコアと VLM の確信度を，検証用の分割で温度スケーリングなどにより較正してから比べる．
- **深さ0の語彙の選択：**
  - 物体の名前としての使われやすさで語彙を絞る（具体性の評定，LVIS・ImageNet・Objects365 での出現頻度など）．
  - 部分を表す語（leg，screen）を除く．
  - min_count を大きくする．
  - 同じ深さという制約を緩め，「COCO のいとこと，その子」までを許す．
- **VLM の規模：** SigLIP は B/16 だけを評価した．より大きいモデル（SO400M など）や SigLIP2 は未評価である．
- **seed：** seed は42の1つだけで，分割の違いによるばらつきは評価していない．
- **論文とスライド：** 結果を論文とスライドに反映するかは未決定である．
- **tokens.json：** ファイルがないため，ai-agent による検証は行っていない．
