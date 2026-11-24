# report_026：COCO外の物体で，経路が正解の枝から外れる段階の分析

## 1. 概要

- **問い：** COCO外の物体について，提案手法（確率を WordNet の上位語へ集約し，上位語から貪欲に下位語へ進む greedy）が，最初の段階（深さ0）で誤った上位語を選び，その後も誤った枝の中で細かい語へ進む事例がある．例えば，kangaroo が ditch になる．これがどこで，なぜ起きるかを調べた．
- **結論：**
  1. **深さ0で外れる割合は小さい．** 対象の領域の約10%であり，外れた事例全体の約2割にとどまる．
  2. **深さ0で外れた事例の約8割は，見た目の誤りではない．** WordNet における語義の置き場所の違いが原因である．
     - **「物」と「物質」：** lettuce，onion，mushroom などは，植物（object の下）と食品（matter の下）の2つの語義を持つ．GT と出力で語義が食い違う．
     - **「部分」と「全体」：** handle，knob，doorknob などは，WordNet で part（object の下）に置かれている．出力は whole → artifact の側に進む．
  3. **外れる段階で最も多いのは artifact の直下である．** 例えば，covering（覆い）が正しいのに instrumentality（道具）を選ぶ．
  4. **原因は，集約による「語の多い枝」への偏り．** 外れた事例の63〜76%で，選んだ枝の方が，正しい枝より下にある語彙の語が多かった．
     - 選んだ枝の語数は，正しい枝の中央値で2.8〜4.5倍である．
     - instrumentality の下には1632語，covering の下には343語ある．
     - 確率を足し合わせるため，1語ずつの確率が低くても，語の多い枝の合計が大きくなる．
  5. **ご指摘の kangaroo の画像：** 現在の手法では ditch にはならなかった．代わりに，次の2つの誤りが起きる．
     - YOLO11x が sheep と bear と検出し，そのクラスがそのまま出力される．
     - 加えた領域は mammal → placental（有胎盤類，0.41）→ ungulate → even-toed ungulate と進む．正しい metatherian（有袋類）は0.022しかない．語彙の中で有袋類の下は kangaroo と koala の2語だけで，有胎盤類の下は181語ある．
     - ditch は，以前の版（ex010，report_011）で灯台の画像の地面に付いた語である．

## 2. 方法（`programs/ex015c_divergence_analysis/`）

### 2.1 対象
- **画像：** テスト用の画像．
- **領域：** COCO外の GT と IoU 0.5以上で対応した加えた領域．対応付けは ex015b の paired naming と同じで，検出のスコアの順に行う．
- **OWLv2 の重み：** ST（6,385領域），FT（8,415領域）．
- **設定：** ex015 で選んだもの．新しい学習は無い．

### 2.2 定義

| 項目 | 定義 |
| :--- | :--- |
| 正解の枝 | GT の Synset の上位語・自身・下位語 |
| 経路 | 根（physical_entity）から出力した語までの Synset の列 |
| fine | 出力が GT 自身か下位語 |
| coarse | 経路が全て正解の枝にあり，GT の上位語で止まった |
| diverged | 経路の途中で正解の枝から外れた |
| unreachable | GT の Synset が根の下に無い．poster，tag，calendar などは WordNet で抽象物（abstraction）の下にある |
| 分岐元 | 外れる直前の（正解の枝にある）Synset |
| 深さ0 | 分岐元が physical_entity，object，whole のいずれか（論文の図で省略している，全ての経路が通る語） |
| 正しい子 | 分岐元の下位語のうち正解の枝にあるもの．その確率と，下位語の中での順位を求めた |

### 2.3 実画像
ex004 の実画像（kangaroo，camel，penguin，helicopter，灯台2枚）にも，YOLO11x，OWLv2 の領域，命名の順に適用した．各段階で確率が上位3件の下位語を記録した．

## 3. 結果

### 3.1 経路の結果の割合（COCO外の GT と対応した加えた領域）

| 結果 | ST | FT |
| :--- | :--- | :--- |
| fine（GT 自身か下位語） | 14.9% | 44.3% |
| coarse（GT の上位語で停止） | 28.1% | 11.8% |
| diverged（外れた） | 55.2% | 42.1% |
| ┗ うち深さ0で外れた | 10.2% | 8.5% |
| unreachable | 1.7% | 1.7% |

fine と coarse の合計（ST 43.0%）は，論文の「同じ領域の命名の正しさ」と一致する．

### 3.2 深さ0で外れた事例（ST 652件）

| 正しい子 → 選んだ子 | 件数 | 主な GT | 種類 |
| :--- | :--- | :--- | :--- |
| part → whole | 207 | handle，wall socket，knob，doorknob | WordNet の構造 |
| object → matter | 162 | onion，mushroom，lettuce，plate（GT は植物・食器の語義，出力は食品の語義） | WordNet の語義 |
| matter → object | 161 | log，toilet tissue，bell pepper，soap（GT は物質の語義，出力は物の語義） | WordNet の語義 |
| artifact → living thing | 47 | — | 見た目の誤り |
| natural object → artifact | 26 | — | 見た目の誤り |
| その他 | 49 | — | 見た目の誤り |

- **WordNet の構造による割合：** 81.3%（FT では74.4%）．part と whole，object と matter の組である．これらは語義を区別すれば正しいとみなせる事例が多い．
  - 例：lettuce は，GT が植物の語義（lettuce.n.02）であるのに，食品の語義の枝に進んだ．
  - 例：plate は，GT が食器（object の下）であるのに，料理（matter → food → dish）に進んだ．
- **見た目の誤り：** 動物を artifact とするなど．深さ0で外れた事例の約2割である．
- **外れた後の長さ：** 外れた後も平均5.1語進む．深さ0で選んだ子の確率の平均は0.57で，正しい子は0.23であり，確信を持って誤った枝に入っている．

### 3.3 深い段階で外れた事例（ST 2,874件）

| 分岐元 | 件数 |
| :--- | :--- |
| artifact | 1,549 |
| instrumentality | 458 |
| device | 99 |
| covering | 82 |

| 正しい子 → 選んだ子 | 件数 |
| :--- | :--- |
| covering → instrumentality | 348 |
| decoration → instrumentality | 174 |
| implement → device | 165 |
| instrumentality → structure | 135 |
| commodity → instrumentality | 112 |

- **正しい子の順位：** 外れた事例の66%（FT 75%）で2位だった．
- **確率の差：** 選んだ子との差が0.1未満の事例は38%（FT 36%）だった．僅差で外れる事例が多い．
- **語数の偏り：** 選んだ枝の方が語数の多い割合は75.8%（FT 63.4%）．語数の比の中央値は4.5倍（FT 2.8倍）である．

| Synset | 下にある語彙の語数 |
| :--- | :--- |
| instrumentality | 1,632 |
| covering | 343 |
| decoration | 102 |
| placental | 181 |
| metatherian | 2 |

### 3.4 実画像（ST）

| 画像 | 結果 |
| :--- | :--- |
| kangaroo | YOLO11x：sheep，bear．加えた領域1（体全体）：animal 0.73 → chordate 0.48 → vertebrate → mammal 0.46 → **placental 0.41（metatherian 0.022）** → ungulate 0.19 → even-toed ungulate 0.16．flat は cub |
| camel | YOLO11x：dog．加えた領域は artifact → device や geographical area．頭部などの物体の一部と砂地 |
| penguin | 加えた領域は penguin まで正しく細分化（例：0.38，0.20） |

FT では，kangaroo の領域1は ruminant → bovid まで進んだ．

### 3.5 図
- `outputs/ex015c_divergence_analysis/{st,ens}/42/fig_examples.png`：深さ0で外れた例．GT の語ごとに1件．緑が GT，赤が加えた領域で，外れた段階の確率も示す．
- `outputs/ex015c_divergence_analysis/{st,ens}/42/fig_open_images.png`：実画像への適用．
- `outputs/ex015c_divergence_analysis/fig_divergence.png`：経路の結果の割合と，正しい子の順位．

## 4. 考察と改善の候補

- **深さ0の誤りの多くは評価の問題でもある．** object と matter，part と whole の食い違いは，同じ語の別の語義に進んだ事例である．次の2つで大きく減らせる．
  - 評価で同じ語（lemma）の別の語義を正解とみなす（例：lettuce の植物と食品）．
  - 階層の上の方の分岐で，複数の語義を持つ語を両方の枝に入れる．
- **語数の偏りの補正（提案の候補）：** 集約した確率を，枝の下の語数で補正する．次の方法が考えられる．
  - 合計の代わりに，最大値や上位の語の確率を用いる．
  - 語数で割った事前確率で正規化する．
  kangaroo のように，語の少ない枝（有袋類）が正しい場合の誤りを減らせる可能性がある．再学習は要らず，検証用の画像で選べる．
- **僅差で外れる事例：** 正しい子が2位で差が0.1未満の事例が約4割ある．僅差の場合は分岐元で止まる規則にすれば，誤った粒度へ進む事例を，正しい粗い語に置き換えられる．
- **YOLO11x のクラスの誤り：** kangaroo を sheep や bear とする誤りは，命名の手法の外にある．YOLO11x のボックスでも，領域の特徴の確率がクラスと大きく食い違う場合に，その確率を用いる余地がある．

## 5. 作成・変更したファイル

| ファイル | 内容 |
| :--- | :--- |
| `.orders/order_026.md` | 指示と解釈 |
| `programs/ex015c_divergence_analysis/{data,model,train}.py` | 分析のプログラム |
| `tests/test_divergence_analysis.py` | テスト3件．全体で60件が合格 |
| `outputs/ex015c_divergence_analysis/` | 結果 |
| `visualize_result.ipynb` | ex015c のセルを加え，実行した |
| `document.md` | ex015c を追記した |

論文とスライドは変更していない．

## 6. 未解決の論点
- 語数の偏りの補正と僅差での停止を実際に評価していない．AP_strict と命名の正しさへの効果は未確認である．
- 語義の食い違いを正解とみなす評価は，指標の定義の変更になる．論文での扱いを決める必要がある．
- ai-agent による検証は，`tokens.json` が無いため行っていない．
