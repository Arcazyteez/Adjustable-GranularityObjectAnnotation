# order_038

## ユーザーの指示
> @outputs/ex018_owl_region_check/fig_examples_gain.png このような比較実験画欲しいが、もう一回やり直してください

（直前の質問：「検出器として、YOLOとDETRの性能が知りたい、実験結果とその画像を教えてください」）

## 解釈
- ex018 の fig_examples_gain.png（左に YOLO11x，右に OWLv2 の検出を色分けして並べた比較画像）と同じ形式で，**YOLO11x と DETR** の検出を比較する画像を作る．
- 「もう一回やり直す」は，YOLO と DETR の比較として検出の比較実験をやり直すことと解釈した．新しく検出器を動かすのではなく，ex017 で保存した COCO val2017 の test 500枚の検出結果（YOLO11x conf 0.25，DETR 閾値 0.5）を用いる．これは ex019〜ex021 で用いた検出結果と同一である．
- 正解は COCO の注釈（iscrowd=0）とし，クラスを問わず IoU 0.5 以上のボックスがあれば検出できたとする（ex017 の E2 と同じ定義）．
- 画像に加えて，正解を「両方が検出」「YOLO だけ」「DETR だけ」「両方が見逃し」に分けた集計表を作り，人物と人物以外に分けて示す．
- 比較画像は，DETR だけが検出した正解が多い画像（fig_examples_detr_gain.png）と，YOLO だけが検出した正解が多い画像（fig_examples_yolo_gain.png）の2種類とする．
