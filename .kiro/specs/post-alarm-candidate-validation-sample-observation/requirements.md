# 要求 revision 1

## 1. 観測

- 1.1 When ラベル観測後の1標本、候補の分類器、参照モデルIDごとの分類器が渡されたとき, the observation operation shall 候補と各参照の分類器でその標本の有界損失を1回ずつ評価し、標本位置・候補の損失・参照モデルIDごとの損失を、損失収集へ同じ観測回として追加する。
- 1.2 When 追加を終えたとき, the observation operation shall 損失収集が規定の検証標本件数へ到達したかどうかを返す。到達しても採否評価・確定を呼ばない。
- 1.3 The observation operation shall 候補と参照の分類器のparameter/grad/学習modeを変更せず、乱数を消費しない。損失収集以外の状態を変更しない。

## 2. 呼出し契約

- 2.1 If 損失収集が対象の具体型でない、または参照分類器の対応がexact dictでないなら, the observation operation shall TypeErrorを送出し、損失収集を変更しない。
- 2.2 If 標本が1件ぶんでない、または候補・参照の分類器、特徴、ラベルが既存の損失評価の検証で拒否されるなら, the observation operation shall 例外を伝え、損失収集を変更しない。
- 2.3 If 標本位置が直前の位置に連続していない、参照モデルIDの集合が収集開始時と異なる、または損失収集が既に規定件数へ到達しているなら, the observation operation shall 損失収集の例外を伝え、損失収集を変更しない。
- 2.4 The observation operation shall 全ての損失評価を終えてから損失収集へ1回だけ追加する。

## 3. 境界と移植

- 3.1 The observation operation shall 損失収集を借用して既存APIで更新し、自身の永続状態を持たない。分類器の生成・複製・学習、収集の開始、採否評価、確定、記録、通知、計算量診断を行わない。
- 3.2 When 同じ値の候補・参照モデルと同じ標本列を観測したとき, the new observation operation shall 実旧の観測処理が蓄積する候補損失の系列と参照モデルごとの損失の系列を、値と順序を含めて維持する。規定件数へ到達した後に既存の採否評価と確定を適用すると、実旧の観測処理が到達時に行う確定と同じ状態になるようにする。
