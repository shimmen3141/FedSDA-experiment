# 要求 revision 1

## 1. 登録確認の組立

- 1.1 When 現在の学習帰属IDが負で、同IDの学習状態が保有されているとき, the confirmation operation shall 同モデルを通知された非負の正式IDへ付け替え、統計・学習標本・評価標本の存在する元項目で先を上書きし、学習/割当計数を先へ加算移管する。元項目欠落の場合は既存先を維持する。
- 1.2 When 現在の学習帰属IDが既に非負のとき, the confirmation operation shall 送信保留だけを解除し、その他のownerを変更せずNoneを返す。
- 1.3 When 付替えを完了したとき, the confirmation operation shall 現在帰属IDを正式IDへ変更した後に送信保留を解除し、変更前後のID recordを返す。通知/ログは自動発行しない。
- 1.4 The confirmation operation shall モデルparameter/grad、共有参照、optimizer本体と蓄積state、既存標本payload、取得済みsnapshot/recordを保持し、数値計算・乱数抽出・optimizer resetを行わない。

## 2. 呼出し契約

- 2.1 If 正式IDがbuiltin int以外または負なら, the confirmation operation shall TypeErrorまたはValueErrorを送出し、どのownerも変更しない。bool/int派生型/暗黙変換を受理しない。
- 2.2 If 渡されたownerが対象の具体型でないなら, the confirmation operation shall TypeErrorを送出し、どのownerも変更しない。非負現在IDの分岐でも全引数を検証する。
- 2.3 If 負の現在IDのモデルが保有されていないなら, the confirmation operation shall 更新開始前にKeyErrorを送出し、全ownerを維持する。欠落モデルの再構築は呼出し側の責務である。

## 3. 境界と移植

- 3.1 The confirmation operation shall 各ownerを借用して既存APIで更新し、自身の永続状態を持たない。新モデル作成/選択・ID採番・損失評価・サーバ対応表・送受信・予測重み変更を行わない。
- 3.2 When 有効な同じ保有済み状態へ確認を適用したとき, the new confirmation operation shall 実旧確認処理のID、辞書順序、上書きと加算、保留解除、数値/参照保持を維持する。後続学習が付替え済みIDと同じ蓄積optimizer stateを使えるようにする。
