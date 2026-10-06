# 要求 revision 1

## 1. 吸収

- 1.1 When 保有済みモデルのIDと帰属確定標本の列が渡されたとき, the absorption operation shall 各標本を渡された順に、そのモデルの学習標本の末尾へ追加する。
- 1.2 When 標本に真の概念IDが付いているとき, the absorption operation shall そのモデルの割当概念計数の該当概念を1増やす。概念IDがない標本では計数を変更しない。
- 1.3 The absorption operation shall 各標本について、そのモデルの現在のparameterで標本1件の有界損失を評価し、その損失と標本の観測クラスで、そのモデルの全体と当該クラスの損失統計を1件ぶん更新する。統計が未登録のモデルは件数0から始める。
- 1.4 When 標本列が空のとき, the absorption operation shall どの状態も変更しない（そのモデルの空の標本列も作らない）。
- 1.5 The absorption operation shall モデルのparameter/grad/optimizer、他モデルの標本・計数・統計、学習計数、現在の学習帰属ID、評価標本を変更せず、乱数を消費しない。

## 2. 呼出し契約

- 2.1 If モデルIDがbuiltin int以外、またはownerが対象の具体型でないなら, the absorption operation shall TypeErrorを送出し、どの状態も変更しない。
- 2.2 If 標本列がexact tupleでない、要素が学習標本recordでない、概念ID列がexact tupleでない、長さが標本列と異なる、または要素がNoneでもbuiltin intでもないなら, the absorption operation shall TypeErrorまたはValueErrorを送出し、どの状態も変更しない。
- 2.3 If そのモデルが保有されていないなら, the absorption operation shall 標本列が空でもKeyErrorを送出し、どの状態も変更しない。
- 2.4 If いずれかの標本が1件ぶんでない、または特徴・ラベルが既存の損失評価の検証で拒否されるなら, the absorption operation shall 例外を伝え、先行する標本を含めどの状態も変更しない。
- 2.5 The absorption operation shall 2.1〜2.4の検証と全標本の損失評価を終えてから状態を変更する。検証済み入力に対する既存owner APIの順次更新であり、途中例外へのrollbackは提供しない。

## 3. 境界と移植

- 3.1 The absorption operation shall 各ownerを借用して既存APIで更新し、自身の永続状態を持たない。帰属先の決定、現在の学習帰属IDの変更、保留標本の収集/解放、学習の実行、評価標本の保存、計算量診断の記録を行わない。
- 3.2 When 有効な同じ状態へ吸収を適用したとき, the new absorption operation shall 実旧の吸収処理による学習標本の列と順序、割当概念計数、損失統計の全field（全体とクラス別、クラスの初出順を含む）を維持する。吸収後の学習が実旧と同じ数値で継続できるようにする。
