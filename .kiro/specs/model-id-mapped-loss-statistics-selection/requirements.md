# 要件: モデルID対応後の損失統計選択
## Boundary Context
サーバから届いたモデルID対応に従って、クライアントの全体・クラス別損失統計を選び直す。
統計の加算・平均・クラスの合成はせず、旧実装の候補選択規則を保つ。
モデル実体、学習データ、予測重み、現在の帰属ID、登録・送信・サーバ集計・警報後進行は含めない。
統計の所有と逐次更新は既存store、ID対応表の生成は上位処理が担う。

## 入力契約
ローカル統計とサーバ統計は、重複しないモデルIDと既存の変更不能な全体・クラス別統計の順序付き組。
ID対応は元IDから変更先IDへの表。全モデルIDはbool以外のbuiltin intで、負値も許す。
統計の件数・平均・偏差平方和・クラスIDの契約は既存統計型と同じ。クラス上限やモデル集合との一致は要求しない。
空統計・空対応を許す。サーバ統計の省略は空統計と同じ。未使用の対応項目も含め入力全体を検査する。

## Requirements
### Requirement 1: ローカル統計のID対応と候補選択
1. When ローカル統計とID対応表を受ける, the Mapped Loss Statistics Selector shall 対応を一度だけ同時に適用し、対応のないIDを保ち、連鎖・循環を追跡せず、統計にない元IDの項目から統計を生成しない。
2. When 複数のローカル統計が同じ変更先IDになる, the Mapped Loss Statistics Selector shall 全体件数最大の統計を丸ごと選び、同件数ではローカル入力順で先の統計を採用し、全体・クラス別の数値とクラス順を保持する。
3. The Mapped Loss Statistics Selector shall 変更先IDがローカル入力に初めて現れた順で結果を返し、候補の置換でその位置を変えない。

### Requirement 2: サーバ統計による補完
1. When サーバ統計を受ける, the Mapped Loss Statistics Selector shall サーバIDへ対応表を再適用せず、ローカル結果にないIDまたはローカル全体件数ゼロのIDだけをサーバ統計で丸ごと置換する。
2. While 選択済みローカル全体件数が正である, the Mapped Loss Statistics Selector shall サーバ件数がそれより大きくてもローカル統計を維持する。
3. The Mapped Loss Statistics Selector shall サーバ件数ゼロも補完に使用し、既存IDの位置を維持し、サーバのみのIDをサーバ入力順で末尾へ追加する。

### Requirement 3: 検査・独立性・接続
1. If 入力の型・ID・重複・統計値が契約に反する, the Mapped Loss Statistics Selector shall 項目名と理由を示して入力全体を拒否し、入力を変更せず結果を返さない。
2. The Mapped Loss Statistics Selector shall 入力・別呼出の結果と独立した変更不能な統計を返し、共有乱数・既定数値環境を変更しない。
3. The Mapped Loss Statistics Selector shall 選択値とモデル/クラス順を旧処理へ直接照合でき、結果を既存storeの新実体へ明示保存して帰属更新・基準値選択へ接続できる。
4. The Mapped Loss Statistics Selector shall 旧import/aliasなしで独立起動でき、既存golden不変と完成範囲・実測した発見事項を記録する。

