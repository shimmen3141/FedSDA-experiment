# 要件: モデル全体・クラス別損失統計の管理

## Boundary Context
一つの所有者がモデルIDごとの全体/クラス別損失集計を保持し、明示seedの受取、帰属損失の追加と独立した参照を提供する。
モデル実体/クラス上限、seedのbatch算出、統計merge/ID変更/削除時機、監視基準・候補進行・学習・サーバ同期は対象外。
更新は監視ごとに自動実行せず、上位が帰属した損失を明示して呼ぶ。

## 入力契約
model_idはbool以外のsigned builtin int。class_idはNoneまたはbool以外の非負builtin int、クラス上限は上位が検査する。
全体/各classはbounded-loss-momentsの契約に従う。各集計の件数/M2から追加の全体-class整合条件を推測しない。
初期model集合と全体/class seedを明示する。class順・model順は受取/最初の追加順を保持する。

## Requirements
### Requirement 1: 明示集計の保持と参照
1. When 初期モデル全体/クラス別集計を受ける, the Loss Statistics Store shall 全要素を検査して独立保持し、非零1件seed・class欠落・全体だけのseedも再計算せず受け取る。
2. When 一つのモデルの集計を明示して保存する, the Loss Statistics Store shall 新IDなら追加し、既存IDなら全体と全classを一括置換し、他IDの値と順序を変えない。
3. When モデル単位または全状態を参照する, the Loss Statistics Store shall 保持順を維持した独立した変更不能な結果を返し、未登録モデルは登録済み0件と区別して示す。

### Requirement 2: 帰属損失の追加
1. When model_idと一つの有界損失を受ける, the Loss Statistics Store shall 対象の全体集計へ旧Welford順で一件追加し、未登録モデルなら空集計から作る。
2. When 正解class_idを指定した帰属損失を受ける, the Loss Statistics Store shall 全体とそのclassだけへ同じ損失を追加し、未登録classなら空集計から作る。
3. While class_idを指定しない, the Loss Statistics Store shall 全体だけへ追加し、他モデル/全classを変更せず、全体とclassの件数合計が一致することを要求しない。

### Requirement 3: 原子的拒否と独立性
1. If 型・ID・集計field・損失または更新結果が契約に反する, the Loss Statistics Store shall 項目名と理由で拒否し、全体/classの部分更新や空entry追加を含め全状態を変更しない。
2. The Loss Statistics Store shall seed入力と参照結果の変更を内部へ伝播させず、別の所有者やモデル/classに可変参照を共有しない。
3. The Loss Statistics Store shall 統計操作で共有乱数・既定数値型/deviceを変更しない。

### Requirement 4: 移植・接続の証拠
1. The Loss Statistics Store shall 正/負モデルID、class指定/なし、seedと未登録からの各更新の全値を旧モデル統計更新へ直接照合できる。
2. The Loss Statistics Store shall 全体集計を既存の用途別基準選択へ明示して監視/参照比較に接続し、class統計を監視の基準として暗黙選択しない。
3. The Loss Statistics Store shall 許可した数値部品と標準機能だけを参照し、旧import/aliasなし、既存golden不変・独立起動・完成範囲/既存発見事項の扱いを記録する。
