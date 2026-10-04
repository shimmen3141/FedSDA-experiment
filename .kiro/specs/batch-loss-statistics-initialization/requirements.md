# 要件: batch損失からモデル初期統計を作る
## Boundary Context
学習後に外部で計算された標本別有界損失と正解クラスから、モデル全体・クラス別の初期統計を作る。
損失生成/モデルforward・学習・登録・ID採番・送信・統計所有/merge・監視・警報後進行は含めない。
入力は非空の同じ標本集合の損失とラベル、明示class_count。各損失は有限の0〜1、ラベルは0以上class_count未満の整数値。
本移植の数値範囲は既存CPU float32分類経路。1件時の全体偏差平方和0.1とクラス別0の違いも維持する。

## 入力契約
損失はCPU float32のdense strided Tensorで形[N]、正解ラベルは同じ型・layoutで形[N,1]。Nは両者で同一の正整数とする。非連続strided入力も受理する。
class_countはbool以外のbuiltin intで2以上。ラベルは有限の整数値かつ0以上class_count未満、損失は有限の0〜1とする。型・形・範囲を暗黙変換しない。

## Requirements
### Requirement 1: 初期集計
1. When 非空batchの標本別損失と正解クラスを受ける, the Batch Loss Statistics Initializer shall 標本数・旧batch演算順の平均・不偏分散から得る偏差平方和を全体初期統計として返す。
2. When batchにクラスの標本が含まれる, the Batch Loss Statistics Initializer shall そのクラスの件数・平均・偏差平方和を作り、欠落クラスを省き、クラス番号の昇順で返す。
3. While batch全体または一クラスが1件だけである, the Batch Loss Statistics Initializer shall 全体1件の偏差平方和を0.1、クラス別1件を0として保持し、観測を逐次追加する集計に置き換えない。

### Requirement 2: 入力拒否と副作用
1. If 入力型・数値範囲・形・標本数対応・クラス数/ラベルが契約に反する, the Batch Loss Statistics Initializer shall 項目名と理由を示して拒否し、入力を変更しない。
2. The Batch Loss Statistics Initializer shall 入力と独立した変更不能な数値結果を返し、入力の変更や返却後の変更を別結果へ伝播させない。
3. The Batch Loss Statistics Initializer shall 勾配計算状態・共有乱数・既定数値型/deviceを変更しない。

### Requirement 3: 移植・接続
1. The Batch Loss Statistics Initializer shall 全体/各クラスの全値を、二値/多クラス、複数件/1件、欠落クラスと入力順を含む旧初期統計へ直接照合できる。
2. The Batch Loss Statistics Initializer shall 返却した初期統計を既存storeへ明示して保存・帰属更新・基準値選択へ接続でき、モデル登録や送信を暗黙に実行しない。
3. The Batch Loss Statistics Initializer shall 旧import/aliasなしで独立起動でき、既存golden不変と完成範囲・発見事項を記録する。
