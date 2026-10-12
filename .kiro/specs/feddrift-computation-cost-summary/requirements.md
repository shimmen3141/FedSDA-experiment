# Requirements Document

## Project Description (Input)

FedSDAとFedDriftを比較する研究者向け。FedDriftのモデル計測は接続済みだが、保有モデル×標本の分母とパラメータ算術の計数がなく、新FedSDAと同じ計算量summaryを得られない。実際の保有数と実行した演算を記録し、共通単位の指標へ接続する。

## Boundary Context

対象は予測前の保有モデル数、サーバの集約/統合/診断距離MAC、client自身の統合MAC、基本指標導出への共通summary接続。用途別損失監視、追加診断、保存/CLI、方式や数値の変更は対象外。固定旧/golden、学習・乱数・通信・既存基本指標を維持する。

## Requirements

### Requirement 1: 保有モデル×標本の分母

1. When 標本の処理が成功したとき, the FedDrift client shall 予測直前の実モデルの保有数を標本順に1件記録する。切替やモデル作成より前の数を使う。
2. If 標本処理が失敗したとき, the 記録 shall その標本の件数を追加しない。同期・終端や未処理tailでも標本記録を追加しない。
3. The 分母 shall 全clientの処理済み標本数と保有数の延べ合計であり、現在モデルだけを使う予測のforward標本数やrun終了時のモデル数を代用しない。

### Requirement 2: 実行したパラメータ算術の計数

1. When 集約が成功したとき, the FedDrift server shall 加重和へ実際に参加した正の標本重みのパラメータ値数を累積する。コピーのみ/標本重み0/除算/損失統計は含めない。通常R回の集約と終端の集約を含む。
2. When モデル統合が成功したとき, the FedDrift server shall 保存済みモデルの平均へ実際に掛けて足した全メンバーの値数を数える。実演算される重み0メンバーと単独群も含み、重み全0の既存均等平均も数える。統合なしなら0増分。
3. When 診断のパラメータ距離を計算したとき, the FedDrift server shall 対ごとの全独立モデルパラメータ値数×3（差の2乗和と2つの2乗和）を別計数へ累積する。判定の距離/損失評価/通信はこの計数へ混ぜない。
4. When clientが自分のモデルを加重統合したとき, the 計数 shall 同じ値数のMACをclient統合として別に得られる。サーバ統合に重複計上せず、モデルLinear hookのMACとも混ぜない。
5. If 事前検査で操作を拒否したとき, the 計数 shall 変更されない。計数の追加は既存の数値・モデル状態・乱数・通信量を変えない。

### Requirement 3: 共通summaryと指標

1. When 基本指標を導出したとき, the 導出 shall 共通ServerComputationCounts、処理標本総数、保有モデル×標本数、client統合MACを返す。保有数の型/値/記録件数の不整合を読取りのみで拒否する。
2. When 正しいModelComputationCountsを渡したとき, the 導出 shall 共通ComputationCostSummaryを同じ単位・正規化・ゼロ分母NaNで返す。初期準備はモデルMACに含めない。
3. If モデル計数を渡さないとき, the 導出 shall model_computation_countsとsummaryをNoneとし、未計測を0と表さない。既存の無計測呼出しを維持する。
4. The 検証 shall 実旧/既存基本goldenとの一致、実操作に基づく独立計数、作成/統合/隔離/0重み/R/終端/ゼロ同期/破損記録、fresh/exact境界、全回帰で無干渉と内訳を検証する。
