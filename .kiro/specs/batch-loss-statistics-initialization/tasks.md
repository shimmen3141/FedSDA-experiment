# 実装タスク
保存前の独立task graphレビュー後に承認する。既存venv/pytest基盤で順次実行。

- [x] 1. batch損失から全体・クラス別初期統計を作る
  - CPU float32の外部loss/labels/class_count入力を検査し、旧torch reduction順と全体/classのsingleton値・昇順classを保つ。
  - TDDで旧registerを学習しないstubから直接呼び、二値/多クラス・singleton/複数件・欠落・非連続・丸め境界の全fieldを照合する。
  - 完了時、初期集計を返すpure関数が旧正常seedへ完全一致し、学習・モデル登録・store更新を持たない。
  - _Boundary: batch initial statistics_
  - _Requirements: 1.1, 1.2, 1.3, 3.1_

- [x] 2. 拒否・独立性と統計管理への接続を検証する
  - 不正type/dtype/device/layout/shape/空/件数差/数値/labelを拒否し、入力を変えないことを確認する。
  - frozen結果と入力/別結果の独立性、gradmode・共有RNG/default型/device、keywordを検証する。
  - 完了時、seed→store保存→帰属追加を旧更新へ照合し、全体基準値選択へ明示接続できる。LEGACY007の空NaN登録と新非空拒否も同入力で対照する。
  - _Depends: 1_
  - _Boundary: test integration_
  - _Requirements: 2.1, 2.2, 2.3, 3.2_

- [ ] 3. 依存境界・全回帰と独立起動を統合検証する
  - exact数値module/型とtorchだけをASTで許可し、store/別symbol/private/別module/methods/runtime/旧/NumPyを禁止注入する。
  - 完了時、全tests旧11/最終3golden不変、CPUtorch fresh smoke、9条件/roadmap/LEGACY007扱いとLuna最終GOの証拠が揃う。
  - _Depends: 1, 2_
  - _Boundary: final integration_
  - _Requirements: 3.3_

## Implementation Notes
事前学習の逐次Welfordseedやモデルprepare/登録には本関数を先取り適用しない。
