# 実装タスク
保存前の独立task graphレビュー後に承認する。既存venv/pytestで順次実行。

- [x] 1. ID対応後のローカル選択とサーバ補完を実装する
  - TDDでモデル実体なしの旧apply_server_mappingを直接呼び、通常/欠落/identity/負ID/chain/cycle/unused/collision最大n/tie反転とserver補完を照合する。
  - 正localをより大件数のserverより優先し、欠落/zero localへzero serverも補完する。空local/空対応、server省略/None/空を受理する。
  - 完了時、全体/class全値・クラス順・モデル順が旧正常入力と完全一致し、whole record選択だけを行うpure関数を返す。
  - _Boundary: mapped statistics selection_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [x] 2. 拒否・独立性と統計管理への接続を検証する
  - 型/ID/重複/後段不正/壊された統計と入れ子を拒否し、入力非変更とfrozen結果/別結果独立/共有環境を確認する。
  - 完了時、store snapshot→選択→新store構築で旧sourceIDが残らず、次帰属更新を旧更新へ照合し、用途別baselineへ明示接続できる。
  - _Depends: 1_
  - _Boundary: test integration_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 3. 依存境界・全回帰と独立起動を統合検証する
  - exact型module/symbolだけをASTで許可し、store/private/別module/上位/旧/torch/NumPyを禁止注入する。
  - 完了時、全tests旧11/最終3golden不変、stdlib fresh smoke、全10条件とroadmap/発見有無/完成範囲を記録し、Luna最終GOの証拠が揃う。
  - _Depends: 1, 2_
  - _Boundary: final integration_
  - _Requirements: 3.4_

## Implementation Notes
サーバ集計やモデル処理は含めない。旧localdict参照共有との差は新独立コピー契約であり、旧正常不具合の修正とは扱わない。
