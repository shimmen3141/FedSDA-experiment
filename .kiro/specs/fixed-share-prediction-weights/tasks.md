# 実装タスク

全段階承認と命名revisionはspec.jsonが正本。順番に一つずつTDD→Lunaレビュー→主担当検証→commit。既存Python/pytestを再利用し環境導入は不要。共有controllerを変更するので並列化しない。task 1～3は対象機能とschemaを段階検証し、新controllerの依存許可を加えるtask 4で全src境界検査を完了する。

- [x] 1. 明示条件と独立したモデル集合・重み観測を用意する
  - 型・固定条件を再検証し、空/重複/不正IDを状態変更前に拒否する。
  - 一様初期化、負ID、昇順、集合差だけのreset、独立copy・読取診断を公開する。
  - 観測値変更や別実体が所有状態に影響せず、初回resetを計数しないテストが通る。
  - _Boundary: FixedSharePredictionWeightController_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4_

- [x] 2. 観測損失による重み更新と最大重み選択を移植する
  - 有限損失・確率・ID集合を先に検証し、旧演算順で更新する。
  - 単一モデル、制限損失、優先同率と最小IDのleader変更計数を区別する。
  - 各更新後に旧oracleの全状態と完全一致し、不正入力では全状態不変のテストが通る。
  - _Boundary: FixedSharePredictionWeightController_
  - _Requirements: 1.2, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 6.1_

- [ ] 3. 損失列再生と集約後の再較正を移植する
  - 全列を先行検証・コピーしてから再生し、行間のモデル集合変更を許す。
  - 通常再生、集約後再生、明示reset、空列の計数を区別する。
  - 再生後に旧oracleと完全一致し、不正後段でも全状態不変のテストが通る。
  - _Boundary: FixedSharePredictionWeightController_
  - _Requirements: 1.2, 1.3, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1_

- [ ] 4. 依存境界へ統合し、既存回帰と完成範囲を検証する
  - 設定への依存だけを新controllerに許し、上位・旧実装・torch/NumPy・他機能を禁止例で確認する。
  - 新src全走査、全refactoringテスト、既存schema、tests/test_regression.py、tests/test_proposed_regression.pyを含む全testsを実行する。
  - 旧src・goldenに差分がなく、全検証結果と未接続の全体runを区別したGO証拠が残る。roadmapの完成範囲も部品移植として更新する。
  - _Boundary: 依存境界検査とFixedSharePredictionWeightControllerの明示統合_
  - _Depends: 1, 2, 3_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3_

