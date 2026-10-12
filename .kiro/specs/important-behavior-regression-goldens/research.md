# Research & Design Decisions
## Summary
- Feature: important-behavior-regression-goldens
- Discovery Scope: Extension（検証の追加）
- 既存13構成builder・3手法のexecuteとrecord tablesをそのまま使用できる。既存環境goldenは旧名の指標/列を基準とし、新指標・計算量・通信量すべての新実装基準とは別である。
## Research Log
### 数値環境と既存検証
`environments/golden/windows-cpu/README.md`、既存環境別test、CIを読んだ。ホスト数値照合は診断でローカル全回帰が必須。新構造testは必須を維持し、新数値testにだけmarkerを付ける。
### 重要動作
`test_fedsda_run_metric_derivation.py`の既存条件と経路assertを調べた。sine2/1500件は採用・first_interval棄却・alternative_reference_refit再利用・統合を、mnist2/100件は採用・second_interval棄却を含む。旧helperを使う探索scriptで両条件の新設定と全記録列を取得した（venv/refactoring-tests/behavior-probe-*.json、Git管理外）。この探索はgolden生成/新testに旧helperをimportする理由にはしない。FedDrift既存Linuxドリフト条件seed71/17件/閾値0.03を参照する。
## Design Decisions
### 版の分離
Gitは全変更履歴を保存する。baseline_versionは意図した動作の基準、schema_versionは検証JSONの形式、environmentは同じ動作の環境別数値を識別する。毎commitの版増加やlatest自動選択は採らない。LEGACY/IMPROVE後の版追加は後続spec。
### 実装量
独自の生産golden subsystemを作らずtest helperへ閉じる。公開serializer/table/snapshot APIを採用する。既存goldenを拡張すると旧基準と新基準を混同するので別directoryに保存する。新依存なし、外部技術調査不要。
## Risks & Mitigations
現在の挙動の固定は新たな正しさの証明ではない。既存移植の対照とレビューを根拠とし、生成2回一致と必須経路検査で空比較を防ぐ。環境一致でもCPU/ビルド差はありうるためCIは診断。完全一致を緩めて環境差を吸収しない。
## Skills
fable-method、kiro init/requirements/design/tasks/impl/validate-impl/verify-completionを使用。要求/設計gate、task graph inline sanityの対象はこのspec。独立CLI仕様レビューは別段階に維持する。
