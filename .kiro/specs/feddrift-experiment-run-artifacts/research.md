# Research & Design Decisions

## Summary
- Feature: feddrift-experiment-run-artifacts。既存機能の接続（Extension）。
- FedSDAは完全設定→実験run→表→JSON/NPZの層を持ち、掃引はその入口を呼ぶ。FedDriftは実行/計測/指標まで完成し、保存層がない。
- 使用skill: fable-method、ローカルkiro-spec-init/requirements/design/tasks/impl。cc-sddの標準雛形とEARS/境界/traceabilityを用いる。主担当が調査/実装、独立CLIレビューは仕様群と全task後の各1回。

## Research Log

### 単一run境界
- Sources: runtime/fedsda_experiment_run.py、fedsda_run_settings.py、fedsda_run_artifact_files.py、sweep/experiment_sweep_execution.py。
- FedDrift固有の実行設定と参加者設定がある。FedSDAのexact型を受ける部品へaliasを置かず、同じ責務のFedDrift入口を作る。
- 今回は1 runの実行と保存の流れ。掃引軸/CLI/presetは別の利用境界であり、保存先の識別/再開/並列と合わせて次に接続する。

### 利用できる記録
- Sources: feddrift_client_stream_processing.py、feddrift_run_server.py、feddrift_model_diagnostic_metric_derivation.py、fedsda_run_record_tables.py。
- processorは単独では短いflushも扱うが、全体runは生成端数を処理しない。今回は全体runの記録だけを保存する。予測は切替前の現在モデル。登録・クラスタ・クロス評価の既存round_indexは通常時刻timestep_indexを格納するので、保存列名もserver_timestep_indexとする。
- 基本/計算量/モデル診断は既存の共通形式。予測混合/候補検証/再較正はFedDriftに適用しない。

### Oracle
- 既存test_feddrift_stream_run_execution.run_inputs/run_legacyとtest_feddrift_run_metric_derivation.basic_metric_valuesを再利用する。旧全体入口は前specまでに実測済み。今回も仕様レビュー前に既存対象testをbootstrapする。
- 表の新形式は旧と同じキーではない。testで予測・切替・概念列等の対応を明示し、共通記録の全fieldは既存実旧対照testと新表の配列照合を合わせる。

## Design Decisions
- 一般化: JSON/NPZの形式は共通の構成を使うが、既存FedSDAの保存コードの再編を混ぜない。共通serializerはそのまま再利用する。
- 採用: 既存NumPy NPZ/標準JSON/Pathと計測・指標の関数。新依存や汎用artifact frameworkは作らない。
- 簡略化: 今回はpresetなし、利用者が既存部分型を束ねる。全列の公開関数を増やさず表builder一つで扱う。

## Risks & Mitigations
- 初期/終端と通常時刻の混同: 別表と明示的timestep_index。同期なしの空2次元列を検査。
- 欠測と一時負モデルID: 欠測を使う列を限定しモデルIDはそのまま保存。
- 保存の失敗: 全内容生成後に書込む。失敗注入で作成物の掃除を確認。同名への並行保存は対象外。
