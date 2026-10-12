# Research & Design Decisions

## Summary
- Feature: feddrift-experiment-sweep、既存入口の拡張。
- 単一run/artifactsは完成済み。既存sweepはFedSDA型と実行を固定しているが、計画/並列/再開/CSVは再利用できる。

## Research Log
### 接続と既定
- Sources: runtime/feddrift_run_settings.py、feddrift_run_client_settings.py、feddrift_server_settings.py、sweepの5module、旧config.py、experiment.py、既存test_experiment_sweep.py。
- 旧の検出バッチは粒度/集約間隔/学習量を兼ねるため既存間隔軸と一致させる。閾値はclient/server両方が読む。同じ値というparticipant契約を維持。
- 旧dataset別の設定とoptimizerは既存FedSDAの既定と同じ値だが、FedSDA完全設定を生成して転用せず、FedDriftの型へ直接設定する。
- 新たな実旧メソッドoracleは追加しない。既定は実旧config/旧get_dataset_specを照合し、実行は上流の実旧対照済み単一run APIを再利用する。

## Architecture Pattern Evaluation
同じCLI/計画のunion拡張を採用。別FedDrift専用CLIは再開・並列を複製する。将来の手法registryは現在2手法に不要。

## Design Decisions
- 専用閾値軸の明示指定はFedDrift選択時だけ許す。mixedではFedSDAを増やさない。既定軸はNoneで表し省略と明示を区別する。
- hashは完全設定を含む既存canonical JSONのまま。結果の指標/NPZは新しく組み直さない。
- 利用skillsはfable-method、kiro-spec-init/requirements/design/tasks/impl/validate-impl/verify-completion。手順の実装主担当・独立CLIレビュー規約を優先する。

## Risks & Mitigations
- mixedで不要runの複製: 件数/順序のtest。
- 型追加でFedSDAのhash/値が変わる: 固定識別子と既存sweep全test。
- wrong-method成果物をskipする: 手法別loaderと不一致test。
- spawnとCLIの実経路: 小さい実runを別processで実行し全成果物を比較。
