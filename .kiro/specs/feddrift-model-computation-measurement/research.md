# Research and Design Decisions

## Summary

既存FedSDA計測はモデルの全結合層hook、準備後からの差分、ラウンド内2区分と終端を持つ。FedDriftはexact factory契約と同期中のR回学習を持つため、factory/serverを型違いのwrapperへ差替えず同じrun内部で成功境界を観測する。

## Research Log

- `runtime/fedsda_measured_run_execution.py`、`learning/models/model_computation_measurement.py`: 共通field/単位とhook finallyの再利用。独立モデルのModelOwnedHiddenLayersも既に対象。
- `runtime/feddrift_stream_run_execution.py`: initial_preparationにMNIST生成器読込みを含む。event/失敗stage/torchscopeの境界は維持。
- `runtime/feddrift_run_server.py`: synchronize_timestep内部でprepare後にR回train/aggregate。終端はreadyの場合だけ追加集約。計測で順序を作り直さない。
- `comparison-computation-metrics`: サーバ算術/保有モデル数/summaryは既存FedSDA側の別責務。FedDriftには記録がまだないため本specはモデル計測を先に完結させる。
- `tests/refactoring/test_feddrift_stream_run_execution.py`: 実旧入口oracleと全状態照合を再利用。旧用途別counterは今回のgrad/Linear定義と一対一でなく、MAC検証は独立小モデル算式を加える。

## Decisions and Risks

- 標本処理/synchronizationは手法上の段階。FedDriftの学習は同期内、FedSDAの区間末学習はlocal_processing内。比較は共通MAC合計で行い段階名を混同しない。
- 外側でpublic methodをmonkeypatchする方式、exact型をsubclassで回避する方式、runループ複製を不採用。private checkpointだけを共通内部実行へ追加。
- fable-methodとcc-sdd init/requirements/design/tasks/implを参照。証拠を先に定義し、仕様書群の独立レビュー前にtest/sourceへ進まない。
