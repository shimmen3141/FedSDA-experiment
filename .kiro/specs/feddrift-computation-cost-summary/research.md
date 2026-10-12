# Research and Design Decisions

## Summary

モデル計測の次の責務として、保有モデルの分母とパラメータ算術の実計数を共通summaryへ接続する。共通型/単位を維持し、FedDrift固有のclient平均も別fieldで保存する。

## Research Log

- `evaluation/computation_cost_summary.py`: 共通summaryのclientはLinear MAC、serverはparameter MAC。分母0はNaN。構造変更なしで利用可能。
- `runtime/feddrift_client_stream_processing.py`/`feddrift_detection_batch_processing.py`: 現在モデル1個で予測し、block末で保有数が変わる。順伝播標本数は分母の代わりにならない。既存HeldModelCountRecordStoreをstreamが所有する。
- `runtime/feddrift_run_server.py`: 集約upload一覧が正の標本重みで平均した実snapshotと一致。終端も同じ内部関数を利用する。
- `methods/feddrift/cached_model_cluster_consolidation.py`: 平均は0重みも単独群もmultiplyし、全0の群は均等重みにする。共通FedSDAの「正重みメンバーだけ」と異なる実演算を数える。単位は同じ。
- 同serverのprepare_timestep: server snapshot平均とは別にclientそれぞれのsnapshot平均を実行。これをserverへ数えると責務を誤るため独立fieldで保持。
- `runtime/feddrift_run_metric_derivation.py`: 基本7指標と入力対応検査を維持し、任意モデル計数で共通summaryを追加。既存testsのbasic_metric_valuesは旧との対応を維持。

## Decisions and Risks

新しい数値処理/共有counter基盤は追加しない。各成功操作に対応する4累積整数と既存共通snapshotを使う。失敗全体のrollbackを新設しない。追加記録の型検査が状態更新後に初めて落ちないように先に検査する。client統合は比較から消さず別計数で開示するが、既存Linear summaryの意味を変更しない。fable-methodと既読cc-sdd skillsに従い、仕様書群レビュー前にsource/testへ進まない。

- client平均の入力snapshotをtupleへ先に収集した。cloneだけを平均の前へ移し、client列・平均式・torch/Python/NumPy乱数の消費は維持。既存server同期/終端の対照testが全値/grad/optimizer/通信/RNGの一致を確認する。
