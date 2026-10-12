# 公開名と役割

| 公開名 | 役割と似た名前との差 |
|---|---|
| runtime/feddrift_measured_run_execution.py | FedDrift全体runのモデル計算の計測接続。通常入口や指標導出とは別。 |
| FeddriftTimestepModelComputationCounts | 1時刻の標本処理と同期の計数。FedSDAの1ラウンドとは異なり同期にR回の学習を含む。 |
| FeddriftMeasuredRun | FedDriftの実run/参加者と準備・本実行・時刻別・終端のモデル計数。サーバ算術summaryではない。 |
| execute_feddrift_stream_protocol_run_with_computation_measurement | 共通hook区間で全体runを実行する入口。通常runの無計測入口と区別する。 |

既存公開名は役割もsignatureも変更しない。引数/field/非公開名/test名は本表の対象外。
