# Design Document

## Overview
runtimeにFedDrift専用の全体run入口と指標導出を置く。標本生成・概念スケジュール・CPU torch隔離・成功イベント/例外・結果・評価計算は共通部品を再利用する。FedSDAのclient/server操作Protocolを偽装しない。

## Architecture & Components
- `FeddriftStreamProtocolExecutionSettings`はrun条件とランダム概念スケジュールを保持し、exact型と既存field宣言を再検査する。FedDriftの固定順はclient順の検出block→R同期であり、FedSDA逐次戦略の文字列は流用しない。
- factoryに`validate_run_conditions`を追加し、生成前にrun条件と検出chunkの一致を検査する。prepare_runも同じ検査を呼び、重複条件を散在させない。
- `execute_feddrift_stream_protocol_run`はexact factoryを検査し、局所Random/RandomStateと既存CPU torch scopeを用いる。既存invoke wrapperで成功と元causeを記録し、準備したexact bundleのclient数/ID/同じserver参照を検査する。全標本をclient順に生成した後、検出chunkごとに全client順のblock観測後、既存server.synchronize_timestepを1回呼ぶ。同操作が時刻準備→server設定local_training_round_count回の「全client順local_train相当（検出件数×updates_per_sample）→server学習round」を内包する。全体runnerへこの内側loopを複製しない。最後の処理時刻（0 roundなら0）でfinalize_observed_streamを呼ぶ。
- 共通イベントに`observed_stream_finalization`を加える。観測blockのイベントはclient/先頭sample/roundを持つ。block成功はsample_processing、通常同期成功はserver_synchronization、終端操作全体（flushと条件付き最終回収/集約/配布）成功はobserved_stream_finalizationを記録する。通常同期/終端の内部操作に個別イベントは付けず、wrapper呼出しがreturnした時だけ記録する。途中例外は共通cause/位置を保持し、結果を返さず、内部まで成功イベントへ偽装しない。終端roundは最後の処理時刻（0 roundは0）、sample位置はNoneとする。既存FedSDAのイベント/Protocol/実行順を変えない。
- 戻り値は既存`StreamProtocolRunResult`。準備済みbundleはfactoryの既存propertyから参照する。
- `derive_feddrift_run_metrics`は共通`RunMetricSettings`、`DetectionMetrics`、`CommunicationVolumeSnapshot`を使用する。精度/定常精度/切替検出は同じ純粋関数へ渡す。独立モデルでは共有部を別途数えず、repository全modelのparameterを数える。FedSDA固有のfieldや偽の0/NaNを追加しない。
- 導出はrun recordとbundleを再検査し、client IDs/件数、概念/観測長、生成/処理/末尾の対応、client予測件数/位置/ラベル/概念の対応を確認する。概念全体から変更位置を導き、実処理部分の正誤と切替記録へ共通計算を適用する。

## Traceability
| 要求 | 実装/検証 |
|---|---|
| 1.1,1.2,1.3 | 共通bootstrap部品→FedDrift loop→終端。実旧全体runと全状態/3RNG、未処理末尾/0 round/再実行対照 |
| 2.1,2.2 | 設定/互換性preflight、共通invoke、scope全出口復元。入力拒否と各段階の注入失敗 |
| 3.1,3.2 | 共通評価関数を使う読取り導出。実旧基本指標と不整合拒否/無変化対照 |
| 4.1,4.2,4.3 | actual旧全体run observer、Linux再現性と別golden、共用fresh/exact依存 |

## Differences from Legacy
旧はglobal乱数をseedするが、新はrun局所乱数とCPU torch隔離を使う。成功時は旧の順と数値を維持する。契約外入力は準備前に拒否する。旧の`chunk未満`は正常に0 roundで終わる（ready無しなら未定義tを参照しない）。未観測末尾をflushで追加処理しない。

## Test Strategy
actual旧全体runのsetup/概念列/標本生成をobserverで捕捉する。式や進行をoracleへ複製しない。既存make_inputs/assert_server_equal/RNG helperを再利用する。synthetic/MNIST、多class、SGD/Adam、threshold/round/seed/確率、未処理末尾/0 roundで全状態と指標を照合。各実行段階の失敗とprepared集合保持/乱数復元、指標input不整合と読取り性を検証する。Linuxは別process旧2回の一致を確認した条件だけ別fixtureで固定し、同じ条件の新runを照合する。全pytestは独立実装レビュー後に固定環境で実施する。

prepare_runの契約外入力では、chunk対応検査を乱数源/生成器の型検査より前へ移した。全体runとfactoryで同じpreflightを再利用し、正常入力での数値/消費順は変えない。
