# Design Document
## Overview
FedDriftの実同期の成功出口でモデル数を保存し、既存ownerの学習計数・クラスタ対から読取り専用の診断を導出する。runtimeのFedSDA型は借りず、evaluationの既存共通算式を再利用する。

## Boundary Commitments
- evaluation/feddrift_model_diagnostic_records.py: 不変の記録型のみ。
- runtime/feddrift_run_server.py: 同期モデル数列をsole ownerとして私有listで保持。
- runtime/feddrift_model_diagnostic_metric_derivation.py: 既存ownerと系列を読取り、FedDrift診断を作る。
- runtime/feddrift_run_metric_derivation.py: 既存の対応検査を維持し、モデル診断を接続。

## Out of Boundary
共通算式・FedSDA source/typeの変更、相補性用forwardの追加、変更応答/予測診断、保存/CLI、旧alias・旧読込み・golden更新は行わない。

## Allowed Dependencies
新記録はdataclassesのみ。新導出はdataclasses、numpy、共通model_diagnostic_metric_calculations、既存の型付きowner（registry/training sample/counts/clustering）、新記録、FeddriftRunClient/Serverへ依存。サーバは新記録だけを追加。run指標は新導出の公開型・関数だけを追加。sourceで使うsymbolのみexact依存境界へ登録する。

## Revalidation Triggers
計数の時点、モデルIDの集計単位、AUC/空値の定義、公開名、task境界の変更は影響範囲の仕様再レビューを行う。

## Interfaces and Data Models
- FeddriftSynchronizedTimestepModelCounts: timestep_index（非負builtin int）、global_model_count（正builtin int）、client_held_model_counts（非空tuple、正builtin int）。__post_init__で検査。
- FeddriftClientHeldModelTrainingAmount: client_id、model_id（負の一時IDも許す）、stored_training_sample_count、trained_sample_count、parameter_update_step_count。IDはbuiltin int、client非負、計数は非負builtin int、__post_init__で検査。
- FeddriftRunServer.snapshot_synchronized_timestep_model_counts() -> tuple[FeddriftSynchronizedTimestepModelCounts,...]: immutable snapshot。型/値/client件数を読取り時にも検査。
- FeddriftModelDiagnosticMetrics: 同期記録tuple、global数の平均/最大/延べ合計、clientモデル別学習量tupleと3種類のValueDistributionSummary、真概念既知/同概念対数、共通ClusterAssignmentTrueConceptAgreement、2距離のAUC。
- derive_feddrift_model_diagnostic_metrics(*,run_clients:tuple[FeddriftRunClient,...],run_server:FeddriftRunServer) -> FeddriftModelDiagnosticMetrics。
- FeddriftRunMetrics.model_diagnostic_metricsへ結果を接続。既存公開API/基本値/計算量定義は保持。

## Update and Validation Order
synchronize_timestepの既存準備→R学習/集約/配布が全て成功した出口だけで、repositoryのglobal ID件数と各clientの実held registry件数を記録。get_held_model_ids（統計/標本のみのIDも含む）は数えない。保存列の読取り検査を既存operation入口にも入れ、破損記録を状態変更前に拒否する。途中失敗の既存部分更新はrollbackせず、新系列は追加しない。初期/終端は追加なし。

導出はexact client tuple（非空・ID一意・server.run_clientsと同じ実体/順）、server型を先に検査。触れるownerのexact型、held model IDと各計数dictのID/非負builtin int、pair記録のexact型/__post_init__を使用前に検査する。classifier/optimizerの無関係な全検査は追加しない。モデルごとのsnapshot・集計は新辞書/tupleへ行う。保有されない統計のみのIDを学習量に含めない。負IDも従来FedSDAと同じmodel ID単位でまとめる。

pairは全記録を再検査してから、true_concepts_matchがNoneでないものを共通算式へ渡す。無限距離は共通AUCで除外、どちらかの有限列が空ならNone。空のagreementは0。正誤比較が未実施のクロス評価へ偽の相補性を付けない。

run指標で同期記録のtimestep_indexが0..synchronization_interval_count-1、列長が同期数、各client件数がrun参加者数と対応することを検査。standalone診断は任意の有効な時刻列を許す（public server操作は同じ時刻を再実行できる）。同期recordは実同期イベントごとにappendする。

## Requirements Traceability and Testing
| 要求 | 証拠 |
|---|---|
| 1.1, 1.2, 1.3 | 実同期直後のrepository/registry観測と実旧telemetry、R/失敗/終端/0同期 |
| 2.1, 2.2 | 実旧experimentのモデル学習量summary、統合/一時ID/欠落計数 |
| 2.3 | 実旧clustering_oracle_diagnostic_summaryと共通AUC契約、未知/空/同率/inf |
| 3.1, 3.2 | 既存whole runへの接続、破損型/整数/位置/件数、state/RNG不変 |
| 3.3 | 共用fresh、exact境界、独立レビュー、全pytest/品質/names/progress/identity/旧golden |
