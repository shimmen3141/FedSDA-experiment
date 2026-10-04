# 命名: サーバ向け損失平均集約
revision: 2

serverは用途を表す。aggregateは件数・平均の集約を意味し、分散/クラスをpoolingするmergeではない。

| 名前 | 役割・型・単位・状態 |
|---|---|
| server_loss_mean_aggregation.py | サーバ用途の損失平均計算。モデルパラメータ集約とは別 |
| aggregate_participating_client_loss_means | keyword-only既存moments tupleを受け、件数加重平均のmomentsまたはNoneを返す。永続状態なし |
| participating_client_loss_moments | 上位が参加条件で選別したクライアント順の全体集計tuple。学習データ件数は含まない |
| _copy_validated_participating_client_loss_moments | 全入力検査・publicconstructorコピー、同tupleを返すhelper。加算前に完了 |
| copied_participating_client_loss_moments | helper内の検査済み組立list→tuple。単一集計ではない |
| validated_participating_client_loss_moments | 主関数の検査済み独立tuple |
| loss_moments | 現クライアントのexactBoundedLossMoments。全体の統計 |
| copied_loss_moments | 一クライアントのpublicconstructorコピー |
| total_observed_loss_count | 観測件数の合計int。FedAvgのtraining sample countとは異なる |
| weighted_mean_sum | 平均×観測件数の入力順float合計 |
| aggregated_mean_loss | 正の合計件数で割ったサーバ用float平均 |
| validation_error | 入力再検査のTypeError/ValueError。項目名を添える |
| aggregation_error | 算術または算出結果検査の失敗。入力非変更 |
| build_loss_moments_test_seed | test-only正常moments seed |
| run_legacy_server_loss_statistics_aggregation | test-only両旧サーバupdateをモデル実体なしstubで実行 |
| convert_loss_moments_to_legacy_statistics | test-only n/mean/M2へ明示変換 |
| assert_aggregated_loss_moments_match_legacy_statistics | test-only結果全fieldまたはNone時の既存保持を照合 |
| build_valid_server_loss_mean_aggregation_inputs | test-only後から一箇所を壊すための正常tuple |

既存BoundedLossMoments/observed_loss_count/mean_loss/sum_squared_loss_deviationsを維持する。
testのclient/modelID/gate/旧record/local snapshot等は旧oracle・上流specの既存名で明示して、production引数へ取り込まない。

## task1開始前のtest-only追加
run_legacy_server_loss_statistics_aggregation(*,clients,operation,global_stats=None,model_id=0)は旧client stub列と旧unbound update関数を受け、既存server統計を含むstubへ実行してglobal_statsを返す。
clients/client/participating_clientsは旧client列・単一client・選別済みclient、operationは両旧update関数、global_stats/model_idは旧統計辞書/対象ID。
legacy_server/legacy_statisticsは旧stub/旧結果、aggregated_loss_momentsは新純関数のmomentsまたはNone。
test_server_loss_mean_aggregation_matches_legacy_serversは両旧サーバと全fieldを比較する正常テスト。
