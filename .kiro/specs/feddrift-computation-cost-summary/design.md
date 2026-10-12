# Design Document

## Overview and Boundary Commitments

FedDriftの既存streamとサーバ操作へ記録を足し、既存derive_feddrift_run_metricsへ共通summaryを接続する。共通summary/model hookや手法の数値処理を変更しない。client統合のパラメータMACはFedDrift固有の追加計数とし、共通summaryのclient Linear計算と区別する。

## Architecture and Interfaces

- `FeddriftDetectionBatchProcessor.held_model_count`: 実HeldModelTrainingStateRegistryのordered snapshot件数を返す読取り専用property。評価標本/統計だけのIDではなく実モデルの数。
- `FeddriftClientStreamProcessor.held_model_count_record_store`: 既存HeldModelCountRecordStoreをstreamが一つ所有する。予測列と同じ標本粒度で保存し、client/ownersへ別の正本を作らない。
- `FeddriftRunServer.snapshot_server_computation_counts() -> ServerComputationCounts`: サーバの私有累積3整数の不変snapshotを共通型で返す。
- `FeddriftRunServer.client_consolidation_parameter_multiply_accumulate_count -> int`: serverが進行を担当するclientの平均の累積MAC。読取り時にexact非負intを検査。
- `derive_feddrift_run_metrics(..., model_computation_counts: ModelComputationCounts | None = None)`: 既存引数に任意の計数を追加。既存基本fieldに、server_computation_counts/processed_sample_count/held_model_sample_count/client_consolidation_parameter_multiply_accumulate_count/model_computation_counts/computation_cost_summaryを追加。

記録fieldはnaming表の対象外。設定/実行/既存公開名の役割は変更しない。

## 更新順と検査

### 保有数

streamの既存block検査後、標本ごとにheld_model_countを予測前に読み、既存予測・buffer・検出flushを実行する。予測記録をappendする成功地点で保有数もappendする。標本ごとに正のintの事前検査をその標本の状態更新前に行い、追加記録のconstructor/append検査が初めて更新後に失敗しないようにする。flush失敗時の既存processed_count/bufferの非atomicな挙動は変更せず、予測/保有数の記録には失敗標本を含めない。

### 集約

`_aggregate_and_distribute_models`が既に作る`uploads`一覧は、実際に加重和へ入った正の重みのsnapshotと同じ。この一覧の全numelを増分にして、配布が成功した出口で私有aggregation計数へ足す。準備の初期snapshotコピーは対象外。終端も同じ関数を使う。

### 統合と診断

`prepare_timestep`の実統合がある分岐だけで、既存`parameters`（全global IDをall_clustersに含む）のnumel合計をサーバ統合増分とする。既存average関数は0重みもsingletonも掛けて足すので、値が0でも演算を数える。client自身のparameter snapshotを既存と同じ順に一度作り、その一覧の全numelをclient統合増分とする。平均/配布/repository反映の成功地点でそれぞれの累積へ足す。private integer追加で平均式/乱数/順序を変更しない。

`_record_clustering`では既存distance計算済みの対数×3×全独立モデルparameter数を、観測record appendの成功出口で診断累積へ足す。全モデルのschema一致は既存サーバ事前検査で保証される。対が空なら0、クロス評価のLinear計数は既存モデルhookへ委譲する。

操作入口の既存_validate_operationでserver/clientの累積snapshotを先に検査し、手動でbool/負数へ壊したcounterも加算前に拒否する。仕様レビュー担当もこの検査位置を確認済み。既存の一部commit後に下流が失敗した場合のrollbackを新設しない。各計数は対応する成功操作の境界でcommitし、事前検査失敗時は変更しない。全run失敗でsummaryを成功結果として返さない。

### 指標

既存のresult/participant/予測/transitionの対応検査に、各streamのexact HeldModelCountRecordStore・列長==processed・全要素exact正intを追加。モデル計数はexact型/全field非負intを、summary算式へ渡す前に共通summarize_computation_costで検査する。None時は同関数を呼ばない。サーバsnapshotの共通型は__post_init__で全計数を検査。client統合は別fieldなのでserverの合計へ加えない。導出は状態/乱数を変更しない。

`ComputationCostSummary`はclientのLinear順伝播/逆伝播見積りとサーバの集約/統合を表す既存定義を保持する。clientパラメータ統合を含む全client算術の総計という意味へ変更しない。モデル計数を引数だけで受ける契約は既存FedSDAと同じで、別runの計数を認証する新APIは設けない。

## Requirements Traceability and Testing

| 要求 | 実装/証拠 |
|---|---|
| 1.1–1.3 | processor保有数/store、block順と失敗/作成/統合の各観測前の実registry件数を独立照合 |
| 2.1 | uploadの正重み/0/初期コピー/R/終端、実旧状態一致 |
| 2.2–2.4 | server/client別の実平均呼出しを観測、全0/0メンバー/singleton/隔離/統合なし、診断対の3倍算式 |
| 2.5 | 事前検査失敗の計数snapshot、全state/RNG/通信の実旧照合 |
| 3.1–3.3 | 不正型/件数/値、Noneと0/NaN、既存基本指標・計測済みrunとの共通summary算式、読取り性 |
| 3.4 | 共用fresh、exact importだけの依存境界、Linux別golden、全pytest/品質/names/progress/identity |

実旧oracleの既存pytestをbootstrapしてからtest-firstでtaskを進める。新公開methodはstubのNotImplementedErrorからREDを確認し、未import/collection失敗をREDへ数えない。独立計数は実操作の引数snapshot/標本処理前registryに基づき、出力した計数自身をoracleにしない。正常全状態照合では追加fieldを旧へ無理に対応付けず、既存basic_metric_valuesと旧goldenを維持する。
