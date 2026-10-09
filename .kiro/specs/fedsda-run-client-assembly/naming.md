# 命名表: fedsda-run-client-assembly

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `fedsda_run_client_settings` | module（runtime） | clientの組立てが受け取る設定と値の束 | `configuration/run_settings`は、run全体の検証済みの設定（部分型）。本moduleの束は、client 1つの組立てに要る値で、置き場所のない値を含む仮の形 |
| `FedsdaRunClientScalarSettings` | frozen dataclass | 機能別の設定型に置き場所がまだない、数値と文字列の値（許容する増加量、最小改善量、待ちラウンド数、最小件数、batchの件数、評価標本の件数、検出器の候補数の上限、検出器の表示名） | 機能別の設定型（`LossChangeDetectionSettings`など）は、1つの機能の値だけを持つ。この型は、複数の機能にまたがる値をまとめる |
| `FedsdaRunClientSettings` | frozen dataclass | 機能別の設定9つ、`FedsdaRunClientScalarSettings`、検出器の賭け率。組合せの検査を持つ | `ValidatedExperimentRunSettingsSubset`はrunの条件と機能別の設定の部分で、clientの生成に足りない |
| `fedsda_run_client` | module（runtime） | clientと、その組立て | `observed_sample_processing`は標本1件の処理（clientが呼ぶ） |
| `FedsdaRunClientOwners` | frozen dataclass | clientが持つ全ownerへの参照（標本1件の処理が受け取る18個と、共有部のoptimizerの状態）。状態はowner自身が持つ | `RunParticipants`は、実行の枠へ渡す参加者（clientの操作とサーバの操作） |
| `FedsdaRunClient` | class | 最終構成のFedSDAのclient 1つ。実行の枠の契約`RunClientOperations`の5操作を提供し、ownerの記録と束を持つ。判断は持たない | `RunClientOperations`は契約（Protocol）、本classはその実体 |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `assemble_fedsda_run_client` | client ID、初期モデルのID、初期の分類器と2つのoptimizerの状態、初期の損失統計、束、乱数生成器 | `FedsdaRunClient` | 全部の検査の後、初期モデルを写し、ownerを作って、clientを返す | なし（渡されたものを変えない。乱数を使わない） |
| `FedsdaRunClient.process_observed_sample` | 観測標本、位置、任意で真の概念ID | `ObservedSampleProcessing` | 観測標本を1行のtensorへ変換して、標本1件の処理を呼ぶ | 標本1件の処理が更新する全owner |
| `FedsdaRunClient.flush_pending_local_updates` | ラウンドの番号 | 完了した共同更新の損失 | 保留中の学習要求を学習する | 保有モデル、optimizer、計数、学習要求、乱数 |
| `FedsdaRunClient.has_model_ready_for_server_registration` | なし | bool | 送信保留のモデルが、送信できる状態かを返す | なし |
| `FedsdaRunClient.advance_new_model_upload_wait_after_synchronization` | ラウンドの番号 | なし | 送信保留のモデルの待ちを1ラウンド進める | 送信保留 |
| `FedsdaRunClient.finalize_incomplete_candidate_validation` | なし | 回収の結果またはNone | 保持中の未完了の候補検証を回収し、概念IDの保持を外す | 候補検証の保持、適応記録、学習データ、損失統計、計数、保留標本のownerの概念ID |
| `FedsdaRunClient.client_id`、`FedsdaRunClient.owners` | なし（property） | int、`FedsdaRunClientOwners` | 実行の枠の検査と、評価・testが読む | なし |

## 判断が必要な点

- 操作の5つの名前は、実行の枠の契約（single-run-executionで承認済み）の名前をそのまま使う。
- 束の2つの型は「Settings」で終える（既存の設定型と同じ語）。値の型は「Scalar」で、数値と文字列の単独の値であることを表す。
- `maximum_tolerated_mean_loss_increase`は、旧の`distance_threshold`（何の距離かが名前から分からない）に当たる。標本1件の処理の2つの引数（参照モデルの損失の増加の許容、警報区間の再利用評価の損失の増加の許容）へ同じ値を渡す。
