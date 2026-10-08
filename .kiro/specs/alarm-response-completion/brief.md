# 警報応答の完了処理

旧`FedSDAClient._resolve_drift`は、標本処理と応答選択（完了済みの`respond_to_alarm_with_buffered_samples`）の後に、適応イベントの記録→検出器のreset→FIFOのclear（変化区間が不足のときはclearしない）を行い、戻り値でモデル操作の有無を返す。本specはこの後半を扱う。

再開案内の候補「警報応答後の後始末・記録・通知」を次の2つに分ける。

- 本spec: 完了した応答を受け、既存ownerだけで実行できる後始末（損失監視の再開、保留位置の消費）を行い、呼出側の記録・通知・session保持に必要な情報を不変recordで返す。
- 後続spec: 適応イベント一覧・切替位置一覧・再利用計数・検出episodeのowner、学習帰属変更の通知（予測重み）、候補検証sessionの保持と解除。これらは新実装にownerがまだなく、候補検証の到達時・未完了回収の完了情報（完了済みの2spec）とも共通に使うため、警報応答だけの都合で先に形を決めない。

既存の`PostAlarmCandidateValidationCompletion`（候補検証の到達時）と同じく、一覧への記録は行わず、記録に必要な値とpropertyを返す形にそろえる。旧の不足時FIFO保持（LEGACY-002）は修正しない。全体runの接続前の部品spec。固定旧`748c3aa`とgoldenを変更しない。
