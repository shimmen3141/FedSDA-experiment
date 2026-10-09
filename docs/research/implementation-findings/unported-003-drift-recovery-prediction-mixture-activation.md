# UNPORTED-003: 回復期だけ混合予測を有効にする方針と、警報側の通知を新実装へ移植していない

- 記録日: 2026-10-10。対象: 旧`federated_drift_experiment/expert_routing.py::SoftRoutingActivationController`（521〜578行）の`drift_recovery`方針と、`clients/fedsda.py`の`_on_drift_alarm`・`_on_drift_resolution`（1444〜1479行）、`_use_soft_routing`（1481〜1497行）、`_record_hard_prediction`（1499〜1522行）、`_fifo_routing_loss_sequence`（1415〜1442行）。固定基準748c3aa。
- 判断したspec: observed-sample-prediction。種別: 移植範囲の判断。状態: 当面移植しない（主担当Claude Codeの判断。ユーザーが必要と判断すれば覆せる）。

## 旧の機能

混合予測（保有する全モデルの確率を重みで結合する予測）を、常時ではなく、警報から回復までの期間だけ有効にする。

- 有効化方針`drift_recovery`では、最初は無効で、現行モデルだけで予測する（`_record_hard_prediction`）。
- 警報のとき（`_on_drift_alarm`）に有効にし、その時点の保留標本で全モデルの損失を計算し直して、globalのAdaHedgeとFixed-Shareの重みへ再生する。
- 警報への応答が決まったとき（`_on_drift_resolution`。候補検証を保持していれば、その確定のとき）に、モデル集合が変わっていれば同じ標本でもう一度再生し、そこからFIFO長の標本数だけ有効を保つ。
- 期間が過ぎた後、Fixed-Shareの重みが最大のモデルが現行モデルなら無効に戻し、違えば有効を続ける。

## 移植しない根拠（確認した事実）

- 既定値は`always`（`config.py` 116行）。最終構成の回帰（`tests/test_proposed_regression.py`の`ALGORITHM`）は`soft_routing_activation_policy="always"`。
- `always`では、`_on_drift_alarm`は基底の空のhookを呼んで戻るだけである。`_on_drift_resolution`は、再生の条件が`drift_recovery`を要求するので再生せず、再生用の2属性（常に空）を空で上書きし、`resolve`は即戻る。`_use_soft_routing`は常に真で、`_record_hard_prediction`は通らない。したがって、最終構成では、警報側の通知に移植する処理がない。
- 新の設定（`PredictionCombinationSettings.prediction_mixture_activation_policy`）が受け付ける値は`always`だけである。

確認していないこと: 過去の実験成果（`results/`）に`drift_recovery`のrunがあるかどうか。旧のCLIと掃引軸には残っている。

## 新実装の現状

- 予測は、常時有効の1経路だけを持つ（`predict_observed_sample_and_update_prediction_weights`）。有効・無効の状態を持つownerはない。
- 標本1件の処理（`process_observed_sample`）に、警報側の通知の呼出しはない。
- Fixed-Shareの重みのownerは、損失の列の再生（`replay_observed_losses`）を持つ。警報時の再生には使っていない。

## 後で移植する場合

- 置く場所: 有効・無効の状態を持つownerを足し、`process_observed_sample`の、警報の位置の記録の後・警報の処理の前（旧`_on_drift_alarm`。警報の処理より前の保留標本を読む）と、警報の処理の後で候補検証を保持していないとき、および候補検証の確定のとき（旧`_on_drift_resolution`）に呼出しを足す。予測の関数に、無効のときの経路（現行モデルだけで予測し、記録の項目を旧`_record_hard_prediction`と同じ値で埋める）を足す。
- 必要な検証: 実旧の`drift_recovery`のclientを使う対照test、設定の値の追加、有効にした条件の基準（golden）の作成。

関連: [observed-sample-predictionのresearch.md](../../../.kiro/specs/observed-sample-prediction/research.md)、[observed-sample-processingのdesign.md](../../../.kiro/specs/observed-sample-processing/design.md) 2節の「予測側の通知の位置」。
