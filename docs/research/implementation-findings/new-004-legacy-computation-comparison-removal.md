# NEW-004: 旧実装の計算量との照合は、旧実装を外すときに取り除く（残すものと、置き換えるもの）

- 記録日: 2026-10-10（ユーザーの指示で、文脈を理解しているうちに記録）
- 記録したspec: [model-computation-measurement](../../../.kiro/specs/model-computation-measurement/tasks.md)
- 種別と状態: 後でやる作業の記録・未着手。旧実装を外す段階（リファクタリングの最終段階）で行う。時期と方法は、ユーザーが決める。
- 基準commit: `65c3bf4`（model-computation-measurementの完了）

## 前提（なぜ、いまは残しているか）

旧実装（`federated_drift_experiment/`。固定）は、計算量を、clientが処理の各所で足す計数（`compute_counters`）で測る。単位は「モデルへ入力した標本の数」と「更新の回数」で、モデルの大きさを反映しない。

2026-10-10のユーザー決定で、新実装は、手法の比較のために、全結合層の積和演算の数を測る（`learning/models/model_computation_measurement.py`。実行中に、外側から数える）。旧と同じ件数（標本数の計数）も、同じ計測で得られる。

旧の計数との照合は、次の2つのために、いまは残している。

1. **新実装が、旧と同じ量の計算をしていることの確認。** 実際に、新実装は3箇所で、旧より多く順伝播を行っていた（結果は同じ）。照合を足して見つけ、解消した。
2. **部品を変えたときの検出。** 新旧を同じ条件で実行するtestのhelperが、操作ごとに、新の計算量と、旧の計数の増分を比べる。部品を変えて順伝播が増減すると、ここで分かる。

旧の計数そのものは、正確だった（goldenの3ケースで、外側からの数え直しと完全に一致）。

## 旧実装に依存しているもの（旧実装を外すときに、取り除く・置き換える）

| もの | 場所 | 旧実装を外すとき |
|---|---|---|
| 旧の計数の検査（旧の計数＝外側からの数え直し） | `tests/refactoring/test_legacy_computation_count_audit.py` | 取り除く（旧実装だけを調べるtest。役目は終わっている） |
| 操作ごとの、新の計算量と、旧の計数の増分の照合 | `tests/refactoring/test_fedsda_run_client.py`の`run_in_both`・`assert_model_computation_matches_legacy_counter_increase`・`sum_legacy_computation_counts`。複数clientの操作は、4つのtest fileが`legacy_clients=legacy_server.clients`を渡す | 取り除く。**代わりの検出の網を先に用意する**（下の「置き換え」） |
| clientの検出器の計数と、旧の計数の照合 | 同ファイルの`assert_run_client_matches_legacy`（`drift_detector_updates`・`drift_detector_hypotheses`） | 取り除く（clientの全状態の新旧照合と一緒に） |
| goldenの条件・小さい条件での、計算量の7項目の照合 | `tests/refactoring/test_fedsda_run_metric_derivation.py`（`derive_legacy_metric_values`の`compute_*`の対応、旧の`_add_telemetry_results`との照合、Windows用のgoldenとの照合） | 置き換える（下の「置き換え」） |
| goldenの計算量の7項目 | `tests/proposed_regression_golden.json`（`compute_inference_examples_total`ほか）、`tests/test_proposed_regression.py` | 旧の回帰testと一緒に取り除く。新実装の値を基準にしたgoldenへ置き換える |

## 残すもの（旧実装を外した後も、新実装の機能として使う）

- 計測そのもの（`measure_model_computation`、`ModelComputationCounts`）。積和演算の数が、手法の比較の主指標。
- **標本数の計数**（`ModelComputationCounts`の`*_example_count`）。旧と同じ件数だが、名前は新実装の語で、旧実装に依存しない。積和演算の数は「標本数×層の形」で決まるので、内訳の説明に使える。保持の負担は小さい。
- 検出器の計算の計数（`LossMonitoringComputationCountStore`）、計測つきの全体run、指標の導出の計算量の項目。
- 計測のtest（`tests/refactoring/test_model_computation_measurement.py`。手計算との照合で、旧実装に依存しない）。

## 置き換え（旧実装を外す前に、用意するもの）

旧の計数との照合を取り除くと、「部品を変えたときに、計算量が変わっていないこと」を確かめるものがなくなる。取り除く前に、次のどちらか（または両方）を用意する。

1. **新実装の計算量を基準にしたgolden。** goldenの条件（とdatasetごとの代表条件）で、新実装の計算量（標本数、積和演算の順伝播と逆伝播、optimizerの更新回数、検出器の計数）を固定値として保存し、回帰testで比べる。名前は、新実装の語にする。旧の7項目との対応は、`derive_legacy_metric_values`（上のtest）が持っているので、置き換えの時点で、旧の値と一致していることを、最後に1回確かめてから、対応表を捨てる。
2. **操作ごとの期待値。** 代表的な操作（標本1件の処理、警報、候補の学習、クロス評価、再較正、配布）について、順伝播の標本数を、処理の定義から計算した値と比べるtest。旧の計数との照合が見つけた重複（クロス評価、警報時の区間の準備、集約後の再較正）は、この形のtestで守れる。警報時の区間の準備と、有界損失の評価の入口には、すでに、この形のtestがある（順伝播の標本数を、直接の値で確かめる）。

## 判断が必要な点（未定）

- 取り除く時期。旧実装を外す段階でよいか、それより前に、計算量の照合だけを先に外すか。主担当の考え: 旧実装を外す段階でよい（それまでは、検出の網として役に立つ）。
- goldenの置き換えの単位。計算量だけを先に新しいgoldenへ移すか、全指標を一度に移すか。
- 旧の用途別の内訳（予測・検出・統計・クロス評価・初期化・再較正）は、新実装では作っていない。旧実装を外す前に、要るかどうかを確かめる（要るなら、計測の区間を、処理の単位で区切る形で足せる）。

## 関連

- 調査の記録（旧の計数の検査の結果、新実装の3箇所の重複）: [model-computation-measurementのrequirements.md・research.md](../../../.kiro/specs/model-computation-measurement/)
- 計算量の測り方の決定（積和演算、逆伝播の見積り、サーバの演算数、保有モデル数での正規化）: 再開案内（`.kiro/steering/resume.md`）の「ユーザー確認待ち・未解消の事項」

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

保留（旧実装を外す準備のときに行う）。変更なし。

## 旧実装を外すときに、一緒に外すもの（2026-10-13追記）

旧実装との照合を続けるために、挙動を直す変更に、旧と同じ扱いを設定で残している。旧実装を外すときに、値と分岐を外す。

| 修正 | 残している値 | 外す箇所 |
|---|---|---|
| LEGACY-016（[pending-model-registration-by-own-id](../../../.kiro/specs/pending-model-registration-by-own-id/design.md)） | `keep_latest_pending_model_and_reassign_current_model_id`（設定`PendingModelRegistrationSettings.pending_model_registration_policy`） | `methods/fedsda/model_registration/pending_model_registration_settings.py`の値、`pending_model_upload.py`の`_append_or_replace_pending_model_upload`の置換えの分岐、`runtime/held_model_registration_confirmation.py`の、現行を付け替える分岐。testの`use_pending_model_registration_policy_matching_legacy`・`assert_pending_model_registration_policy_matches_legacy`、基準版1のgolden（旧と同じ扱いで照合している）。**FedDriftのclientが、この値を使っている**（`runtime/feddrift_run_client.py`）。外す前に、FedDriftを、もう一方の値へ替えても結果が同じであることを確かめる（FedDriftでは、確認のとき、送信保留のモデルが常に現行なので、同じになる見込み。未検証） |
| LEGACY-002（[too-short-alarm-earlier-sample-release](../../../.kiro/specs/too-short-alarm-earlier-sample-release/design.md)） | `keep_all_pending_samples`（設定`TrainingDataAssignmentSettings.pending_sample_retention_after_too_short_change_interval_policy`） | `methods/fedsda/training_data_assignment/training_data_assignment_settings.py`の項目と値（項目ごと外せる）、`pending_training_assignment_buffer.py`の`releases_earlier_samples_after_too_short_change_interval`、`runtime/alarm_response_completion.py`の、短い警報で何も外さない分岐。testの`use_legacy_compatible_policies`・`assert_run_client_uses_legacy_compatible_policies`、基準版1・2のgolden（旧と同じ扱いで照合している） |

## 旧実装を外したあとに直す候補（2026-10-13、ユーザー決定）

診断だけの修正で、古い基準版のgoldenの記録の列を変えるため、旧実装を外すまで待つもの。旧実装と、旧と同じ扱いの設定・古い基準版の照合を外したら、着手する。

- [LEGACY-014](legacy-014-adopted-model-pending-samples-not-counted.md)の割当概念計数: 採用のとき、保留標本の真の概念を数える。
- [LEGACY-001](legacy-001-candidate-decision-reason-rounding.md): 採否の演算を正本にし、区間ごとの合否から理由を作る。

