# Research & Design Decisions

## Summary

- **Feature**: `fedsda-run-client-assembly`
- **Discovery Scope**: Extension（移植済みの部品を、既存の実行の枠の契約へつなぐ）
- **Key Findings**:
  - 実行の枠の契約`RunClientOperations`の5操作は、それぞれ移植済みの部品1つに対応する。新しく書くのは、ownerの生成、値の検査、観測標本の変換だけである。
  - 旧のclientは、初期モデルを`deepcopy`するので、事前学習で蓄積したoptimizerの状態も引き継ぐ。初期モデルは、分類器だけでなく、2つのoptimizerの状態と組で渡す必要がある。
  - 旧の値のうち9つは、新の機能別の設定型に置き場所がない。うち2組は、旧では1つの値を2箇所で使う。

## Research Log

### 実行の枠の契約と、対応する旧処理・新部品

- **Sources Consulted**: `execution/run_participant_contracts.py`、`execution/stream_protocol_execution_loop.py`、`runtime/single_run_execution.py`、旧`experiment.py`（60〜93行、94〜118行、2283〜2287行）、旧`clients/base.py`（198〜212行、414〜425行）、旧`clients/fedsda.py`（359〜410行、412〜418行）。
- **Findings**:

| 契約の操作 | 旧 | 新部品 |
| --- | --- | --- |
| `process_observed_sample(observed_sample, sample_index)` | `process_one_step(x, y, concept_id)` | `process_observed_sample` |
| `flush_pending_local_updates(round_index)` | `flush_pending_updates()` | `train_held_models_for_pending_training_requests` |
| `has_model_ready_for_server_registration()` | `has_pending_model()` | `PendingModelUploadState.has_ready_model_upload` |
| `advance_new_model_upload_wait_after_synchronization(round_index)` | `promote_pending_to_ready()`（FedSDAの版） | `PendingModelUploadState.advance_upload_readiness_at_round_boundary` |
| `finalize_incomplete_candidate_validation()` | `finalize_incomplete_forward_validation()` | `finalize_held_incomplete_candidate_validation`と、概念IDの保持の解除 |

  - 契約の`process_observed_sample`は、真の概念を受け取らない。実行の枠は、clientの準備（`prepare_run`）の後に真の概念の列を作る（乱数の消費順を旧と合わせるため）。したがって、clientの生成時に真の概念の列を渡すこともできない。
  - 旧の`flush_pending_updates`・`promote_pending_to_ready`は、ラウンドの番号を受け取らない。
- **Implications**: clientの`process_observed_sample`は、契約の2引数に加えて、任意の引数として真の概念IDを受け取る（契約を満たしたまま、診断用の値を渡せる）。実行の枠からこの引数へ値を渡すかどうかは、全体runを接続するspecで決める（下の「Decision」）。

### 旧のclientの生成（最終構成のクラスの`__init__`の連鎖）

- **Sources Consulted**: 旧`clients/base.py`（24〜80行）、`clients/fedsda.py`（43〜73行、1135〜1142行、1183〜1197行、1294〜1397行）、`clients/shared_backbone.py`（33〜51行）、`experiment.py`（294〜358行）。
- **Findings**:
  - `self.models = copy.deepcopy(initial_models)`、`self.model_stats = copy.deepcopy(initial_stats)`、`current_model_id = 0`、`next_temp_id = -100 − client_id`。旧のモデルは、自分のoptimizer（概念固有部と、共有部の`backbone.optimizer`）を属性に持つので、`deepcopy`で状態ごと写る。
  - 検出器の基準は、初期統計の平均（件数が0なら0.01。上下限つき）。新の`select_loss_monitoring_baseline_mean_loss`が同じ値を返す（移植済み）。
  - FIFO長は`FIFO_BUFFER_SIZE`で、Fixed-Shareの時間尺度も同じ値。候補検証の標本数（2以上）、送信までの待ちラウンド数（1以上）は、`__init__`が確かめる。
  - 共有部の再接続（`_share_model_backbones`）は、保有モデルが1つなら何もしない。
  - 最終構成で使わない属性（検出episode、Cached用のパラメータの控え、文脈別・Metaのルータ、評価するモデルを絞る方式）は、新のownerに対応がない。
- **Implications**: 新の組立ては、初期モデル（分類器、概念固有部のoptimizerの状態、共有部のoptimizerの状態）を1回の`deepcopy`で写す（3つの間のパラメータの同一性が保たれる）。

### 置き場所のない値

- **Findings**（旧の設定名 → 標本1件の処理・ownerが受け取る引数）:

| 旧 | 新の引数 |
| --- | --- |
| `distance_threshold`（`FEDSDA_DISTANCE_THRESHOLD`） | `maximum_reference_mean_loss_increase`と`maximum_alarm_interval_mean_loss_increase`（同じ値） |
| `NEW_MODEL_EARLY_STOPPING_MIN_DELTA` | `minimum_candidate_mean_loss_improvement`と、`CandidateEpochTrainingSettings.minimum_validation_loss_decrease`（同じ値） |
| `FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS` | `upload_delay_round_count` |
| `MIN_DRIFT_DATA` | `minimum_change_interval_sample_count` |
| `CLIENT_BATCH_SIZE` | `batch_sample_count`と、`CandidateEpochTrainingSettings.maximum_batch_sample_count`（同じ値） |
| `STORED_DATA_LIMIT`、`EVAL_STORE_SAMPLE_SIZE` | `ModelEvaluationSampleStore`の生成の2引数 |
| `ADWIN_MAX_WINDOW`、検出器の賭け率の既定 | `OverallAndTrueClassLossMonitor`の生成の2引数 |
| `_detector_label()` | `detector_name` |

  - 機能別の設定型のうち、検証済みのrun設定の部分型（`ValidatedExperimentRunSettingsSubset`）に入っていないもの: `LocalTrainingScheduleSettings`、`CandidateParameterInitializationSettings`、`CandidateEpochTrainingSettings`、optimizerの設定。
- **Implications**: 下の「Decision: 値の束」。

### oracleの実行確認（REDより前）

- 2026-10-10、Windows基準環境で下書きのtest（commitしていない）を実行した。旧の設定（`config`の属性）を小さい値へ差し替え、実`_pretrain_initial_model(ResidualAdapterMLP)`で初期モデルと統計を作り、`ResidualAdapterRestartingSoftRoutingFedSDAClient(client_id=1, initial_models={0: model0}, initial_stats={0: stats0}, distance_threshold=0.1, verbose=False)`を実`__init__`で作って、サーバなしで、`process_one_step`を120標本、10標本ごとに`flush_pending_updates`と`promote_pending_to_ready`、最後に`finalize_incomplete_forward_validation`を実行できた。適応の結果は、不足・候補検証の開始・候補検証中の警報・棄却・採用・現行の維持・他モデルの再利用を通った。clientのモデルとoptimizerは、初期モデルとは別のオブジェクトだった。
- 旧の標本は、特徴が1次元（2要素）のfloat32、ラベルが1次元（1要素）のfloat32のtensorで、`process_one_step`が2次元へ直す。新の`ObservedSample`の特徴は、float32へ丸めた値のfloatなので、float32のtensorへ戻すと同じ値になる。

## Design Decisions

### Decision: 値の束を、runtimeに1つ置く

- **Context**: 置き場所のない値を、どこから受け取るか。
- **Alternatives Considered**:
  1. 既存の機能別の設定型へfieldを足し、検証済みのrun設定の部分型も広げる。
  2. clientの組立てが受け取る値の束を、runtimeに新しく置く。
- **Selected Approach**: 2。数値と文字列の値は、既存の検査の仕組み（fieldのmetadata）で宣言する型にまとめ、機能別の設定と賭け率を合わせて、clientの設定の束にする。
- **Rationale**: 1は、承認済みの設定型と、それを作っている多数のtestを変える。保存表現とpresetを決める段階（完全なrun設定のspec）で、値の正式な置き場所を決めるほうが、二度手間にならない。検証済みのrun設定の部分型も「完全なrun設定ではない」と明記している。
- **Trade-offs**: 束は、機能別の配置としては仮の形である。再開案内の「次の候補」へ、置き場所を決める課題として書く。
- **Follow-up**: 旧で1つの値を2箇所で使う3組は、束では、許容する増加量を1つのfieldにし（2つの引数へ同じ値を渡す）、最小改善量とbatchの件数は、束のfieldと候補の学習の設定が同じ値であることを確かめる。

### Decision: 真の概念IDは、clientの操作の任意の引数にする

- **Context**: 契約は真の概念を渡さないが、標本1件の処理は診断用に受け取る。
- **Selected Approach**: `process_observed_sample(*, observed_sample, sample_index, evaluation_concept_id=None)`。実行の枠は変更しない。
- **Rationale**: 契約の呼び方（2引数）のままで動き、対照testと将来の配線は、3つめの引数で概念IDを渡せる。実行の枠の契約を変えるかどうかは、全体runのgoldenの比較項目（概念別の計数や診断を比べるか）で決まるので、全体runを接続するspecで決める。
- **Follow-up**: 再開案内へ書く。渡さない場合、概念別の診断と、モデル別の割当概念の計数が行われない。

### Decision: 初期モデルは、分類器と2つのoptimizerの状態の組で受け取り、1回の`deepcopy`で写す

- **Context**: 旧は、事前学習の後のoptimizerの状態を、各clientが引き継ぐ。
- **Selected Approach**: 組立ては、3つを同時に`deepcopy`して、写しを保有モデルとして登録する。渡された3つの対応（optimizerのパラメータが、分類器の概念固有部・共有部と同一で同じ順）を、写す前に確かめる。
- **Rationale**: 同時に写すと、写しの中でも、optimizerが写しの分類器のパラメータを指す。

### Synthesis

- **Generalization**: 5つの操作は、同じowner一式への薄い呼出しなので、ownerを1つの不変の記録にまとめ、操作はそれを読むだけにする。
- **Build vs. Adopt**: 新しく書く判断はない。値の検査は、既存のmetadataの仕組みを使う。
- **Simplification**: 方式を差し替える抽象、サーバ向けの操作、準備のfactoryは作らない。

## Risks & Mitigations

- 束の検査の範囲が、値を使う部品の検査より狭いと、処理の途中で拒否される — 各値について、使う部品の検査（型と範囲）を読んで、同じか狭い範囲にする。
- 対照のoracleが、旧の設定の差し替え漏れで、新と違う条件になる — 生成直後の全状態の照合で検出する。差し替える設定の一覧をtestの1箇所にまとめる。

## References

- [observed-sample-processingのdesign.md](../observed-sample-processing/design.md) 4節 — 「確かめないこと」（設定と、ownerどうしの整合は、組立ての役目）。
- [single-run-executionのspec](../single-run-execution/) — 実行の枠の契約。
