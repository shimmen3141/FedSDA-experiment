# 標本1件の処理 — 設計 revision4

## 1. Boundary Commitments

- runtimeの新module `observed_sample_processing.py`: 関数`process_observed_sample`と結果record `ObservedSampleProcessing`。既存の部品を旧の標本処理と同じ順に呼ぶ。
- `methods/fedsda/training_data_assignment/pending_sample_observation_store.py`: `PendingSampleObservationStore`。保留中の標本そのものと、候補検証へ渡した標本の概念IDを持つ。
- `evaluation/loss_change_alarm_record_store.py`: `LossChangeAlarmRecordStore`と`LossChangeAlarmRecordSnapshot`。監視の値の列と、警報ごとの3つの位置を持つ。
- 既存のmoduleは変更しない。

Out of Boundary: 予測と予測側の警報の通知、検出episode、計算量と所要時間、標本ごとの結果種別の列、run終端・ラウンド境界・サーバ同期の呼出し、新client・全体run。

## 2. 旧処理との対応（`federated_drift_experiment/clients/fedsda.py::process_one_step`）

| 旧 | 本spec |
| --- | --- |
| `x_in.unsqueeze(0)`（1次元の入力を2次元へ直す） | なし。標本は、特徴が1行の2次元、ラベルが1行1列のtensorで受け取る（1次元は拒否する。下の「旧と違う点」） |
| `idx = self.processed_samples`、加算 | 位置は標本（`IndexedObservedTrainingSample.sample_index`）が持つ。保留位置のownerと監視の「次の位置」であることを確かめる |
| `_record_prediction` | 範囲外（予測の接続のspec） |
| `_observe_forward_validation(x, y, idx)` | `advance_held_candidate_validation`。確定したら概念IDの保持を外す |
| `error = self.models[self.current_model_id].get_absolute_error(x, y)` | 現在のモデルで`evaluate_classifier_per_sample_bounded_losses`（1標本） |
| `_update_drift_detectors(error, y, idx)`（基準は検出器の生成・reset時に履歴統計から選ぶ。新しいクラスの検出器は、その時点の履歴統計から選ぶ） | 標本ごとに`select_loss_monitoring_baseline_mean_loss`で現在のモデルの履歴統計から基準を選び、`observe_loss_after_label_observation`へ渡す。監視は、渡された基準を、新しいクラスの検出器を作るときにだけ使う（全体の検出器の基準は、警報の処理のresetが決める）。使われる時点と値は旧と同じになる |
| `history_detector_log_e.append(...)` | `append_monitored_log_e_value` |
| `self.buffer.append((x, y, concept_id))` | 保留位置のownerと保留標本のownerへ足す |
| `_forced_drift_check(idx)` | なし（最終構成の検出器では常に偽） |
| 警報: `flush_pending_updates()` | `train_held_models_for_pending_training_requests` |
| 警報: `detected_event_positions`・`estimated_drift_start_positions`・`detector_candidate_start_positions`へ追加 | `append_alarm_record` |
| 警報: `_on_drift_alarm(idx)` | 範囲外（予測側の通知。下の「予測側の通知の位置」） |
| 警報: `detection_episodes.observe_detection(idx)`、`mark_operation()`、`_resolve_episode_duplicate` | なし（検出episodeは無効が前提。episode IDはNoneを渡す） |
| 警報: `_resolve_drift(idx, estimated_start, episode_id)` | `handle_alarm_occurrence` |
| 警報: `_on_drift_resolution(idx)`（候補検証を保持していないとき） | 範囲外（同上） |
| 警報なし: FIFOの容量超過ぶんの確定、`train_step()` | `assign_released_pending_samples_to_current_training_model`、`record_training_request_and_train_held_models_when_due` |
| `history_drift_type.append(drift_type)` | 範囲外（下の「旧と違う点」） |
| `phase_seconds`・`processing_times`・`_record_model_compute` | 範囲外（未移植の診断） |

変化区間の先頭の位置は、旧`_estimated_drift_start`（`max(0, idx − min(len(buffer), span) + 1)`）と同じ式で、保留標本の件数と監視の観測の`estimated_change_span_sample_count`から求める。検出器の候補開始の位置は、監視の観測の`detector_candidate_start_sample_index`をそのまま記録する（旧`_detector_candidate_start`と一致することは、実旧との対照で示す）。旧は`max(0, …)`で0未満を防ぐ。新では、推定区間長は1以上で、監視がresetの後に観測した標本数以下であり、位置は非負で連続しているので、候補開始の位置は0以上で警報の位置以下になる。変化区間の先頭は、候補開始の位置以上で警報の位置以下になる（Nは1以上で推定区間長以下）。警報の記録のownerが確かめる「0≦候補開始≦変化区間の先頭≦警報」は、この理由で常に成り立ち、拒否に到達しない。

旧と違う点:

- **保留の持ち方**: 旧は標本（特徴・ラベル・概念ID）の組をFIFOに持つ。新は、位置を保留位置のownerが、標本を保留標本のownerが持つ。警報の処理と帰属の確定は保留位置のownerだけを更新するので、その後に、保留標本を保留位置に残った位置へ合わせる。
- **候補検証へ渡した標本の概念ID**: 旧は候補検証のsessionが概念IDつきの標本を持つ。新のsessionは標本だけを持ち、確定のときに呼出し側が概念IDを渡す。警報の処理が候補検証を開始したら、sessionが持つ標本が、警報時の保留標本の末尾と同一のオブジェクトであることを確かめて、その末尾の概念IDを、保留標本のownerに確定まで保持する。sessionが持つ標本は、変化区間の標本で、変化区間は警報時の保留標本の末尾である（警報時の区間の準備が、保留の末尾から変化区間を切り出し、区間解決がそのままsessionへ渡す）。
- **標本の形**: 旧は1次元の入力を2次元へ直して受け取り、複数行の入力も拒否しない。新は、特徴が1行の2次元、ラベルが1行1列のtensorだけを受け取る（保留へ入った標本は、後の警報や確定まで形を検査されないので、最初に確かめる）。1次元から直すのは、標本を作る側の役目とする。
- **標本ごとの結果種別の列**: 旧`history_drift_type`は記録しない。適応記録（位置と結果種別）から導けると見込むが、旧の数値との対応は確かめていない。保存のspecで扱う。
- **予測側の通知の位置**: 旧は、警報の位置の記録の後・警報の処理の前に`_on_drift_alarm`を、警報の処理の後（候補検証を保持していなければ）に`_on_drift_resolution`を呼ぶ。前者は警報の処理より前の保留標本を読む。予測の接続のspecで、本処理の同じ位置へ呼出しを足す。
- 計算量・所要時間の記録と検出episodeは扱わない。乱数は、旧のmodule全体の`random`に対して、借りた`Random`を使う（既存の部品の契約）。

## 3. 契約

```
process_observed_sample(*, indexed_observation, （owner）validation_session_holder, adaptation_record_store, diagnostic_evidence_collection, loss_change_monitor, loss_change_alarm_record_store, pending_training_assignment_buffer, pending_sample_observation_store, held_model_training_state_registry, loss_statistics_store, training_sample_store, model_training_and_assignment_counts_store, current_training_model_assignment, model_evaluation_sample_store, temporary_model_id_allocator, pending_model_upload_state, local_training_request_schedule, （候補検証）candidate_model_training_and_acceptance_settings, maximum_reference_mean_loss_increase, minimum_candidate_mean_loss_improvement, upload_delay_round_count, （警報）minimum_change_interval_sample_count, maximum_alarm_interval_mean_loss_increase, candidate_parameter_initialization_settings, architecture_reference_classifier, parameter_optimizer_settings, candidate_epoch_training_settings, detector_name, （学習）batch_sample_count, python_random_generator, local_training_settings, shared_feature_extractor, shared_parameter_optimizer) -> ObservedSampleProcessing
```

- owner以外の引数は、既存の部品の同名の引数へそのまま渡す。乱数生成器は、警報の処理と学習の両方へ同じものを渡す（旧は両方がmodule全体の乱数を使う）。
- `ObservedSampleProcessing`: `held_validation_advance`、`loss_monitoring_observation`、`alarm_occurrence_handling`（警報がなければNone）、`released_sample_observations`（警報があれば空）、`completed_joint_update_losses`。

`PendingSampleObservationStore`: `append_pending_sample_observation(indexed_observation)`、`snapshot_pending_sample_observations() -> tuple`、`retain_latest_pending_sample_observations(retained_sample_indices)`、`validation_assignment_sample_concept_ids`（property。保持していなければNone）、`hold_validation_assignment_sample_concept_ids(sample_concept_ids)`、`release_validation_assignment_sample_concept_ids() -> tuple`。

`LossChangeAlarmRecordStore`: `append_monitored_log_e_value(log_e_value)`、`append_alarm_record(alarm_sample_index, estimated_change_point_sample_index, detector_candidate_start_sample_index)`、`get_state_snapshot() -> LossChangeAlarmRecordSnapshot`。

## 4. 検査と処理順

| 検査（例外） | 位置 | 値の出所 |
| --- | --- | --- |
| 本処理が受け取るownerすべてがexact型（TypeError）: 候補検証の保持、適応記録、診断証拠、監視、警報の記録、保留位置、保留標本、保有モデルの学習状態、損失統計、モデル別の標本、学習量の計数、現在の学習帰属、評価標本、一時IDの採番、送信保留、学習要求の件数管理。乱数生成器がexact `random.Random`（TypeError） | 最初 | 引数。後の段でしか使われないownerは、使う部品の検査が、先の段の更新の後になる（候補検証の確定の適用、警報の処理）ので、最初に確かめる |
| 標本がexact `IndexedObservedTrainingSample`、位置がbuiltin int、概念IDがbuiltin intまたはNone、`training_sample`がexact `ObservedTrainingSample`、特徴とラベルがexact `torch.Tensor`（TypeError）。位置が非負、特徴が1行の2次元、ラベルが1行1列（ValueError） | 更新前 | 引数。frozenな値だが、手で組み立てた入力がありうる。標本の型・形・概念IDは、保留へ入った後、警報や確定まで検査されないので、ここで確かめる |
| 位置が、保留位置の最終観測位置の次で、監視の最後の観測の次（ValueError。どちらも未観測なら問わない） | 更新前 | 引数とownerの現在の状態。後の段（監視、保留への追加）が同じ理由で拒否すると、候補検証の進行だけが済んだ状態が残るので、先に確かめる |
| 保留標本の位置の並びが、保留位置の並びと同じ（ValueError） | 更新前 | 2つのownerの現在の状態。別々に渡されるので食い違いうる |
| 候補検証を保持している⇔概念IDを保持している（ValueError） | 更新前 | 2つのownerの現在の状態 |
| 標本の中身（特徴の数、ラベルの範囲、有限の値） | 候補検証の観測の中（保持中）、または現在のモデルの損失の評価の中（保持なし）。どちらも、どの更新より前 | 既存の部品の契約。保持中の候補検証の観測は、候補と参照の損失を評価してから記録する |
| そのほかの引数（設定、閾値、学習へ渡す値、本処理が直接は触れないowner） | それぞれを使う部品の中 | 既存の部品の契約。使う段に達しなければ検査されない |

処理順: (0)上の検査、(1)候補検証の進行。確定したら概念IDの保持を外す、(2)現在のモデルの損失の評価→基準の選択→監視の観測→監視の値の記録、(3)保留位置と保留標本へ追加、(4a)警報あり: 保留中の学習要求の学習→変化区間の先頭の計算→警報の位置の記録→警報1回ぶんの処理→候補検証を開始していれば概念IDの保持→保留標本を合わせる、(4b)警報なし: 帰属の確定→保留標本を合わせる→学習要求の記録、(5)結果を返す。

(1)より後の段が失敗した場合、済んだ段は残る（要求4.3）。段の中で失敗したときに、ownerの間の対応が崩れたまま残る例: 候補検証の進行が、sessionを外した後の診断への通知で失敗すると、概念IDの保持だけが残る。警報の処理が、保留位置を消費した後の段で失敗すると、保留標本が保留位置より多く残る。どちらも、その後の呼出しは(0)で拒否される。途中の失敗からの回復は扱わない（旧も、例外は標本処理の外まで伝わる）。(0)で確かめた値に由来する拒否（位置の連続性、保留の並び）は、(1)より後では起きない。保留標本を合わせる操作は、保留位置のownerが常に古い側から外す（容量超過の解放、警報の後の全消費、区間不足のときの全保持）ので、残す位置が末尾と一致し、拒否しない。

概念IDの対応づけは、sessionが、警報の処理へ渡した保留標本の末尾の`training_sample`そのもの（同一のオブジェクト）を、同じ順で持つことに依存する。末尾と同一でなければ`ValueError`にする。末尾と位置で対応させるので、同じ標本のオブジェクトが保留の古い側にもあっても、末尾の概念IDを選ぶ（実旧との対照testが、保持した概念IDを、実旧のsessionが持つ標本の概念IDと照合する。古い側に同じオブジェクトがある場合と、末尾と同一でない場合は、対応づけの関数の単独のtestで確かめる）。

候補検証の進行が検査しない「標本の位置が、候補検証を開始した警報の位置より後であること」は、(0)の位置の連続性と、警報1回ぶんの処理の検査（警報の位置＝保留位置の最終観測位置）から従う: 候補検証は警報の位置で始まり、次に処理される標本の位置は、その最終観測位置の次である（held-candidate-validation-progressの設計4節に残っていた限界は、本処理を通る限り起きない）。

確かめないこと: owner・乱数生成器・標本のほかの引数（設定、閾値、学習へ渡す値。要求4.4）。たとえば、候補検証を保持していないときの候補検証の閾値や、警報のない標本での警報の設定は読まれない。不正な設定が、使う段で初めて拒否されると、先の段の更新は残る（例: 変化区間の最小件数の不正は、警報1回ぶんの処理の中で拒否されるが、その時点で、保留中の学習要求の学習と警報の位置の記録は済んでいる）。設定は実行の間変わらないので、組立てのときに確かめるのが本来の位置で、新clientのspecで扱う。ownerどうしの整合（監視のクラス数と分類器のクラス数が同じであること、共有部が保有モデルのものであること）も、組立ての役目として、本処理では確かめない。

## 5. Allowed Dependencies

- `runtime/observed_sample_processing.py`: 契約の引数の型（型注釈と、(0)のexact型の検査）、`torch.Tensor`、`ObservedTrainingSample`、呼び出す既存の関数（`advance_held_candidate_validation`、`evaluate_classifier_per_sample_bounded_losses`、`select_loss_monitoring_baseline_mean_loss`、`train_held_models_for_pending_training_requests`、`handle_alarm_occurrence`、`assign_released_pending_samples_to_current_training_model`、`record_training_request_and_train_held_models_when_due`）、結果recordのfieldの型、`dataclasses.dataclass`、`random.Random`、`torch.optim.Optimizer`。
- `pending_sample_observation_store.py`: `IndexedObservedTrainingSample`だけ。
- `loss_change_alarm_record_store.py`: `dataclasses.dataclass`と`math`の`inf`・`isnan`だけ。

許可集合を既存と同じ形で登録する（symbolごとの注入契約testは足さない）。

## 6. Revalidation Triggers

呼び出す7つの部品の引数・検査の位置・更新順、保留位置のownerが外す位置の規則、sessionが持つ標本の同一性、監視の観測のfield、適応記録の結果種別が変わったら、本specの対照testを再検証する。予測の接続で本処理へ呼出しを足すときは、足した位置より前後の順序を対照testで確かめ直す。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/methods/fedsda/training_data_assignment/pending_sample_observation_store.py | 新規 |
| src/federated_learning_experiments/evaluation/loss_change_alarm_record_store.py | 新規 |
| src/federated_learning_experiments/runtime/observed_sample_processing.py | 新規 |
| tests/refactoring/test_pending_observation_and_alarm_record_stores.py | 新規。2つのownerの単独の動作 |
| tests/refactoring/test_observed_sample_processing.py | 新規。実旧との対照、拒否、順序 |
| tests/refactoring/test_single_run_dependency_boundaries.py | 許可集合の登録 |
| tests/refactoring/fresh_process_smoke.py | 各流れの最後に、標本1件の処理を続けて呼ぶ |

## 8. Testing Strategy / Requirements Traceability

oracleは、実旧client（`SharedBackboneClassConditionalESRFedSDAClient`）の実物の`process_one_step`。予測（`_record_prediction`）だけを止め、候補検証の観測と確定、検出器、警報の処理（`_resolve_drift`）、FIFO、帰属の確定、学習stepは実物のまま使う。警報1回ぶんの処理のoracle（`build_alarm_occurrence_oracle`）で最初の警報を両実装に処理させ（この一致は上流のtestが確かめている）、その直後の状態から、同じ標本列を両実装へ1件ずつ渡して、各標本の後に全状態を照合する。標本列は、決まった規則のラベルの列と、その時点の実旧の現行モデルの損失が最小・最大になるラベルを選ぶ列を使う（実旧のモデルは入力を作るためだけに読み、期待値には使わない）。履歴統計の平均と、候補の採否の余裕を変えて、警報と確定の全結果を通す。新しく使う旧メソッドは`process_one_step`だけで、警報のない標本での帰属確定のtestが、部分的に止めた形で実行済み。

| 要求 | 証拠 |
| --- | --- |
| 1.1〜1.5, 5.1 | 上の対照。2値・多クラス、最初の警報の結果（保持なし／保持ありで始まる）、標本列、履歴統計、採否の余裕を変え、各標本の後に要求5.1の全項目を既存のhelperで実旧と照合。候補検証へ渡した標本の概念IDは、実旧のsessionが持つ標本の概念IDと照合。監視の基準は、監視の状態の照合（検出器ごとの基準を含む）で確かめる |
| 5.2 | 対照の全条件で観測した適応結果の集合が、警報の5結果と確定の4結果に一致すること（対照の全条件を実行したときだけ確かめるtest。全pytestで、skipされずに実行されたことを証拠に記録する） |
| 1.1〜1.5（順序） | 各段の呼出しを、呼出しの順と、その時点の保留・警報の記録・監視つきで記録し、警報のない標本と警報のある標本のそれぞれで、旧と同じ順であること、渡る値が正しいこと |
| 1.6 | 結果recordの型と、警報の有無とfieldの対応 |
| 2.1〜2.3, 3.1〜3.3 | 2つのownerの単独のtest（追加・読取り・残す操作・保持と解除・拒否で状態が変わらないこと・snapshotの独立） |
| 4.1, 4.2 | 候補検証を保持している状態と保持していない状態のそれぞれで、各不正入力（標本の型・形・概念ID・位置、2つのownerの対応、特徴の数、ラベルの範囲）を渡し、全状態が変わらないこと、拒否の後に正しい標本を処理できること。本処理が受け取るownerそれぞれと乱数生成器の、別の型・派生型で、全状態が変わらないこと |
| 4.3 | 途中の段（帰属の確定）を失敗させ、後の段（学習要求の記録）が呼ばれず、例外がそのまま伝わり、先の段の更新が残ること（段の中で失敗したときの各部品の状態は、各部品のspecが確かめている） |
| 4.4 | 本specでは新しいtestを置かない。設定・閾値・学習へ渡す値の検査と、使う段に達しなければ読まれないことは、既存の部品のtestが確かめている（学習要求の処理の「学習へ進まないとき、学習へ渡すだけの値を読まない」、候補検証の進行の「保持がなければ何も変えない」ほか） |
| 5.3 | 汎用の変異tool（新しい関数とownerのメソッド）、依存の許可集合、共用のfresh process script、全pytest、Ruff・Pyright |

未検証として残す: 予測の接続と予測側の通知、標本ごとの結果種別の列と旧の数値の対応、計算量・時間、検出episodeを有効にした場合、run終端・ラウンド境界・サーバ同期、新全体runのgolden一致。
