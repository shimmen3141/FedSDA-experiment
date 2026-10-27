# 影響範囲の調査の報告（2026-10-28、gpt-6.1-sol、読取りだけ、HEAD 0d6c586d）

調査の依頼文は`results/_sol_prompts/improve023-scoping-prompt.txt`（gitの管理外）。以下は報告の本文をそのまま写したもの。

調査対象は **`refactor/architecture`、HEAD `0d6c586d`** です。編集・実装・commit・テスト実行はしていません。コードの静的追跡と、版1〜14の保存済みgolden JSON（各版2環境、計28ファイル）の読取りを行いました。

主な結論は次の3点です。

- **版7〜14にも候補検証ありのgoldenが4ケース残っています。**
- **検証なしの数値経路は切り離せますが、設定・記録・指標を削除すると、既存goldenとの比較形式とrun識別子は変わります。**
- **`post_alarm_candidate_validation_session_start.py`には検証なしでも使う候補学習があるため、丸ごと削除できません。**

以下の「維持できる」は静的解析による見込みです。削除後の数値一致は未検証です。

**1. 消す対象と、残す対象の境界**

候補検証を選ぶ採否方針は、現在の新実装では **1値だけ**です。

| 項目 | 確認結果 |
|---|---|
| 消す方針値 | `current_model_first_reuse_then_two_segment_candidate_validation` |
| 残る方針値 | `current_model_first_reuse_then_candidate_adoption_without_validation` |
| 通常の構成から検証ありを選ぶ入口 | `fedsda-with-candidate-validation` |
| 設定を直接組み立てる入口 | `CandidateModelTrainingAndAcceptanceSettings.candidate_model_acceptance_policy` |
| そのほかの選択元 | golden helper、旧実装照合fixture、研究ツールの検証あり変種 |

根拠は、方針の許容値を宣言する[設定型:14](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/methods/fedsda/candidate_model_selection/candidate_model_training_and_acceptance_settings.py:14)、構成を組み立てる[experiment_configurations.py:269](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/sweep/experiment_configurations.py:269)です。

検証あり構成は採否方針に加え、次も明示します。

- 候補と一緒に共有部を更新。
- 警報区間は検出器の推定と損失増加許容量で選択。
- 学習者は再初期化のみ、初期統計は再代入。
- 評価clientは対象保有者から選択、上限3。
- 損失監視の基準を警報まで維持。

これらの値は**候補検証だけの値ではありません**。構成の削除に伴って各値まで一括削除するのは別の変更になります。

削除対象の中心となるモジュールは以下です。パスは `src/federated_learning_experiments/` からの相対位置です。

| モジュール | 削除対象の主な型・関数 |
|---|---|
| `runtime/candidate_validation_session_holder.py:8` | `CandidateValidationSessionHolder`、保持・解放メソッド |
| `runtime/held_candidate_validation_progress.py:60` | `HeldCandidateValidationAdvance`、`HeldIncompleteCandidateValidationFinalization`、`apply_alarm_response_to_validation_session_holder`、`advance_held_candidate_validation`、`finalize_held_incomplete_candidate_validation` |
| `runtime/candidate_validation_adaptation_recording.py:47` | `record_completed_candidate_validation`、`record_incomplete_candidate_validation_finalization` |
| `runtime/post_alarm_candidate_validation_progress.py:51` | `PostAlarmCandidateValidationCompletion`、`PostAlarmCandidateValidationProgress`、`advance_post_alarm_candidate_validation` |
| `runtime/post_alarm_candidate_validation_sample_observation.py:30` | `observe_post_alarm_candidate_validation_sample` |
| `runtime/post_alarm_candidate_validation_resolution.py:57` | `PostAlarmCandidateValidationResolution`、`apply_post_alarm_candidate_validation_resolution` |
| `runtime/incomplete_post_alarm_candidate_validation_finalization.py:32` | `IncompletePostAlarmCandidateValidationFinalization`、`finalize_incomplete_post_alarm_candidate_validation` |
| `runtime/post_alarm_reference_model_fixation.py:23` | `FixedPostAlarmReferenceModels`、`fix_reference_models_at_alarm` |
| `methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py:12` | `PostAlarmCandidateLossCollectionState`、`PostAlarmCandidateLossCollection` |
| 同ディレクトリ `post_alarm_candidate_loss_evaluation.py:127` | `select_available_reference_within_historical_loss_tolerance`、`PostAlarmCandidateLossEvaluation`、`evaluate_candidate_using_post_alarm_losses` |
| 同ディレクトリ `post_alarm_candidate_validation_decision_record.py:9` | `PostAlarmCandidateValidationDecisionRecord` |
| 同ディレクトリ `incomplete_post_alarm_candidate_validation_decision_record.py:7` | `IncompletePostAlarmCandidateValidationDecisionRecord` |
| 同ディレクトリ `candidate_validation_decision_record_store.py:11` | `CandidateValidationDecisionRecordStore` |
| `runtime/post_alarm_candidate_validation_session_start.py:54` | **部分削除**：`PostAlarmCandidateValidationSession`、`start_post_alarm_candidate_validation_session`と検証専用部分 |

最後のファイルには、残すべき `TrainedAlarmIntervalCandidate` と `train_candidate_on_alarm_change_interval` があります。検証なしでも使うことが[同ファイル:239](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/post_alarm_candidate_validation_session_start.py:239)に明記され、[検証なしの分岐:332](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/alarm_change_interval_resolution.py:332)から呼ばれています。

追加で取り除く対象は次のとおりです。

| 種別 | 削除対象・境界 |
|---|---|
| 設定 | `candidate_post_alarm_validation_sample_count`、`minimum_candidate_mean_loss_improvement`、検証ありの方針値 |
| 設定間制約 | 「MDLと検証ありを組み合わせない」「共有部固定と検証ありを組み合わせない」「候補検証の改善量と早期終了の改善量を同値にする」 |
| 設定型 | 現在の `CandidateModelTrainingAndAcceptanceSettings` は2 fieldとも整理対象になる。型と、それを持つ設定field自体の削除候補 |
| 閾値定数 | 検証標本数10は削除。改善量0.0001は**学習の早期終了にも使うため値は残し、定数名を整理** |
| 補助関数 | `loss_baseline_selection.py:49` の `select_post_alarm_reference_historical_mean_loss`。製品コードの利用元は参照固定だけ |
| client状態 | 検証session holder、判定record store、検証に渡した標本の概念ID保持 |
| 実行契約 | `finalize_incomplete_candidate_validation`、終端呼出し、`incomplete_candidate_validation_finalization` stage、FedAvg側のno-op |
| 構成 | 定数・登録一覧・builder分岐・専用testの `fedsda-with-candidate-validation` |

制約は[client設定:304](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_run_client_settings.py:304)、定数の共有は[既定設定:130](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py:130)、概念ID保持は[pending_sample_observation_store.py:65](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/methods/fedsda/training_data_assignment/pending_sample_observation_store.py:65)、終端呼出しは[実行loop:173](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/execution/stream_protocol_execution_loop.py:173)が根拠です。

残す境界は明確です。

- 候補の初期値選択・生成・epoch学習・早期終了。
- 候補の採用、共有部の統合、ローカル登録、送信保留。
- 警報の検出・応答、現在モデル維持、保有モデル再利用、MDLによる区間・帰属選択。
- `maximum_tolerated_mean_loss_increase`：検証以外の再利用判定でも使用。[設定の説明:69](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_run_client_settings.py:69)
- `minimum_validation_loss_decrease`：候補**学習の早期終了**の設定であり、警報後の候補検証とは別。
- `fedsda-without-candidate-validation`：現在は「候補の共有部固定を外す」構成でもあり、今回の削除対象には直結しない。

記録と指標の削除対象は次です。

| 対象 | 項目 |
|---|---|
| 記録表 | `candidate_validation_decision.*` の全16列 |
| run指標 | `candidate_validation_decision_count` |
| 指標の型・まとまり | `CandidateValidationDecisionMetrics` 全23 field、`candidate_validation_decision_metrics` |
| 適応指標 | `start_candidate_validation_count`、`defer_during_candidate_validation_count`、`reject_candidate_model_count` |
| 適応結果値 | `alarm_during_candidate_validation`、`alarm_interval_candidate_validation_started`、`post_alarm_validation_*` の5値 |
| 判定理由値 | `both_segment_margins_passed`、`first_segment_margin_failed`、`second_segment_margin_failed`、`both_segment_margins_failed`、`current_reference_within_historical_loss_tolerance`、`alternative_reference_within_historical_loss_tolerance` |

16列は、ID・位置・標本数6列、flag3列、文字列2列、損失5列です。宣言は[記録表:342](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_run_record_tables.py:342)、指標は[指標型:74](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_change_response_metric_derivation.py:74)、結果値の対応は[同ファイル:29](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_change_response_metric_derivation.py:29)です。採用・維持・再利用の指標自体は残します。

**2. 版1〜14のgolden**

「検証あり方針を設定する」と「実際に検証判定記録が発生する」を分けました。

次の略称を使います。

- **C**：以下の小規模構成10ケース  
  `fedsda`、`fedsda-without-prediction-weight-recalibration`、`fedsda-loss-distance-clustering`、`fedsda-overall-loss-esr-detection`、`fedsda-adahedge-prediction-weights`、`fedsda-single-model-prediction`、`fedsda-classwise-adwin-detection`、`fedsda-overall-loss-adwin-detection`、`fedsda-without-concept-specific-adapter`、`fedsda-independent-models`
- **E**：`fedsda-sine2-events`
- **M**：`fedsda-mnist2-decisions`
- **R**：`fedsda-sine2-reference-reuse`
- **A**：`fedsda-sine2-alternative-reference-reuse`

| 版 | 全case数 | 検証あり方針のcase | 保存済み判定記録が1件以上あるcase |
|---:|---:|---|---|
| 1 | 17 | C＋E・M＝12 | E・M |
| 2 | 18 | C＋E・M・R＝13 | E・M・R |
| 3 | 18 | C＋E・M・R＝13 | E・M・R |
| 4 | 19 | C＋E・M・R・A＝14 | E・M・R・A |
| 5 | 19 | 同上 | 同上 |
| 6 | 19 | 同上 | 同上 |
| 7 | 21 | E・M・R・A＝4 | 同じ4ケース |
| 8 | 21 | 同上 | 同上 |
| 9 | 21 | 同上 | 同上 |
| 10 | 21 | 同上 | 同上 |
| 11 | 22 | 同上 | 同上 |
| 12 | 22 | 同上 | 同上 |
| 13 | 22 | 同上 | 同上 |
| 14 | 22 | 同上 | 同上 |

両環境で、方針を設定するcase集合と判定記録があるcase集合は一致しました。**Cは検証あり方針ですが、保存された判定件数は0です。**

根拠は[版1〜6の方針上書き:253](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/behavior_golden_cases.py:253)、[case一覧:297](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/behavior_golden_cases.py:297)、[検証経路の4ケース:405](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/behavior_golden_cases.py:405)、[版7以降の構成切替え:460](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/behavior_golden_cases.py:460)です。

検証なしで残せる比較はあります。

- `fedsda-without-candidate-validation`：版1から存在。
- Cの10ケース：版7以降は検証なし。
- `fedsda-sine2-adoption-without-validation`、`fedsda-mnist2-adoption-without-validation`：版7以降。
- `fedsda-mnist4-warm-started-learner`：版11以降。
- FedDrift・FedAvgの各ケース。

ただし、後述の保存形式を正規化しない限り、これらも既存JSONへの完全一致比較は維持できません。

**3. 旧実装との照合test**

候補検証そのものを旧実装と比べる主な関数は以下です。ファイルは `tests/refactoring/` 配下です。

| ファイル | 該当関数と開始行 |
|---|---|
| `test_post_alarm_candidate_loss_collection.py` | `test_candidate_loss_collection_matches_legacy_session:62`、`test_candidate_loss_collection_readiness_matches_legacy_client:273` |
| `test_post_alarm_candidate_loss_evaluation.py` | `test_post_alarm_candidate_reference_selection_matches_legacy:49`、`test_post_alarm_candidate_evaluation_matches_legacy_finalization:270`、`test_post_alarm_candidate_public_loss_connection_matches_legacy:320` |
| `test_post_alarm_reference_model_fixation.py` | `test_fixed_references_match_actual_legacy_reference_snapshots_and_random_state:207`、`test_historical_mean_losses_match_actual_legacy_session_start:289`、`test_fixation_then_observation_and_resolution_match_actual_legacy_session:481`、`test_references_stay_fixed_while_held_models_continue_actual_joint_training:531` |
| `test_post_alarm_candidate_validation_session_start.py` | `test_session_start_matches_legacy:505` |
| `test_post_alarm_candidate_validation_sample_observation.py` | `test_observed_loss_sequences_match_actual_legacy_observation:250`、`test_observation_then_evaluation_and_resolution_match_actual_legacy_observation:517`、`test_observed_and_resolved_validation_continues_actual_joint_training:569` |
| `test_post_alarm_candidate_validation_resolution.py` | `test_resolution_matches_actual_legacy_forward_validation_branches:230`、`test_resolved_validation_continues_actual_joint_training:602` |
| `test_post_alarm_candidate_validation_progress.py` | `test_validation_progress_matches_legacy:351`、`test_validation_decision_record_matches_legacy:604`、`test_validation_progress_connects_actual_training_and_resolution:709` |
| `test_incomplete_post_alarm_candidate_validation_finalization.py` | `test_incomplete_validation_finalization_matches_legacy:230`、`test_incomplete_validation_decision_record_matches_legacy:574`、`test_incomplete_validation_finalization_connects_actual_training:718` |
| `test_held_candidate_validation_progress.py` | `test_alarm_response_updates_holder_like_legacy_session_attribute:155`、`test_completed_held_validation_is_recorded_then_released_like_legacy:355`、`test_incomplete_held_validation_is_recorded_then_released_like_legacy:695` |

さらに、共有機能のtestにも検証経路が含まれます。

| ファイル | 該当関数・扱い |
|---|---|
| `test_alarm_change_interval_resolution.py` | `test_alarm_change_interval_resolution_matches_legacy:762`、`…starts_candidate_validation_like_legacy:887`、`…observes_samples_after_started_validation_like_legacy:1775` |
| `test_alarm_buffer_response.py` | `test_buffer_response_matches_real_legacy:339`、`test_active_candidate_alarm_absorbs_all_and_preserves_session:440` |
| `test_alarm_occurrence_handling.py` | `test_alarm_occurrence_matches_real_legacy_alarm:175` |
| `test_observed_sample_processing.py` | `test_sample_processing_matches_real_legacy_sample_processing:488`と適応経路の網羅test |
| `test_fedsda_run_client.py` | `test_run_client_matches_real_legacy_client_from_construction_to_run_end:959` |
| `test_fedsda_stream_protocol_run.py` | `test_whole_stream_protocol_run_matches_real_legacy_whole_run:472`、`test_whole_run_of_other_datasets_matches_real_legacy_whole_run:651` |
| `test_fedsda_run_metric_derivation.py` | `test_golden_condition_metrics_match_real_legacy_experiment_result:1143`、計算量照合`:1226`、trace照合`:1400`、旧golden照合`:1433`、小規模run照合`:1966` |
| `test_fedsda_configuration_variants.py` | `test_configuration_variant_matches_real_legacy_whole_run:367`、MNIST2の構造変種照合`:1119` |
| `test_adopted_candidate_local_adoption.py` | `test_local_adoption_matches_actual_legacy_forward_validation_acceptance:319`、連続採用`:413`。**採用機能自体は残す** |

旧照合用fixtureは `NEW_MODEL_CREATION_POLICY="forward_persistent"` を設定します。[fixture:168](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/test_fedsda_run_client.py:168)  
旧golden条件のbuilderも、検証あり方針へ明示的に戻しています。[builder:543](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/test_fedsda_final_configuration_run_settings.py:543)

候補検証を通らない旧照合も残せます。

- `fedsda-without-candidate-validation`変種は旧側を `immediate` にする。[変種設定:185](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/test_fedsda_configuration_variants.py:185)
- 候補生成・epoch学習・初期登録・共有部統合・通常学習などの部品単位の旧照合。
- FedDrift・FedAvgの旧照合。
- 版7以降の検証なしgolden。ただし、これは**旧実装との直接比較ではなく、新実装の過去版との比較**です。

注意点として、変種照合testは変種と提案構成の両方を実行します。検証なし変種を残しても、比較相手のfixtureは整理が必要です。[比較loop:381](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/test_fedsda_configuration_variants.py:381)

| 旧golden test | 候補検証を使うか |
|---|---|
| `tests/test_regression.py` | **通常の単独実行では使わない**。`NEW_MODEL_CREATION_POLICY`の既定は`immediate`で、case上書きにも検証指定がない。[case一覧:46](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/test_regression.py:46)、[旧既定:80](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/federated_drift_experiment/config.py:80) |
| `tests/test_proposed_regression.py` | **使う**。`forward_persistent`と前向き検証10標本を固定。[設定:33](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/test_proposed_regression.py:33) |

新実装から検証を消しても、旧パッケージを変更しなければ、この2つの旧実装自身のgolden testは残せます。廃止が必要になるのは、新実装から候補検証ありの旧結果へ合わせる経路です。

**4. 値・乱数・保存形式の安全性**

| 観点 | 確認した事実 | 削除への含意 |
|---|---|---|
| 毎標本の検証処理 | 検証なしでも進行関数を呼ぶが、sessionが`None`なら直ちに戻る | 呼出しの除去で乱数列が変わる箇所は静的追跡では見つからない |
| 終端処理 | holderが空なら直ちに戻る | 検証なしでは学習・採用を追加しない |
| 参照モデル固定 | 分類器の生成時にCPU torch乱数を消費する | **検証開始側だけ**。検証なしの採用経路にはない |
| 候補生成・学習 | 検証なしでも候補を生成してepoch学習する | 生成回数・初期化順序・学習順序を維持する必要あり |
| Python Random・NumPy | sessionなしの進行・終端経路に乱数生成処理は見つからない | 専用のno-op除去は数値維持が見込める |
| 実行event | 全clientに候補検証の終端stageを呼ぶ | stageを削除すると実行event列・契約testは変わる |

根拠は[毎標本の呼出し:486](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/observed_sample_processing.py:486)、[sessionなしの即時return:163](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/post_alarm_candidate_validation_progress.py:163)、[終端の即時return:529](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_run_client.py:529)、[参照生成の乱数消費:68](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/post_alarm_reference_model_fixation.py:68)です。

**推論**：残す候補学習をそのまま移し、検証専用経路だけを削れば、検証なしの予測・学習・通信・MACの維持は可能です。ただし、実時間は処理削減で変わり得ます。数値一致と乱数状態の実測が受入条件になります。

保存形式は確実に変わります。

- 検証なしでも16列の空表を生成するため、その削除でNPZ等のキー集合が変わる。[表の出力:435](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/runtime/fedsda_run_record_tables.py:435)
- ゼロ件・未定義の候補検証指標も消えるため、JSON・CSVの項目集合が変わる。
- 設定serializerは型名と全fieldを保存するため、field・型の削除や改名も保存設定を変える。[serializer:43](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/core/settings_serialization.py:43)
- run識別子は**構成名＋完全な保存設定**をhash化するため、検証を使わないFedSDAも識別子が変わる。[識別子:125](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/sweep/experiment_sweep_plan.py:125)
- 再開補助情報は設定・基準版・source hashも照合するため、既存掃引へそのまま再開できる前提にはできない。[再開情報:132](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/src/federated_learning_experiments/sweep/experiment_sweep_resumption.py:132)

golden readerも現在の設定と保存設定を比べるため、**検証あり4ケースを外すだけでは不十分**です。[golden検査:1458](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tests/refactoring/behavior_golden_cases.py:1458)

manifestについては仕組みを区別する必要があります。

| 仕組み | 影響 |
|---|---|
| 新実装 `run_experiment_sweep.py` | 出力先をまたぐmanifest重複検査は実装されていない。run識別子・再開照合が対象。[文書:191](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/docs/experiments/experiment-sweep.md:191) |
| 旧Pareto系manifest | 設定fingerprintと、commit・旧ソースhash・旧golden等のprovenanceを使用。新設定の削除が直接その設定へ入るわけではないが、commit変更はexecution fingerprintへ影響する。[provenance:77](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/federated_drift_experiment/experiment_spec/manifests.py:77) |

したがって、過去runとの重複・再開の同一性を保つ変更ではありません。過去manifestを書き換えて合わせることは推奨しません。

**5. 研究ツール・docs・.kiro**

| 対象 | 依存と扱いの候補 |
|---|---|
| `run_candidate_validation_waiting_diagnostic.py` | 検証progressを差し替えて強制採用し、判定表を読む。**実行入口の廃止候補**。[依存:27](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/run_candidate_validation_waiting_diagnostic.py:27) |
| `tools/research_candidate_validation_waiting/checks.py` | 検証progress・終端処理・旧fixtureへ依存。対応するchecksも廃止候補 |
| `tools/research_prediction_initial_mass/run.py` | `validation`モードだけ検証あり構成を選ぶ。モードを除き、ほかを残す候補。[構成選択:128](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tools/research_prediction_initial_mass/run.py:128) |
| `tools/research_provisional_candidate_expert/run.py` | `validation`、`provisional-expert`、forward30、sequential-test等が依存。**ファイル全体の削除は不適切**。現在の非検証変種もある。[変種:51](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tools/research_provisional_candidate_expert/run.py:51) |
| 同上の共通出力 | 検証標本数・採否方針fieldを無条件に読むため、非検証変種にも修正が必要。[出力:2914](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tools/research_provisional_candidate_expert/run.py:2914) |
| `tools/research_s3_contribution/diagnostic.py` | 検証を使わないが、検証開始モジュール内の候補生成を直接patch。移動先へ追随する。[依存:117](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tools/research_s3_contribution/diagnostic.py:117) |
| `tools/research_output_layer_learner/policy.py` | 共有の候補学習関数をpatch。関数移動・署名変更への追随が必要。[依存:44](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/tools/research_output_layer_learner/policy.py:44) |
| `tools/research_component_ablation/*` | 主に保存済み結果の解析と検証なし構成名。過去結果を読む機能は残す候補 |
| 旧ablation shell・旧study JSON・旧成果物復元 | 旧実装・旧保存結果の用途。今回の新実装削除へ自動で含めない |

文書の扱いは以下が妥当です。

- **現在の説明を更新**：`docs/overview/proposed-method.md`、`docs/experiments/experiment-sweep.md`、`refactoring-baseline.md`の現行版部分、改善候補一覧のIMPROVE-023状態、再開案内。
- **必要箇所を確認**：`docs/components/new-model-creation.md`、`model-reuse.md`。旧実装の説明を含むため、検証の記述を一括削除しない。
- **過去記録として保持**：2026-10-17〜19の候補検証診断・単純化・共有部固定・待機・予測初期質量・server結果・provisional expert報告、ablation報告、実験台帳。
- **既存specは履歴として保持**：`post-alarm-candidate-*`、`post-alarm-reference-model-fixation`、`held-candidate-validation-*`、`candidate-validation-*`、`fedsda-default-adoption-with-fixed-candidate-shared-features`等。新しい削除specから旧specへ参照する。

過去記録を書き換えない規則は[共通引継ぎ手順:「文書の扱い」](/C:/Users/yshin/Research/FedSDA-experiment/.worktrees/refactoring/.kiro/steering/agent-handoff.md:112)にあります。今回の削除で過去の承認・実験結果まで変更する必要はありません。

**6. 規模と段階分け**

| 範囲 | 実測または見積り |
|---|---|
| 中心14モジュール | **2,022行を実測**。うち1ファイルは共有候補学習を含む |
| 丸ごと削除候補13モジュール | **1,665行**。共有関数を含む357行のファイルを除いた値 |
| 検証名を持つ専用test | **12ファイル、7,948行を実測**。共有fixtureの移設が必要なものを含む |
| 文字列検索で関係した新ソース | 32ファイル。削除ファイル数ではない |
| 同じ検索で関係したrefactoring test/helper | 61ファイル。削除ファイル数ではない |
| 製品コードの純削除量 | **約2,000〜2,800行の見積り**。共有学習の移動は純削除に含めない |
| testの純削除量 | **約7,000〜10,000行の見積り**。旧全体run照合をどこまで外すかで変動 |
| 変更ファイル全体 | **約60〜100ファイルの見積り**。golden JSON・過去docsの削除は含めない |

行数見積りは実装diffではありません。特に巨大な依存境界test、旧照合fixture、研究ツールの残し方で変わります。

推奨する下位taskの分け方は次です。

1. **残す比較と共有部品を確定する**  
   検証なしgoldenの比較方法を決め、候補学習・test fixtureを検証専用モジュールから分離する。
2. **構成・設定の入口を閉じる**  
   検証あり構成、方針、専用field・制約を削除。残す研究ツールの出力も追随させる。
3. **検証の実行と状態を削除する**  
   開始・標本観測・判定・警報保留・終端回収・client owner・実行契約をまとめて除く。
4. **記録・指標・保存形式を整理する**  
   16列、指標、適応結果値を削除し、golden比較とrun識別子変更を扱う。
5. **専用test・研究入口・現行文書を整理し、数値を照合する**  
   残す経路の予測・パラメータ・通信・MAC・乱数状態を比較する。

入口だけ先に削除すると既存testが検証あり設定を組み立てられなくなるため、**最初に比較対象とfixtureの扱いを決める順序**を勧めます。

**判断が必要な点**

| 判断 | 選択肢と影響 | 推奨 |
|---|---|---|
| 過去goldenの実行範囲 | 全旧版の実行をやめる／検証なしcaseだけ残す | **検証なしcaseを残す**。版1〜6のCも検証あり方針なので対象外にする |
| 過去goldenの保存形式 | 過去JSONを更新／test側で削除項目を明示的に除外 | **過去JSONを保持し、test側で正規化**。削除する各field・列を列挙し、ほかの差を隠さない |
| 新しい基準版 | 版14を維持／新しい版を追加 | **新しい版を追加**。数値維持と保存形式変更を記録しやすい |
| 旧全体run照合 | 検証ありだけ廃止／関連ファイルを全面廃止 | **検証ありだけ廃止**。検証なし変種と共有部品の照合は残す |
| 研究ツール | 全体削除／検証ありモードだけ廃止 | **モード単位**。現在の非検証研究を巻き込まない |
| `fedsda-without-candidate-validation`の名前 | 維持／共有部更新を表す名前へ変更 | **今回は維持**。改名まで含めると別のrun識別子変更と利用箇所修正が増える |
| 旧実装自体 | 今回削除／別途扱う | **今回の対象外**。旧goldenと旧成果物の道具が引き続き使う |

主担当が仕様へ落とす際は、まず「検証なしの計算結果は維持する」「保存形式と識別子の変更は受け入れる」「候補検証ありの実行比較を終了する」の3つを別々の受入条件にすることを推奨します。
