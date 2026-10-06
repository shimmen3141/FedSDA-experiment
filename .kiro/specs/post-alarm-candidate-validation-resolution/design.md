# 設計 revision 2

## 配置と責務

runtime/post_alarm_candidate_validation_resolution.pyへ、結果recordと状態を持たない組立関数を置く。FedSDAの評価結果record（methods/fedsda）と、runtimeの採用・吸収の組立、学習層の現在ID ownerを横断する。採用・吸収と同じ外側runtimeに配置し、同層の2関数を呼ぶ。下位層からruntimeへの依存は作らない。

## 型

`POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES: tuple[str, ...] = ("candidate_adopted_as_new_model", "held_reference_model_reused", "current_model_maintained", "candidate_rejected")`

`PostAlarmCandidateValidationResolution`（frozen, kw_only dataclass）:
- `resolution_outcome: str`（上の4値のいずれか。__post_init__で検証）
- `assigned_model_id: int`（保留標本の帰属先。確定後の現在の学習帰属IDと同じ）
- `training_model_assignment_change: TrainingModelAssignmentChange | None`（現在IDが変わったときだけ）

## APIと順序

`apply_post_alarm_candidate_validation_resolution(*, post_alarm_candidate_loss_evaluation:PostAlarmCandidateLossEvaluation, pending_assignment_training_samples:tuple[ObservedTrainingSample, ...], pending_assignment_sample_concept_ids:tuple[int | None, ...], temporary_model_id_allocator:TemporaryModelIdAllocator, adopted_candidate_classifier:ResidualAdapterClassifier, candidate_concept_specific_parameter_optimizer_state:ParameterOptimizerState, initial_statistics_input_features:Tensor, initial_statistics_observed_class_labels:Tensor, upload_delay_round_count:int, candidate_trained_sample_count:int, candidate_parameter_update_step_count:int, held_model_training_state_registry:HeldModelTrainingStateRegistry, loss_statistics_store:ModelAndClassLossStatisticsStore, training_sample_store:ModelTrainingSampleStore, model_training_and_assignment_counts_store:ModelTrainingAndAssignmentCountsStore, current_training_model_assignment:CurrentTrainingModelAssignment, pending_model_upload_state:PendingModelUploadState)->PostAlarmCandidateValidationResolution`

状態を変更しない段階:

1. 評価結果がexact PostAlarmCandidateLossEvaluationでなければTypeError。`candidate_accepted`がexact boolでなければTypeError。`reusable_reference_model_id`がNoneでもexact intでもなければTypeError。採用かつ再利用可能参照ありはValueError。現在IDのownerがexact具体型でなければTypeError。
2. 保留標本列がexact tupleでなければTypeError。概念ID列がexact tupleでなければTypeError、長さが保留標本列と異なればValueError、要素がNoneでもexact intでもなければTypeError。（吸収の検証と同じ条件。採用では概念ID列を使わないが、同じ引数契約を全結果種別で課す。）
3. `previous_model_id = current.current_training_model_id`。

結果種別ごとの状態変更:

- 採用（candidate_accepted）: `assignment_change = adopt_candidate_as_current_training_model(採番owner, 候補, 管理器, 特徴, ラベル, 待機, 2計数, 保留標本, registry, 統計, 標本store, 計数store, 現在ID, 送信保留)`。採用は自身の全検証と登録の数値生成を終えてから状態を変更する（完了spec）。結果は`candidate_adopted_as_new_model`、帰属先`assignment_change.current_model_id`、変更記録`assignment_change`。
- 再利用可能参照あり: `target = reusable_reference_model_id`。`absorb_assigned_training_samples_into_held_model(model_id=target, 保留標本, 概念ID列, registry, 標本store, 計数store, 統計)`の後に`assignment_change = current.assign_model_for_training(model_id=target)`。吸収は吸収先の保有と全標本を状態変更前に検証する（完了spec）ので、拒否時は現在IDも未変更。assignはexact intの入力で拒否しない。変更があれば`held_reference_model_reused`、なければ`current_model_maintained`。帰属先target。
- それ以外: `absorb_assigned_training_samples_into_held_model(model_id=previous_model_id, ...)`。結果`candidate_rejected`、帰属先previous_model_id、変更記録None。

旧の再利用は現在ID切替え→吸収の順。新は吸収→切替えにして、吸収の拒否時に現在IDを変えない。独立したownerの更新で、成功時の状態は旧と同じ。private改変・並行更新・MemoryError等の途中例外へのrollbackは提供しない。

採用以外では候補・管理器・特徴・ラベル・待機・2計数・採番owner・送信保留を読まず、検証もしない（呼出し側が候補を破棄する）。

## 旧との対応（testの対応表）

|新resolution_outcome|旧action|旧戻り値|
|---|---|---|
|candidate_adopted_as_new_model|create|2|
|held_reference_model_reused|reuse|1|
|current_model_maintained|maintain|0|
|candidate_rejected|create_rejected|0|

## 依存とファイル計画

許可importは次の16symbolと`__future__.annotations`（数に含めない）だけ: (1)dataclasses.dataclass、(2)torch.Tensor（注釈用）、(3)PostAlarmCandidateLossEvaluation、(4)ResidualAdapterClassifier、(5)ParameterOptimizerState、(6)HeldModelTrainingStateRegistry、(7)ModelAndClassLossStatisticsStore、(8)ModelTrainingSampleStore、(9)ObservedTrainingSample、(10)ModelTrainingAndAssignmentCountsStore、(11)CurrentTrainingModelAssignment、(12)TrainingModelAssignmentChange、(13)TemporaryModelIdAllocator、(14)PendingModelUploadState、(15)adopt_candidate_as_current_training_model、(16)absorb_assigned_training_samples_into_held_model。合計16symbol。exact AST guardをgeneric runtime許可より前に適用。評価関数evaluate_candidate_using_post_alarm_losses・損失収集・設定・登録関数の直接呼出し・登録確認・学習実行・予測重み・Tensor以外のtorch・乱数・旧実装・private・module import・再exportは禁止。

|ファイル|役割|
|---|---|
|runtime/post_alarm_candidate_validation_resolution.py|評価結果に応じた採用/吸収/現在ID切替えの選択と結果record|
|tests/refactoring/test_post_alarm_candidate_validation_resolution.py|実旧確定処理4分岐の対照・拒否/不変・順序・確定後学習の接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- oracle: 上流の採用oracle（build_local_adoption_oracle）の実旧clientと実ForwardValidationSessionへ、損失列と履歴平均を与えて実_finalize_forward_validationを実行する（4分岐とも実行可能なことは上流testで確認済み）。新側は同じ損失列・履歴平均・閾値（旧distance_threshold、旧NEW_MODEL_EARLY_STOPPING_MIN_DELTA）・保有ID・現在IDを移植済みの評価関数へ与えて評価結果を作り（test-only接続）、旧の判定acceptedと一致することを確認してから本関数へ渡す。手順をtestへ再構成しない。
- Task1: 実装より前にREDを実行して記録する（未実装moduleのimport失敗）。GREEN後、実装を何もしないstubと誤実装（結果種別の取り違え、再利用で現在IDを切り替えない、棄却で吸収しない、採用で概念を計数する等）へ一時的に差し替えてtestが失敗することを確認し、元へ戻してhashを照合する。class2/4×4分岐×保留標本0/複数で、全状態（上流のassert_local_adoption_matches_legacy／assert_absorption_matches_legacyと保有一覧・値・送信保留・採番次値）、結果種別と旧actionの対応、帰属先、変更記録と旧通知引数を照合。1.5の不変。2.1〜2.3の拒否で全状態不変（各結果種別で、委譲先の拒否代表入力を含む）、再利用での呼出順（吸収→現在ID）。AST注入RED→16symbol exact guard GREEN。品質/型/独立Luna/主担当gate。
- Task2: 12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）×結果種別（採用・再利用）の実NNで、共同更新→実旧確定処理/新の評価と確定→各自の標本storeを使う共同更新を実旧と照合。fresh新CPUで旧importなしの評価→確定（4種別）→学習。独立Luna/主担当gate。
- Task3: 固定環境全pytest（旧11/最終3golden、主担当実測＋JUnit、基準はsteering/agent-handoff.md）、Ruff/Pyright/pip/diff、固定旧差分空、承認/source hash。独立Luna/主担当gate後、別feature最終GO。
