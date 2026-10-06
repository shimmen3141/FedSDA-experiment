# 設計 revision 2

## 配置と責務

runtime/adopted_candidate_local_adoption.pyへ状態を持たない組立関数を置く。採番・計数・標本・現在ID（学習層）と初期ローカル登録（runtime、内部でFedSDAの送信保留を使う）を横断するため、登録・登録確認と同じ外側runtimeに配置する。runtime内の既存関数を呼ぶ同層依存で、下位層からruntimeへの依存は作らない。

## APIと順序

`adopt_candidate_as_current_training_model(*, temporary_model_id_allocator:TemporaryModelIdAllocator, adopted_candidate_classifier:ResidualAdapterClassifier, candidate_concept_specific_parameter_optimizer_state:ParameterOptimizerState, initial_statistics_input_features:Tensor, initial_statistics_observed_class_labels:Tensor, upload_delay_round_count:int, candidate_trained_sample_count:int, candidate_parameter_update_step_count:int, pending_assignment_training_samples:tuple[ObservedTrainingSample, ...], held_model_training_state_registry:HeldModelTrainingStateRegistry, loss_statistics_store:ModelAndClassLossStatisticsStore, training_sample_store:ModelTrainingSampleStore, model_training_and_assignment_counts_store:ModelTrainingAndAssignmentCountsStore, current_training_model_assignment:CurrentTrainingModelAssignment, pending_model_upload_state:PendingModelUploadState)->TrainingModelAssignmentChange`

状態を変更しない段階:

1. 採番owner・標本store・計数storeをexact具体型と検証（TypeError）。`candidate_trained_sample_count`と`candidate_parameter_update_step_count`を、exact int（bool・派生型でない）でなければTypeError、負ならValueErrorと検証する。保留標本をexact tupleかつ各要素exact ObservedTrainingSampleと検証（TypeError）。現在IDのownerをexact具体型と検証（TypeError。3の比較で読むため）。これは6・7で呼ぶ既存APIの全拒否条件と同じである: `record_completed_model_training`はmodel_idがexact intでない場合と、2つの増分がexact intでない/負の場合だけ拒否し、`append_model_training_samples`はmodel_idがexact intでない、列がexact tupleでない、要素がexact ObservedTrainingSampleでない場合だけ拒否する（model_idは採番ownerが返すexact int）。
2. `temporary_model_id = allocator.next_temporary_model_id`（消費しない）。
3. 標本storeのsnapshotに一時IDのcollectionがあればValueError。計数storeのsnapshotの3対応のいずれかに一時IDがあればValueError。現在IDが一時IDと同じならValueError。学習状態一覧・損失統計store・送信保留の対応IDでの使用済み確認は、4の登録関数が自身の状態変更より前に行いValueErrorで拒否する（登録の要求2.2）ため、本関数では重複して行わない。これで一時IDをkeyに書き込む全6owner（一覧・統計・送信保留・計数・標本・現在ID）の使用済み確認が、最初の状態変更より前に揃う。

状態を変更する段階:

4. `register_adopted_candidate_as_temporary_held_model(temporary_model_id=..., 候補, 管理器, 特徴, ラベル, 待機, registry, 統計, 現在ID, 送信保留)`。登録関数は、引数・owner型・使用済みID・保有0件の検証、損失評価・初期統計・snapshotの生成を、共有反映より前に行う。続く共有反映APIは候補・管理器・反映先を事前検証してから値を変更し、その後の一覧登録・統計設定・送信保留登録は検証・生成済みの値だけを受ける（登録の設計revision2と要求2.6/2.7、拒否47条件で全状態不変を確認済み）。したがって登録の入力拒否では、登録先ownerも本関数のownerも変更されておらず、採番も未消費である。登録の状態変更段階でprivate改変・メモリ不足等の想定外例外が起きた場合の部分更新は登録の契約どおり巻き戻さず、その場合本関数は5以降を実行しない。
5. `allocator.allocate_temporary_model_id()`。戻り値は2の値と同じ（失敗しない）。
6. `counts_store.record_completed_model_training(model_id=一時ID, trained_sample_count=..., parameter_update_step_count=...)`。
7. `training_sample_store.append_model_training_samples(model_id=一時ID, training_samples=...)`。
8. `assignment_change = current.assign_model_for_training(model_id=一時ID)`。3により必ず実変更。これを返す。

6・7の入力は1で検証済みで、既存APIの検証条件（exact int/非負、exact tuple/要素型）を満たす。private改変・並行更新・MemoryError等の途中例外へのrollbackは提供しない。

旧は採番→登録→計数→待機→標本→現在IDの順。新は登録の検証が通るまで採番を確定しない。成功時の採番値と全状態は旧と同じで、4〜8は互いに独立したownerの更新なので順序の差は観測値を変えない。現在IDの切替えは旧と同じく最後に行う。

## 依存とファイル計画

許可importは次の13symbolと`__future__.annotations`（symbol数に含めない）だけである: (1)torch.Tensor（注釈用）、(2)ResidualAdapterClassifier、(3)ParameterOptimizerState、(4)HeldModelTrainingStateRegistry、(5)ModelAndClassLossStatisticsStore、(6)ModelTrainingSampleStore、(7)ObservedTrainingSample、(8)ModelTrainingAndAssignmentCountsStore、(9)CurrentTrainingModelAssignment、(10)TrainingModelAssignmentChange、(11)TemporaryModelIdAllocator、(12)PendingModelUploadState、(13)同層runtimeのregister_adopted_candidate_as_temporary_held_model。exact AST guardをgeneric runtime許可より前に適用。module丸ごと/再export/private/Tensor以外のtorch/NumPy/乱数/設定/旧実装/候補採否/損失評価/共有反映の直接呼出し/登録確認/評価標本storeは禁止。

|ファイル|役割|
|---|---|
|runtime/adopted_candidate_local_adoption.py|採用時の採番・登録・計数・標本・現在ID切替えの組立のみ|
|tests/refactoring/test_adopted_candidate_local_adoption.py|実旧確定処理の採用分岐対照・拒否/不変・順序・後続学習接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- oracle（実行可能性は設計時に確認済み、research.md）: 上流の登録oracleが作る実旧`SharedBackboneClassConditionalESRFedSDAClient`へ、採用分岐が読む属性だけを与える。`_forward_validation`（実ForwardValidationSession: 候補、training_x/y、held_data、reference_models=保有モデル、target_count、候補の学習量）、`next_temp_id`、`distance_threshold`、`provisional_model_decisions`、`model_training_examples`/`model_optimizer_steps`（defaultdict）、`train_data_store`（defaultdict(list)）、`local_switch_positions`、`adaptation_events`、`model_upload_delay_rounds`、`detection_episodes`（無効のDetectionEpisodeController）、`_on_local_model_change`（呼出しを記録するstub）。config.NEW_MODEL_CREATION_POLICYは最終構成のforward_persistent。sessionへ候補損失が小さく参照損失が大きい損失列を実append_lossesで与え、実`_finalize_forward_validation`を呼ぶ。採用分岐に続く判定記録・switch位置・適応イベント・session破棄・drift resolutionも実行されるが、本specはそれらを比較対象にせず、採用と判定されたこと（戻り値2、判定recordのaccepted）だけを前提確認に使う。比較する状態は一時ID、models順、全値/optimizer、model_stats、pending/待機、学習計数、train_data_store、current_model_id、next_temp_id、`_on_local_model_change`へ渡された(旧ID, 新ID)。手順の再構成による代替oracleは使わない。
- Task1: RED module未実装→GREEN。上流の登録oracleを土台に、class2/4、保有一覧、既存別ID保留、保留標本0/1/複数件、計数0/正、連続2回の採用で、一時ID・一覧・共有部/全モデル値・統計・保留/待機・計数・標本列/順序/payload identity・現在ID・返す変更record・採番ownerの次値を実旧と照合。既存モデルの統計/割当概念計数/標本/計数/評価標本と乱数の不変。拒否時不変は、本関数の検証（2.1〜2.3: 3owner＋現在IDownerの型、2計数の型/負、保留標本の列型/要素型、標本store・計数store・現在IDでの使用済みID）と、登録が拒否する代表入力（2.4: 一覧・統計・送信保留での使用済みID、保有0件、待機ラウンド数、登録先owner型、候補型、管理器不一致、特徴/ラベル不正、非有限parameter）の各点で、採番ownerの次値・一覧record identity・共有部値と接続先・全parameter/grad・全optimizer・統計・送信保留・標本store・計数store・評価標本・現在ID・乱数を観測する。保留標本の追加で既存モデルと新モデルの損失統計・割当概念計数が変わらないこと（実旧も採用分岐では変えない）を実旧のmodel_stats/model_concept_countsと照合する。呼出順（登録→採番→計数→標本→現在ID）を観測（2.5）。AST注入RED→exact guard GREEN。品質/型/独立Luna/主担当gate。
- Task2: 12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）の実NNで、共同更新→候補の独立学習→実旧確定処理/新採用→採用後の共同更新（新側は標本storeの内容と現在bindingを使う固定batch）→正式ID確認→共同更新を実旧と照合。fresh新CPUで旧importなしの採番→採用→後続学習→確認。独立Luna/主担当gate。
- Task3: 固定環境全pytest（旧11/最終3golden、主担当実測＋JUnit、基準はsteering/agent-handoff.md）、Ruff/Pyright/pip/diff、固定旧差分空、承認/source hash。独立Luna/主担当gate後、別feature最終GO。
