# 設計 revision 2

## 配置と責務

runtime/adopted_candidate_initial_local_registration.pyへ状態を持たない組立関数を置く。学習層（共有反映・学習状態一覧・統計・損失評価・snapshot）とFedSDAの送信保留を横断するため、登録確認と同じく外側runtimeに配置する。各部品からruntimeへの逆依存は作らない。設定・モデル生成・採番・通知・計数・標本追加を関数へ追加しない。

## APIと順序

`register_adopted_candidate_as_temporary_held_model(*, temporary_model_id:int, adopted_candidate_classifier:ResidualAdapterClassifier, candidate_concept_specific_parameter_optimizer_state:ParameterOptimizerState, initial_statistics_input_features:Tensor, initial_statistics_observed_class_labels:Tensor, upload_delay_round_count:int, held_model_training_state_registry:HeldModelTrainingStateRegistry, loss_statistics_store:ModelAndClassLossStatisticsStore, current_training_model_assignment:CurrentTrainingModelAssignment, pending_model_upload_state:PendingModelUploadState)->None`

状態を変更しない段階:

1. 一時IDをexact intかつ負、待機ラウンド数をexact intかつ1以上、4ownerをexact具体型と検証する。型はTypeError、値はValueError。
2. registry.snapshot_ordered_held_model_training_statesの一覧から、一時IDが保有済みならValueError。statistics.get_model_loss_statisticsがNone以外ならValueError。pending.get_pending_model_uploadの対応IDが一時IDと同じならValueError。
3. 一覧が空ならLookupError。現在の学習帰属IDと同じIDのrecordがあればその分類器、なければ一覧先頭の分類器のfeature_extractorを反映先とする。current_training_model_assignmentは読むだけで変更しない。
4. evaluate_classifier_per_sample_bounded_lossesで候補の損失を一回評価する（候補・特徴・ラベル・出力を検証）。
5. initialize_model_and_class_loss_statistics_from_batchへ損失・ラベル・候補のclass_countを渡し初期統計を生成する。
6. snapshot_classifier_parametersで候補の独立snapshotを生成する（全parameterの有限性等を検証）。

状態を変更する段階:

7. integrate_adopted_candidate_shared_features（候補・管理器・反映先を事前検証→値反映→接続→個別reset）。
8. registry.register_held_model_training_state（一時ID、候補、管理器）。
9. statistics.set_model_loss_statistics（一時ID、初期統計）。
10. pending.queue_model_upload（一時ID、snapshot、待機ラウンド数）。

旧は接続（7）の後に損失評価とsnapshot取得を行う。7は候補共有部の値を反映先へ複写するので、4・6を反映前の候補で実行しても、同じparameter値・同じ構造のforwardと同じ値のsnapshotになる。新モデルにmode/乱数依存層はない。4〜6を前へ置くことで、特徴・ラベル・非有限parameterの拒否時に共有部・接続・optimizerを含む全状態が不変となる。7の事前検証が拒否した場合も、1〜6は状態を変更していないため全状態が不変。8の検証内容は7の検証に含まれ、9・10は6までに生成・検証した値を渡すので、検証済み入力では拒否されない。private改変・並行更新・MemoryError・monkeypatchによる途中例外へのrollbackは提供しない。

snapshotの各Tensorは新規cloneで、候補・反映先のparameterとstorageを共有しない。7の値反映後も値は一致する（複写元が候補自身の値）。候補の共有部が既に反映先と同一objectの場合は、既存共有反映APIの契約に従う。

戻り値はNone。登録結果は各ownerから取得する。計算量診断に必要な標本数は呼出し側が渡した特徴の件数として知っている。

## 依存とファイル計画

許可importは次の13symbolのみ（__future__.annotations許可）: torch.Tensor（注釈用）、ResidualAdapterClassifier、SharedFeatureExtractor（反映先選択helperの戻り値注釈用）、ParameterOptimizerState、HeldModelTrainingState（一覧snapshotを受けるhelperの引数注釈用）、HeldModelTrainingStateRegistry、ModelAndClassLossStatisticsStore、CurrentTrainingModelAssignment、PendingModelUploadState、integrate_adopted_candidate_shared_features、evaluate_classifier_per_sample_bounded_losses、initialize_model_and_class_loss_statistics_from_batch、snapshot_classifier_parameters。exact AST guardをgeneric runtime許可より前に適用。module丸ごと/再export/private/Tensor以外のtorch/NumPy/乱数/設定/旧実装/候補採否/学習実行/標本store/計数storeは禁止。

|ファイル|役割|
|---|---|
|runtime/adopted_candidate_initial_local_registration.py|採用候補の一時ID登録の組立のみ|
|tests/refactoring/test_adopted_candidate_initial_local_registration.py|実旧登録対照・拒否/不変・順序・後続学習接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- Task1: RED module未実装→GREEN。実旧の共有部構成client（最終構成と同じ_prepare_model_for_registration/_register_trained_new_model）へ同じ初期値の保有モデル・候補・標本を構成し、旧登録＋旧FedSDA待機設定と新関数を対照。保有一覧順×現在ID保有/非保有（反映先選択）、class2/4、singletonクラス/欠落クラス、既存別ID保留の置換。共有部値、全保有モデル出力、統計全field、snapshot値/独立性、待機/ready、候補optimizer reset、不変owner（現在ID/既存統計/既存個別parameter・optimizer/共有optimizer state/RNG）を照合。2.1〜2.6の各拒否で全状態snapshot不変、API呼出順を観測。
- Task2: AST注入RED→exact guard GREEN。12条件（class2/4×Adam標準/AMSGrad/SGD×共有更新有無）で共同更新→候補を独立学習→登録→登録後の共同更新を実旧と新で行い、loss/全値/grad/optimizer state/RNGを照合。登録確認（既存confirm）まで接続し、一時ID→正式IDの付替え後も継続できることを確認。fresh新CPU起動で旧importなしの登録→後続学習。
- Task3: 固定環境全pytest（旧11/最終3golden）、Ruff全適用対象/Pyright/pip/diff、固定旧差分空。LF承認hash/source hash/JUnitを記録。全tasks独立承認後に別feature最終GO。候補session・計数/標本/現在ID切替え・new client/runはこのgateに含めない。
