# 命名・役割一覧

revision: 1。未承認。短さより役割・参照の寿命・既存bindingとの区別を優先する。

|名前|役割・型/入出力・更新|
|---|---|
|held_model_training_state_registry.py|ID別のNN/個別optimizer管理器の保持だけを担うmodule|
|HeldModelTrainingState|frozen状態record。NNと現在optimizer管理器をliveに参照する。生成時optimizerを固定する既存HeldModelTrainingBindingと区別|
|HeldModelTrainingStateRegistry|保有一覧の構造を所有する。NN/管理器は上位が生成して渡す|
|model_id|exact int。負の一時ID/非負IDとも可、採番・対応なし|
|classifier|保有するResidualAdapterClassifier参照。候補採否を意味しない|
|concept_specific_parameter_optimizer_state|現在個別optimizerを所有する管理器。shared ownerを含めない|
|register_held_model_training_state|IDとNN/管理器を登録。同IDはその位置で置換。入力対応を検証しdictだけ更新、None返却|
|get_held_model_training_state|指定IDのreadonly record取得。未登録KeyError、副作用なし|
|snapshot_ordered_held_model_training_states|初出順record tupleを取得。tuple構造が独立、NN/ownerはlive参照|
|snapshot_ordered_held_model_training_bindings|現時点optimizerを参照する既存binding tuple生成。学習はしない|
|_held_model_training_states_by_model_id|registry唯一の永続状態dict[int,HeldModelTrainingState]|
|_validate_model_id|ID型を検証、状態変更なし|
|_validate_held_model_training_state_inputs|NN/owner/個別parameter対応を全検証、状態変更なし|
|held_model_training_state|一覧を反復するreadonly recordの一時変数|
|concept_specific_parameters|adapter→classificationのtuple[Parameter,...]。空/重複/CPU32等を検証|
|parameter_optimizer|ownerが現在保持するAdam/SGD。生成しない|
|optimizer_parameters|groupsをflattenした現在のparameter tuple。上記とのidentity/順序比較|
|parameter_group|現在optimizerのgroup辞書。読み取りのみ|
|parameter, expected_parameter|検証中の現在/期待parameter参照|
|shared_parameter_ids|classifier共有parameterのid集合。個別への混入を拒否|
|registered_state, previous_state, previous_binding|テスト側の現在/置換前record/取得済みbinding。参照寿命検証|
|registry, classifier, optimizer_owner, optimizer_settings|テストfixture側の管理器・NN・owner・既存設定。productionで設定生成しない|
|invalid_case, invalid_model_id|テスト拒否条件/値|
|snapshot, rng_state, parameter_snapshots, optimizer_snapshot|テスト側の取得構造/乱数/値grad/state保存|
|training_bindings, training_batches, shared_optimizer, legacy_client, ordered_training_samples|実旧対照fixtureの借用記録/参加batch/外側共有optimizer/旧namespace/標本列|
|class_count, optimizer_variant, update_shared_features, monkeypatch|既存fixtureのclass数/方式/共有更新フラグ/設定復元fixture|
|concept_owners, model_id, training_batch, training_sample_collection, training_sample|対照内のowner tupleと反復要素。ID対応はテスト側のみ|
|python_random_generator, expected_random_state, expected_losses, actual_losses|明示Randomと旧/新反復の照合値|
|build_registry_classifier_and_owner|テスト補助: class2の既存NNと対応個別ownerを生成して返す|
|register_training_state|テスト補助: classifierとownerを明示IDでregistryへ渡す|
|build_registry_training_oracle_pair|テスト補助: 既存実旧NN/標本fixtureにregistryと現在ownerを接続して返す|
|initial_random_state, global_python_random_state, sampled_batch_history|既存実旧反復へ渡すRandom状態/復元用状態/抽出履歴|
|previous_optimizer, previous_optimizer_state, concept_parameters, parameter_values, parameter_gradients|登録/置換/reset前の借用optimizerと値grad/stateの照合保存|
|replacement_classifier, replacement_owner, old_state, state, binding, states, bindings|同ID置換とsnapshotのテスト記録/実体|
|shared_feature_extractor, participating_training_batches, model_states_by_id|外側共有部/現在optimizer参加記録/ID別照合辞書。テストだけの接続|
|legacy_models, legacy_registration_client, expected_model_ids, registration_ids|実旧登録のnamespaceとモデル列/順序、明示ID列|
|input_features, observed_class_labels, sample_index, step_index|既存NNの実入力・ラベル・標本/更新位置。テスト側で供給|
|IntSubclass|テスト拒否用のint派生型|

テスト関数名は検証する観測内容をtest_*で表す。fixtureの補助名追加が必要なら実装前に本表へ追記してレビューする。
