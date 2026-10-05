# 命名 revision1

意味の明確さを優先する。既存の承認済み名称を維持し、旧名aliasは作らない。

|名前|型・役割・状態更新・区別|
|---|---|
|adopted_candidate_shared_feature_integration.py|採用済み候補の共有学習をactive共有部へ反映するtraining module。候補採否/登録は含まない|
|SharedFeatureExtractor.copy_parameter_values_from|公開モデル操作。sourceから自身へparameter値を転写し参照/gradを維持。attach（参照交換）と区別|
|source_feature_extractor|SharedFeatureExtractor、転写元の借用参照|
|integrate_adopted_candidate_shared_features|公開外側操作、反映→接続→個別reset。登録や学習stepは行わずNoneを返す|
|adopted_candidate_classifier|ResidualAdapterClassifier、採否判定済み候補。接続先だけが変わる|
|candidate_concept_specific_parameter_optimizer_state|ParameterOptimizerState、候補adapter+classification用の既存管理器。現在optimizerだけを交換|
|active_shared_feature_extractor|SharedFeatureExtractor、上位が選んだ現行共有部。候補値を受け取るが参照は保持|
|_validate_adopted_candidate_integration_inputs|private事前検証、副作用前に型/寸法/owner対応を確認|
|candidate_concept_parameters|tuple[Parameter,...]、adapter→classification順の借用parameter列|
|candidate_parameter_optimizer|現在のAdam/SGD、検証だけで更新しない|
|optimizer_parameters|現在optimizer groupsをflattenした借用列|
|shared_parameter_ids|set[int]、両共有部parameterのPython identity（モデルIDではない）|
|parameter, parameter_group, expected_parameter|局所検証ループ。既存parameterと期待対応の比較|
|build_candidate_integration_inputs|test helper、モデル/owner生成。productionの反映先選択は行わない|
|test_adopted_candidate_shared_feature_integration.py|単体とtest-only実旧準備→共同学習対照|

新しい永続状態/型は追加しない。held-model再接続の共有元選択とは別で、activeは呼出側指定。
小さいtest局所変数は既存のcandidate/active/legacy、parameter_snapshots、optimizer_snapshots、training_batches、optimizer_settings、initially_shared、update_shared_features等の意味で使う。
