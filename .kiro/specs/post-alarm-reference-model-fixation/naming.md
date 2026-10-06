# 命名 revision 2

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/post_alarm_reference_model_fixation.py|警報時点の保有モデルの値で、候補検証の比較対象（参照）を固定する。parameter snapshot（値のdict。classifier_parameter_snapshot.py）や送信保留のsnapshotではなく、forwardできる分類器を作る|
|FixedPostAlarmReferenceModels|frozen dataclass。固定した参照分類器の対応と履歴平均の対応|
|reference_classifiers_by_model_id|dict[int, ResidualAdapterClassifier]。観測APIの同名引数と同じ意味。保有モデル本体ではなく警報時点の値で固定した独立の分類器|
|reference_historical_mean_losses_by_model_id|dict[int, float]。評価APIの同名引数と同じ意味。全体統計が2件以上あるモデルの保存平均|
|fix_reference_models_at_alarm|参照分類器を生成し履歴平均を取り出してrecordを返す。torch乱数をモデル生成ぶん消費する。自身の状態なし。snapshot_*（値の複製だけ）と区別するためfixを使う|
|held_model_training_state_registry|既存具体型。保有モデルの順序と分類器を読むだけ|
|loss_statistics_store|既存具体型。各モデルの全体統計を読むだけ|
|held_model_training_states / held_model_training_state|一覧snapshotとその一要素|
|parameter_snapshots_by_model_id|dict[int, dict[str, Tensor]]。生成前に揃える各保有モデルの全parameterの複製|
|parameter_snapshot|既存snapshot関数の戻り値（1モデルぶん）|
|loss_statistics|既存store関数の戻り値（1モデルぶん、またはNone）|
|historical_mean_loss|既存選択関数の戻り値（floatまたはNone）|
|held_classifier / reference_classifier|保有モデルの分類器（読取り）/新しく生成した参照分類器|

## testで使用する名前

|名前|役割|
|---|---|
|build_fixation_oracle|上流の採用oracleから新の一覧・統計storeと実旧clientを取り出し、実旧のモデル生成（model_cls）を設定する|
|fixation_arguments|新関数のkeyword引数dict|
|legacy_client / legacy_reference_models / legacy_session|実旧client/実旧の参照複製の戻り値/実旧のsession開始が作るsession|
|begin_forward_validation_in_legacy_client|実旧_begin_forward_validationを、候補の学習だけ無効化して実行するtest helper|
|assert_fixed_references_match_legacy|参照IDの順序・全parameter値・出力・独立性を実旧の参照と対照|
|statistics_case|統計の与え方のtest軸（件数と平均の組合せ）|
|held_model_ids / class_count|上流testと同じ軸|
|invalid_case|拒否入力|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数|

観測・評価・確定・共同学習への接続の名前は上流observation/resolution specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。

## Task1・Task2で追加した名前（revision 2）

productionの追加名はない。

|名前|役割|
|---|---|
|STATISTICS_CASES|統計の与え方のtest軸の定義。モデルIDごとの(全体件数, 保存平均)、Noneは統計未登録|
|set_overall_loss_statistics_in_both_implementations|新の統計storeと実旧のmodel_statsへ同じ全体統計を設定するtest helper。未登録は両実装で保有していないIDへ退避する|
|statistics_by_model_id / observed_loss_count / mean_loss / statistics|設定する統計の対応とその要素|
|fixed_reference_models|新関数が返したrecord|
|start_validation_in_both_implementations|実旧はsession開始（候補の学習だけ無効化）、新は候補（test側で現行モデルと同じ値の独立分類器を用意）→参照の固定→損失収集の開始を行うtest helper。候補の生成と参照の固定を合わせたtorch乱数の消費が実旧と同じことも確かめる|
|current_classifier / candidate_classifier|現行モデルの分類器/test側で用意する候補|
|initial_torch_random_state / legacy_torch_random_state / other_random_states / random_states / numpy_state|同じ乱数状態から両実装を実行して比べるための記録|
|previous_held_model_training_states / current_held_model_training_states / held_parameter_snapshots / reference_parameter_snapshots / optimizer_owners / previous_optimizers / previous_loss_statistics|保有モデル・参照・optimizer・統計の不変確認用の観測値|
|held_parameter_storage_addresses / reference_parameter_storage_addresses / feature_extractors / legacy_backbones|参照が保有モデル・他の参照と実体を共有しないことを確かめるdata_ptr集合と共有部の列|
|legacy_reference_model / reference_classifier / held_parameter / legacy_parameter / held_classifier|対照する1モデルぶんの実旧参照・新参照・保有モデルとそのparameter|
|historical_mean_losses / expected_model_ids|新関数が返した履歴平均の対応/統計の与え方ごとに期待する対応のID列|
|expected_resolution_outcome|統計の与え方ごとに期待する確定の結果種別|
|reference_classifiers|12条件testで不変を確かめる参照分類器の列|
|run_joint_update_in_both_implementations とその局所名、observation_arguments / validation_samples / resolution / resolution_arguments / legacy_drift_type / previous_model_id|上流の観測・確定testと同じ役割|
|post_alarm_reference_model_fixation_cpu_smoke.py / post_alarm_reference_model_fixation_red_evidence.py|共有venvのrefactoring-testsへ置く、旧importなしのfresh CPU smoke/実装をstubと誤実装へ一時差し替えてtestの失敗を確かめる検証script。Git管理外|
|test_post_alarm_reference_model_fixation_exact_dependency_contract|依存境界testの注入契約|
