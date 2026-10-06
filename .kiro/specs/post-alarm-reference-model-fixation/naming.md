# 命名 revision 1

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
