# Research Log

## 調べたこと（light discovery。既存の製品の拡張）
- 集約の重み: `aggregate_client_models_into_global_models`（`runtime/server_model_registration_and_aggregation.py`）は、柔らかい学習のときだけ`responsibility_training_store.training_masses_by_model_id`を重みにし、それ以外は`snapshot_ordered_model_training_samples()`の件数を使う。参加は「保有かつ重み>0」。共有部の重みは参加モデルの重みの合計。
- 柔らかい学習でも固い割り当ての標本は保持される: `ModelTrainingSampleStore.append_model_training_samples`は、責任度の記録に加えて`_training_samples_by_model_id`へ同じ標本を足す。`reassign_model_training_samples_id`・`remap_model_training_sample_collections`は両方を更新する。`ResponsibilityWeightedTrainingStore.discard`は製品から呼ばれない（grepで確認）。既存のtest（`test_small_soft_run_uses_all_eligible_models_and_preserves_store_mass`）は、追加のたびに責任度の記録の標本数と件数の合計が等しいことを確かめている。
- 既存の構成の固定: `test_soft_training_single_model_prediction.py::test_existing_configuration_mapping_order_identity_and_protected_files`が、既存の全構成の設定の写像・hash・runの識別子・goldenのhashをfixtureで固定している。新しいfieldを既定値の省略の宣言（`omit_default_from_saved_settings`）にすれば、既存の写像は変わらない（`core/settings_serialization.py`）。
- 構成の順: `dwm-neural-network`のtestが「一覧の末尾」を期待している。新しい構成を末尾に足すので、このtestの期待を「`fedsda-soft-training-single-model-prediction`の直後」に替える（既存の構成の順は変わらない）。

## 判断
- 重みの決め方を小さい公開関数へ分ける: 集約の関数の中の数行を置き換えるだけで、要求1.3（1モデルのとき一致）と2.1を単体で確かめられる。集約の残りの手順に触れない。
- 設定はclientの設定に置く: 今の重みの切替もclientの記録の状態（責任度の記録の有無）で決まっており、サーバの設定に置くと、clientの学習の割当方式との組合せの検査が設定の型をまたぐ。
- 学習量と件数の型: 学習量はfloat、件数はint。重み付きの和と割り算は同じ値になる（1モデルのとき一致）。
