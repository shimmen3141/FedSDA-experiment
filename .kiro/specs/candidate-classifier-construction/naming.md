# 命名 revision1

| 名前 | 役割・型・更新/所有 |
| --- | --- |
| candidate_classifier_construction.py | 候補分類器とoptimizerの生成組立。採否・学習を含まない |
| IndependentCandidateTrainingState | frozen record。独立した候補と二つの管理器のbindingをまとめる。保有登録recordと異なりIDなし |
| create_independent_candidate_training_state | 検査済み初期値を候補へコピーし学習状態を生成。既存状態変更なし、モデル生成のtorch RNG消費あり |
| architecture_reference_classifier | 構造を参照する既存exact分類器。候補の初期値をこの値から選ぶという意味ではない |
| initial_candidate_parameter_snapshot | 選択済みのnative新parameter名→CPU float32 Tensor。入力借用、更新なし |
| parameter_optimizer_settings | exact Adam/SGD固定設定。共有部/概念部の両方へ適用 |
| candidate_classifier | 新規分類器。候補の共有部も独立所有 |
| candidate_shared_parameter_optimizer_state | 候補自身の特徴抽出部のoptimizer管理器。client共通共有optimizerと混同しない |
| candidate_concept_specific_parameter_optimizer_state | 候補adapter/分類層のoptimizer管理器 |
| _validate_candidate_construction_inputs | 型・初期snapshot・設定の生成前検査。検査済みoptimizer設定を返す |
| expected_parameter_snapshot | 構造参照の独立snapshot、期待key/shape用。初期値には使わない |
| parameter_name / parameter_values / expected_parameter_values | 検査対象のnative名/初期値/期待shape参照。parameter単位 |
| validated_parameter_optimizer_settings | 固定fieldを再検査した設定 |
| concept_specific_parameters | adapter→分類層順のtuple[Parameter,...] |
| build_candidate_construction_oracle | testのみ、同じ値の参照と実旧client/選択済みsnapshotを準備 |
| create_legacy_candidate / assert_candidate_matches_legacy | testのみ、実旧生成の実行と全値/optimizer/RNGの比較 |
| optimizer_variant / class_count / initial_rng_state | test条件、方式名/クラス数/同じ生成開始時torch RNG |
| construction_arguments / candidate_training_state / legacy_candidate | testのみ、呼出引数/新返却/旧生成結果 |
| input_features / observed_class_labels / batch_index | test学習接続の特徴/ラベル/逐次batch番号 |
| parameter_storage_addresses / initial_parameter_snapshot / invalid_case | test独立性・拒否入力の一時変数 |

既存部品の正式名は既存specの命名承認を再利用する。意味が異なる新しい名前が必要になった場合はrevisionを増やしてレビューへ戻す。
