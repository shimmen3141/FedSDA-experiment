# 命名 revision4

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
| test_candidate_construction_matches_legacy | 正常生成を実旧へ照合するtest |
| test_candidate_construction_rejects_invalid_inputs | 拒否前の乱数・参照不変test |
| test_candidate_training_state_record_contract | frozen/kw_only record契約test |
| test_candidate_classifier_construction_exact_dependency_contract | AST許可/拒否注入test。既存source_module_path/source_text/expected_acceptance変数を再利用 |
| test_candidate_initialization_and_updates_match_legacy | 選択済み初期値→生成→3batch更新を実旧へ照合する接続test |
| reference_parameter_values_and_gradients | testの参照/入力parameterの値とgradの不変記録。parameter snapshot（値だけ）と区別 |
| candidate_parameter_optimizer_manager | 比較するParameterOptimizerState管理器。内部optimizer state dictと区別 |
| candidate_parameter_optimizer / legacy_parameter_optimizer | 新旧optimizer実体の比較 |
| expected_rng_state | 実旧生成後のtorch RNG状態 |
| second_candidate_training_state | 二回目の新規候補生成結果 |
| legacy_initial_parameter_snapshot | 旧native parameter名の初期値snapshot |
| initialization_settings / initialization_source | test-onlyの既存初期値選択設定と選択方式名 |
| selected_initial_parameter_snapshot | 選択関数から取得して候補生成へ渡す初期値 |
| candidate_mean_training_loss / legacy_mean_training_loss | 新旧それぞれの単回更新が返す平均学習損失 |
| local_training_settings | test-onlyの既存共同更新固定設定 |

既存部品の正式名は既存specの命名承認を再利用する。意味が異なる新しい名前が必要になった場合はrevisionを増やしてレビューへ戻す。
