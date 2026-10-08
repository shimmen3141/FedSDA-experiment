# 命名 revision3

## 公開境界と所有状態

| 名前 | 型・役割・更新する状態 |
| --- | --- |
| adahedge_diagnostic_evidence.py | evaluationの単一AdaHedge診断証拠。最終予測controllerではない |
| AdaHedgeDiagnosticEvidence | 独立した累積証拠owner |
| get_diagnostic_weights_before_loss_observation | 診断損失の観測前に重みを取得。集合変更だけを同期 |
| update_evidence_after_loss_observation | 入力損失と診断重みで証拠を更新 |
| restart_evidence_after_concept_operation | 呼出側が決めた概念操作後の明示再始動。通知を発火しない |
| cumulative_losses_by_model_id | dict copyによるモデル別累積損失 |
| mixability_gap | 非負の累積mixability gap |
| model_pool_reset_count | 初回・明示再始動直後を除く集合変更回数 |
| concept_operation_restart_count | 空状態を含む明示再始動回数 |
| _cumulative_losses_by_model_id | owner内部の累積損失dict |
| _mixability_gap | 内部gap |
| _model_pool_reset_count | 内部集合変更計数 |
| _concept_operation_restart_count | 内部概念操作計数 |
| diagnostic_weights_by_model_id | 入力または取得結果の診断重みmap。Fixed-Shareとは別 |
| validated_diagnostic_weights | 入力copyと検査済みの診断重み |
| _validate_model_ids | builtin int/非空/重複検査と昇順化 |
| _validate_numeric_mapping | Mapping copy、IDと有限数値の検査 |
| _prepare_synchronized_evidence | 同期後のloss/gap/pool計数の局所候補を生成。owner未更新 |
| _get_evidence_learning_rate | 局所候補からAdaHedge学習率を計算 |
| _commit_evidence | 計算済みの局所候補をownerへ反映 |
| proposed_cumulative_losses | 同期/更新後の累積損失候補 |
| proposed_mixability_gap | 同期/更新後のgap候補 |
| proposed_pool_reset_count | 同期後のpool計数候補 |
| bounded_losses_by_model_id | 0～1へ制限した損失map |
| learning_rate | 現在証拠の学習率 |
| expected_loss | 診断重み付き損失 |
| mix_loss | AdaHedgeのmix loss |
| minimum_cumulative_loss | 累積損失の最小値 |
| minimum_loss_model_ids | 最小損失の同率ID列 |
| unnormalized_weights | 有限学習率の指数重み |
| weight_sum | 通常sumの指数重み総和 |
| active_losses | 重み>0の有界損失列 |
| log_terms | 重み>0のlog weight − eta loss列 |
| maximum_log_term | log-sum-expの安定化基準 |
| log_mixture | log-sum-exp計算値 |
| validated_mapping | 値をfloat化した検査済みmap |

既存名self、model_ids、validated_model_ids、model_id、observed_losses_by_model_id、parameter_name、specified_value、caught_exception、mapping、value、weight、loss、validated_losses、probabilitiesは同じ役割で再利用する。引数と単位はdesign.mdを参照。stdlib import別名は作らない。

## testの名前

| 名前 | 役割 |
| --- | --- |
| assert_adahedge_matches_legacy | 実旧の全対応証拠・計数を比較 |
| get_adahedge_evidence_snapshot | 拒否前後の4状態を取得 |
| test_diagnostic_evidence_matches_real_legacy_sequences | 正常入力列の旧対照 |
| test_diagnostic_evidence_rejects_invalid_model_ids_before_synchronization | ID拒否の先行検査 |
| test_diagnostic_evidence_rejects_invalid_update_before_synchronization | update拒否の先行検査 |
| test_diagnostic_evidence_isolates_copies_and_independent_owners | 入出力copy・owner独立 |
| test_diagnostic_evidence_preserves_all_random_states | Python/NumPy/torch状態全体の不変 |
| test_adahedge_diagnostic_evidence_exact_dependency_contract | 注入依存契約 |
| diagnostic_evidence | 新owner |
| legacy_router | 実旧AdaHedge |
| legacy_weights | 実旧の重み取得値 |
| evidence_snapshot | 拒否前後の証拠snapshot |
| invalid_model_ids | ID拒否入力 |
| invalid_update_case | map拒否条件名 |
| observed_loss_sequence | 観測損失の列 |
| diagnostic_operation | 取得/update/restartの操作条件 |
| independent_evidence | 別owner |
| acquired_losses | 取得済みloss copy |
| acquired_weights | 取得済みweights copy |
| random_states_before | 操作前の3種乱数状態 |
| random_states_after | 操作後の3種乱数状態 |
| FaultingEvidenceMapping | test専用Mapping。列挙の途中で例外を出し、入力読取りの完了前にownerが更新されないことを検証。runtimeには持ち込まない |
| __iter__ | FaultingEvidenceMappingの標準列挙メソッド。正常IDを1つ列挙した後に読取り例外を出す |
| __len__ | 同Mappingの標準件数メソッド。列挙例外test用に件数を返す |
| __getitem__ | 同Mappingの標準値取得メソッド。test用の損失値を返す |

既存testの局所名source_text、expected_acceptance、exception_type、monkeypatch、initial_state、losses、weights、case、new_weights、result、index、before、afterは同じ役割で使う。追加束縛名が必要なら命名承認へ戻す。
