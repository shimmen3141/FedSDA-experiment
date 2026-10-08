# 命名 revision1
## 公開・永続状態
| 名前 | 役割・型・更新 |
| --- | --- |
| adahedge_diagnostic_evidence_collection.py | 数値式を持たず単一証拠をglobal/真の概念別へ保持 |
| AdaHedgeDiagnosticEvidenceCollection | 一つのclientの診断用owner集合。モデルや帰属ownerを持たない |
| _global_diagnostic_evidence | global一つのAdaHedgeDiagnosticEvidence。再始動対象 |
| _true_concept_diagnostic_evidence_by_id | 真の概念int ID→AdaHedgeDiagnosticEvidence、初回取得時に登録 |
| global_diagnostic_evidence | liveなglobal ownerへの読取り窓口。コピーでない |
| get_true_concept_diagnostic_evidence | 指定した真の概念ID用live ownerの初回生成/再取得 |
| true_concept_id | 診断専用の真の概念ID。モデルID/予測context IDとは別 |
| created_true_concept_ids | 生成済みIDの挿入順tuple。不変snapshot |
| training_assignment_diagnostic_notification.py | 確定帰属変更の値を診断へ通知するruntime |
| notify_diagnostics_of_training_assignment_change | ID検査後にglobalだけを再始動。判定/帰属切替はしない |
| assignment_change | 既存TrainingModelAssignmentChangeまたはNone。IDの単位はモデル識別子 |
| diagnostic_evidence_collection | 対象Collection、exact型。保存/予測機能とは区別 |
| model_id | ID検査用builtin int、負値可。既存名を同じ意味で再利用 |
## testと局所名
| 名前 | 役割 |
| --- | --- |
| test_held_diagnostic_evidence_matches_real_legacy_notifications | 実旧の取得・update・通知列の全値比較 |
| test_diagnostic_collection_keeps_distinct_live_owners | 同ID再取得/別ID独立/live更新/tuple不変 |
| test_diagnostic_notification_rejects_invalid_input_before_restart | 不正通知の更新前拒否 |
| test_diagnostic_collection_rejects_invalid_concept_id_without_creation | bool/不正概念IDで生成しない |
| test_diagnostic_notification_preserves_random_states | 3種類乱数が変わらない |
| test_held_adahedge_diagnostic_exact_dependency_contract | 2moduleのexact依存注入 |
| build_legacy_diagnostic_client | 実旧最終通知mixinを持つclientを__new__し、必要状態だけ与える |
| get_diagnostic_collection_snapshot | global/生成済み全oracle/ID順の読取り値snapshot |
| assert_diagnostic_collection_matches_legacy | 全ownerの証拠と計数を実旧へ比較 |
| legacy_client | 実旧client（global/概念別/現行ID） |
| diagnostic_collection | 新Collectionのtest変数 |
| legacy_concept_evidence | 比較する旧oracleのAdaHedgeRouter |
| concept_evidence | 比較する新oracleのAdaHedgeDiagnosticEvidence |
| concept_ids_before | 生成済みIDの取得済みtuple |
| invalid_notification_case | 通知の拒否条件名 |
| invalid_concept_id | 拒否する真の概念ID入力 |
| expected_exception | 期待する例外型 |
| first_collection | 独立性比較の最初の集合 |
| second_collection | 独立性比較の次の集合 |
| assignment_changes | 確定ID変更の操作列 |
| notification_arguments | runtime呼出しkeyword値 |
| diagnostic_states_before | 拒否/不変を検証する以前の状態 |
| diagnostic_states_after | 操作後の状態 |
| legacy_current_model_id | 旧通知wrapperの前のID |
| acquired_concept_ids | tupleの保存値 |
| observed_losses | modelID別の観測損失dict |
| diagnostic_weights | モデル別の観測前診断重み |
| model_ids | modelID列。既存名と同じ役割 |
| concept_id | loop中の真の概念ID。true_concept_id引数と同じ単位 |
| class_count | test条件。モデルclassや型の数と混同しない |
| source_text | 注入するmoduleの文字列。既存guard名再利用 |
| expected_acceptance | 注入の期待boolean。既存guard名再利用 |
既存testのassert_adahedge_matches_legacy/get_adahedge_evidence_snapshot/diagnostic_evidence/legacy_router/legacy_weights/weights/losses/case/monkeypatch/random_states_before/random_states_after/before/after/index/resultは元と同じ役割で再利用する。__init__/selfは標準constructor/instance。未登録束縛が必要なら実装前に追加承認へ戻す。
