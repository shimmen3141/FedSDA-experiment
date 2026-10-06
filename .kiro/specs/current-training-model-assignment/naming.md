# 命名 revision 2

|名前|型・役割・更新・区別|
|---|---|
|current_training_model_assignment.py|現在の単一学習帰属ID。混合予測の重みや標本storeを持たない|
|CurrentTrainingModelAssignment|ID owner。選択済みIDを保持し選択アルゴリズムは実行しない|
|initial_model_id|int、初期ID。runtimeが0等を明示、採番なし|
|current_training_model_id / _current_training_model_id|int、公開読取り/内部単一状態。負IDを含む、単位なし|
|assign_model_for_training|選択済みmodel_idを現在IDへ設定し変更recordかNoneを返す|
|remap_current_training_model_id|対応表を一段適用、再帰的解決ではない|
|model_id_mapping|dict[int,int]、元→先。全検証後使用、保存しない|
|model_id|int、外側が選択したモデル識別子、モデルobjectではない|
|TrainingModelAssignmentChange|frozen record、帰属IDの実変更。原因/サンプル位置/混合重みなし|
|previous_model_id / current_model_id|変更前/後のint。recordのfieldでありownerの可変状態ではない|
|_validate_model_id|exact builtin intを検証、状態変更なし|
|assignment_change|変更recordまたはNoneの一時値|
|original_model_id / receiving_model_id|mappingループの元/先、検証対象|

## test内の役割

|名前|役割|
|---|---|
|training_assignment|新単一ID owner|
|legacy_client|実旧methodを呼ぶ最小fixture、productionからの依存ではない|
|model_counts_store|既存の独立計数owner|
|local_change_hook|ローカル変更時のMock hook|
|expected_model_id / initial_model_id / selected_model_id|期待値/初期/上位選択済みID|
|invalid_model_id / invalid_mapping / source_text / expected_acceptance|拒否入力/AST注入契約|
|IntSubclass / DictSubclass|exact型拒否用test派生型|
|build_current_assignment_oracle|新ownerと既存旧最小fixtureを構築|
|test_*|契約を記述するpytest名、状態を持たない|

既存fixture関数とassert helperの名前は元spec承認済みの意味で再利用する。

## Task2追加

|名前|役割|
|---|---|
|expected_model_counts_snapshot|現在ID更新前の独立計数snapshot。ID期待値と混同しない|
|test_training_state_owner_dataclass_only_dependency_contract|計数owner/現在ID ownerに共通のdataclass限定AST契約を検証。新owner追加に伴い既存counts専用test名を改める|

stdlib単独smokeは既存のtraining_assignment、assignment_change、model_id_mappingを使用し、import確認はsys.modulesを観測する。
