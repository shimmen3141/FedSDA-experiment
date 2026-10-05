# 命名案 revision 1
意味・実態・類似責務との違いを優先する。既存record名は変更しない。

|種別|名前|役割・型・入出力・更新|
|---|---|---|
|module|model_training_sample_store.py|モデル別に割り当て済み学習標本を保持。評価storeやFIFO未割当位置bufferと区別|
|型|ModelTrainingSampleStore|標本列の構造所有者。ModelTrainingSampleCollectionはsnapshotの一モデルrecord|
|公開method|append_model_training_samples|model_id/tuple標本を末尾追加、None。空tupleも空モデル作成|
|公開method|snapshot_ordered_model_training_samples|初出モデル順・追加標本順のtupleを生成。元構造は変更しない|
|公開method|remap_model_training_sample_collections|一回のID対応で標本列を再編、None。モデル本体や統計は再編しない|
|引数|model_id|bool以外builtin int、負の一時ID可。標本位置ではない|
|引数|training_samples|exact tuple[ObservedTrainingSample,...]、重複可、Tensor借用|
|引数|model_id_mapping|exact dict[int,int]、全項目検証、一回対応|
|状態|_training_samples_by_model_id|dict[int,list[ObservedTrainingSample]]、初出順の内部構造を所有|
|一時|training_sample|型検証する個別ObservedTrainingSample|
|一時|original_model_id / mapped_model_id|一回対応の元ID/宛先ID|
|一時|remapped_training_samples_by_model_id|検証後に作る新dict、最後にstate交換|
|既存型|ObservedTrainingSample / ModelTrainingSampleCollection|payload借用record/一モデルの順序付きtuple record|
|test module|test_model_training_sample_storage.py|保持/追加/一回対応/抽出接続の実旧対照|
|test helper|build_training_sample_storage_oracle / append_legacy_training_samples / remap_legacy_training_samples / assert_training_sample_storage_matches_legacy|CPU標本と最小旧client構築、実旧呼出し、順序/参照照合|
|test helper|snapshot_training_sample_reference_ids|Tensorの等値比較に頼らずrecord/特徴/ラベルのidentityを捕捉|
|test一時|sample_store / legacy_client / legacy_model / training_samples / previous_snapshot / previous_reference_ids / model_id_mapping / sampled_batches / legacy_batches / python_random_generator / legacy_random_state / held_model_ids / batch_sample_count / storage_case|型・値・役割はdesign/testで表す。新旧参照/RNGを独立に保持する|
test関数は検証する観測契約をtest_<behavior>で表現する。pytestの局所param名は上表と既存helper公開引数を再利用する。

補助的なtest局所名: `invalid_value`（拒否される型/値）、`mapping_case`（対応表条件）、`sample_index`（標本位置）、`sample_count`（標本数）、`model_training_samples`（一モデルsnapshot）、`expected_model_ids`（期待初出順）、`global_python_random_state`（旧oracle呼出し前に復元用保存）、`shared_training_sample`（重複参照）、`training_sample_index`（母集団内位置）、`seed`（借用Randomの初期値）。test用型違いsubclassは`TrainingSamplesTupleSubclass` / `ModelIdIntSubclass` / `ModelIdMappingDictSubclass` / `ObservedTrainingSampleSubclass`と実態を表す。
