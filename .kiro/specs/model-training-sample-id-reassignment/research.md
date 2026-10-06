# 調査根拠

- 旧clients/base.py:41–43でtrain_data_store（学習）とstored_data（サーバ評価/容量上限）が別owner。confirm_model_registration:446–449は双方を独立にpop代入する。
- 本specの実旧対照はtrain_data_storeのみ。他ownerを空にした実BaseClient.confirmを使う。stored_dataは評価・容量の契約が異なり、既存学習storeに統合しない。
- model-training-sample-storageは正式登録pop上書きを明示的に後続へ分離済み。既存remap_model_training_sample_collectionsは対応表の一回適用と列連結なので流用しない。
- 元列が空でも登録済みなら先の非空列を空列で上書きする。列長・内容による判断なし。
- 内部list構造はstoreが所有し、公開snapshotはtuple構造分離/recordとTensorの借用。取得済みsnapshotのIDを変更しない。
- 旧の非負current_model_id早期returnは上位条件。汎用signed元IDは直接期待値で検証し、実旧負元ID対照と区別する。
- 新たな旧正常不具合は現時点で観測していない。
