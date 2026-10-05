# 調査記録

## 調査範囲
fable-method、既存cc-sdd規則/steering/命名レビュー規則に従い、固定旧実装748c3aaと既存新samplerの境界を調べた。

## 根拠
- federated_drift_experiment/clients/base.py:162 `_absorb_into_store`は各標本をappendする。その前後のモデル損失統計/概念診断は保持器の外側。
- 同:706 `apply_server_mapping`は元dict挿入順で一回getし、宛先defaultdictへextendする。評価stored_dataの容量制限/random.sampleは学習train_data_storeには適用されない。
- fedsda.py:311/893 の直接extendは空列でも空キーを作成する。空追加で空モデルを作る公開契約を採用し、空absorbの呼出しを行わない判断は上位へ残す。
- base.py:446 の正式登録はpop/宛先上書きで、サーバ統合の連結と異なる。後続specで必要性/衝突条件を確認する。現時点で通常条件の不具合と判定していない。
- learning/training/model_training_sample_records.pyの既存recordとsamplerを再利用する。Tensor検証・cat・乱数を重複実装しない。
- tests/refactoring/test_model_id_mapped_loss_statistics_selection.pyの最小旧client namespaceで実apply_server_mappingを呼べる。評価store/statistics/modelsを空にし標本再編だけ観測する。

## 決定
一つの保持器がdict[int,list[ObservedTrainingSample]]の構造のみを所有する。
snapshotは既存ModelTrainingSampleCollection tupleへ変換する。payload参照は共有する。
新しい汎用store基底型、容量設定、rollback、正式登録処理は不要。
今回の調査で新たな旧正常系不具合を確認していない。
