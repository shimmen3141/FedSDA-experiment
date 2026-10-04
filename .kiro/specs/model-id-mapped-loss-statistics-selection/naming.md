# 命名: モデルID対応後の損失統計選択
revision: 2

短さより役割の明確さを優先する。statistical mergeではなくwhole recordのselection。model_id_mappingはID対応表で、予測のモデル重みや標本帰属を表さない。

| 名前 | 役割・型・単位・状態 |
|---|---|
| model_id_mapped_loss_statistics_selection.py | ID対応後の損失統計の選択のみ。学習/モデル対応操作全体とは異なる |
| select_loss_statistics_after_model_id_mapping | keyword-onlyで両snapshotと対応表を受け独立tupleを返す。入力/永続状態非変更 |
| local_model_loss_statistics | ローカル所有者の順序付きtuple[(signedint,既存統計)]。モデル全体/class双方を含む |
| model_id_mapping | exactdict[元signedint,変更先signedint]。一回対応。生成は行わない |
| server_model_loss_statistics | 変更先IDで既に整理されたサーバtupleまたはNone。再対応しない |
| _copy_validated_model_loss_statistics | snapshot全検査と深い独立コピー。keyword loss_statistics_snapshot/parameter_nameを受けtupleを返す |
| loss_statistics_snapshot | 検査対象の順序付きsnapshot。単一recordではない |
| parameter_name | エラー対象のlocal/serverなどの入力項目名 |
| _validate_model_id_mapping | keyword model_id_mappingの全key/value検査。返却None |
| validated_local_model_loss_statistics | 検査/コピー済みローカルsnapshot。一時値 |
| validated_server_model_loss_statistics | 検査/コピー済みサーバsnapshot。一時値 |
| selected_loss_statistics_by_model_id | 初出順を保持する結果dict。最大overall件数選択/zero補完で置換。呼出内部のみ |
| seen_model_ids | snapshot重複検出のsignedIDset |
| model_loss_statistics_pair | 検査対象のexact2要素tuple |
| original_model_id | ローカル元IDまたは対応表key |
| mapped_model_id | 対応を一回適用した変更先IDまたは対応表value |
| model_id | snapshotまたはserver内のsignedID。class_idとは異なる |
| loss_statistics | 単一モデルのexact ModelAndClassLossStatistics |
| copied_loss_statistics | publicconstructorで再検査/コピーした単一record |
| selected_loss_statistics | 結果に既にある単一record、未存在はNone |
| validation_error | constructor失敗例外。項目名を添えて再送出、入力非変更 |
| copied_model_loss_statistics_snapshot | 共通copy helper内の検査済みpair列。単一copied_loss_statisticsや用途別validated_local/serverとは区別する |

既存型ModelAndClassLossStatisticsとfield名overall_loss_moments/class_loss_moments_by_class_id/observed_loss_countを維持する。
観測単位のmoments更新/件数計算はこの関数で行わない。test-onlyの旧形式変換・stub名は実装APIではない。

## test-onlyの共通helper
build_model_and_class_loss_statistics_test_seed: 正常な既存型seedを作る。
convert_loss_statistics_snapshot_to_legacy_model_stats: snapshot全fieldを旧dictへ明示変換。
select_legacy_loss_statistics_after_model_id_mapping: 最小stubで実旧apply_server_mappingを直接呼ぶ。
assert_loss_statistics_snapshot_matches_legacy_model_stats: model/class順と全fieldの完全一致を確認。
test_model_id_mapped_statistics_match_legacy_selection: 上記を使う正常oracle比較。
