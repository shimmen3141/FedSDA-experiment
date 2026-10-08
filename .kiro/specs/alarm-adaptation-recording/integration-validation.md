# 統合検証

## Task1/2 の部品検証

- 初回RED: 対象は未実装moduleで収集エラー、依存注入は35 failed/34 passed。実装後は対象39＋境界2611成功。
- Task2補強後: 対象40＋境界2611＝2651 passed。venv/refactoring-tests/alarm-adaptation-recording-task2-green.log。
- fresh CPU: 2/4class×5経路の10条件成功、旧/test import=0。alarm-adaptation-recording-fresh-cpu.py/.log。直前specのfresh fixtureから、新store/runtimeの接続と全record field/件数/位置のassertを追加。
- 変異22/22検出。各変更をcompileしてから実sourceへ適用、pytestの失敗を確認し、finallyで両sourceを元byteへ復元。alarm-adaptation-recording-mutations.py、alarm-adaptation-recording-mutants-r2/summary.jsonと各log。collection errorは検出に含めない。
- 初回はcompletion再検査削除だけ未検出（39 passed）。基準損失がboolの不正入力を追加して検出を補った。初回matplotlibの終了時cleanupにsandboxの権限エラーがあったため、再実行はTMP/TEMP/MPLCONFIGDIRを既存venv配下へ指定し、全変異を再測定。
- 命名: 新testの64束縛名は登録漏れなし。guard全ファイルでは古い既存名が未登録と出る（同ファイルを既存集合から除く仕様）。今回のguard差分は新しい束縛名を増やさない。
- identity precheck: 4承認hash/固定旧commitと作業ツリー差分はOK。未コミット差分NGはTask2編集中なので期待通り。HEAD c96699cの271 source/goldenパスhash b8a1d4b630426406a779d0e985189037cbd6f9eb21295c1a11b6e7ec9c219673。Task3で測り直す。

| 変異 | 検出 |
| --- | --- |
| skip_record_append | YES |
| skip_switch_position | YES |
| skip_alternative_count | YES |
| skip_current_count | YES |
| save_original_without_revalidation | YES |
| append_before_revalidation | YES |
| skip_outcome_membership | YES |
| skip_id_outcome_equivalence | YES |
| skip_detector_nonempty_check | YES |
| skip_model_id_type_check | YES |
| skip_optional_index_type_check | YES |
| skip_optional_index_nonnegative_check | YES |
| skip_alarm_index_type_check | YES |
| skip_alarm_index_nonnegative_check | YES |
| skip_completion_recheck | YES |
| skip_runtime_append | YES |
| wrong_previous_id | YES |
| drop_change_point | YES |
| drop_episode_id | YES |
| consume_torch_rng | YES |
| consume_python_rng | YES |
| consume_numpy_rng | YES |

復元したsourceのSHA256（byte）:

- src\federated_learning_experiments\evaluation\adaptation_record_store.py: e9589a50d56f1708550f18b06f2547348552b3abd2596b772e1cdc935414cdb4
- src\federated_learning_experiments\runtime\alarm_adaptation_recording.py: 6458d737005b49d0e18a8ac5b3937bd1f45953e87b6dda13cea65050d50592af

全回帰はTask3で未実施。新client/session接続・予測通知・episode操作・新全体runのgolden一致は本specの範囲外。
