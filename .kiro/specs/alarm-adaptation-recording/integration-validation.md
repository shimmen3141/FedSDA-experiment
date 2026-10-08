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

## Task3 全回帰の主担当実測

source/test commit 9b72182、実行開始時は差分なし。9363 passed / 3 skipped / 2 warnings、165.21s、exit0。JUnitは9366件、failure/errorなし。旧11・最終3の内部ケースを実行する2つのgolden testcaseとも成功。goldenは更新していない。

証拠は元checkoutのvenv/refactoring-tests/alarm-adaptation-recording-full-canonical.log/.xml。最初のsandbox測定は9334 passed/3 skipped/1 failed/28 errors（163.97s）。debuggerの独立XML確認で29件はすべて一時ファイルのPermissionError、最終golden比較に未到達。full.log/.xmlを失敗の記録として保存し、canonical成功と区別する。コード・設定を変えず、主担当execの必要な権限で同じテストを再実行した。レビュー担当のsandboxを外していない。

固定旧748c3aaから旧実装・旧/最終golden・旧回帰test・toolsのcommit/作業ツリーdiffは空。source/golden271パスのLF hashは14816d2b68990d224dc6c8ee4cc62e467e17e60974993f10f863f82cf8e06d1d。source以後は証拠・進捗文書のみ変更。

canonical品質: Ruff check成功、format170 files成功、Pyright共有venv指定で0 errors/0 warnings/0 informations、pip check No broken requirements found（tool chunk b69587/6f169f）。既存warningはfull log参照。

固定環境: ../../venv/Scripts/python.exe、OMP_NUM_THREADS/MKL_NUM_THREADS=1、TMP/TEMP=元checkout/venv/refactoring-tests、MPLCONFIGDIR=元checkout/venv/matplotlib-cache、FDE_MNIST_DATA_DIR=元checkout/data/mnist、PYTHONIOENCODING=utf-8。全pytestは tests -q -p no:cacheprovider --junitxml=<証拠xml>。

| 要求 | 証拠 |
| --- | --- |
| 1.1/1.2/1.3 | 2/4class×5応答の実旧全field・切替位置・計数対照、連続記録 |
| 2.1 | 不正record/completion、更新前拒否・保存copy、検査削除/順序変更変異 |
| 2.2 | snapshotの固定と保存入力copyの独立 |
| 2.3 | 上流所有状態と3種類の乱数不変、RNG消費変異 |
| 3.1 | 実旧イベントの全field・件数・位置対照、旧aliasなし |
| 3.2 | commit済みsource/testの全pytestと旧11/最終3golden成功 |

Task3独立レビュー・別fresh feature最終GOはまだ未承認。
