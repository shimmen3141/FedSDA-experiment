# Task 4: 代表変異の検出証拠

対象実装・testは`8942c41`。runtimeを一時的に変異させ、対象152テストで各変異を検出した。各変異の終了コードは1で、構文・collectionの失敗ではなく振る舞いの検証失敗である。

| 実sourceへの変異 | failed / passed | 時間 |
| --- | --- | --- |
| 前区間tupleを逆順にする | 104 / 48 | 12.09s |
| 前区間の吸収を評価保存より先に呼ぶ | 1 / 151 | 8.93s |
| 前区間全件の事前損失検査を削除する | 13 / 139 | 8.62s |
| 返却前に明示Randomを余分に1回消費する | 113 / 39 | 17.31s |
| 返却前にFIFOをdrainする | 106 / 46 | 9.64s |
| 診断概念IDの代わりにsample_indexを吸収へ渡す | 102 / 50 | 21.73s |

変異ごとに`try/finally`で元のbyteへ復元し、完全一致を検査した。元と最終復元sourceのbyte SHA256はともに`5d3bcb067489540f6ce27a14d3f23cd13dd3686f8f349d9f28f7679b0b492475`。復元後は**152 passed / 9.29s / exit0**。通常sourceに検証用分岐は残していない。

使用コマンド（worktree root、固定Windows CPU・OMP/MKL各1thread）:

```powershell
../../venv/Scripts/python.exe ../../venv/refactoring-tests/alarm-preparation-mutations.py ../../venv/refactoring-tests/alarm-preparation-mutations
```

scriptは、上記6変異を1つずつ適用し、構文検査後に`python -m pytest tests/refactoring/test_alarm_training_interval_preparation.py -q -p no:cacheprovider`を実行する。保存・吸収はblockの順を交換、事前検査は前区間のfor blockを削除、他は表の1箇所だけを置換する。検出後に元byteを復元し、最後に同じ対象suiteを実行する。

このPCだけの外部資材は元checkoutの`venv/refactoring-tests/alarm-preparation-mutations.py`、`alarm-preparation-mutations/report.json`、各変異logと`restored-green.log`。別PCでは上記手順から再作成する。この検出力の証拠は6つの代表変異に限定し、あらゆる誤実装の検出を保証しない。
