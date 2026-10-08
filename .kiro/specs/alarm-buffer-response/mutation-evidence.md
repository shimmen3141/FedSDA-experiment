# 実source変異の検出力

2026-10-08、Task 3。元checkoutの`venv/refactoring-tests/alarm-buffer-response-mutations.py`で実runtimeを一時変更し、対象64テストを実行した。各回に構文成立を確認し、pytest exit1・意味上の失敗・収集エラーなしを確認した。finallyで元byteへ復元した。変異版はコミットしない。

| 変異 | 失敗数 | 主な検査 |
| --- | --- | --- |
| active sessionを無視 | 8 | prepare/resolve禁止、同じsession |
| 前区間を省略し全件を変化区間にする | 27 | 評価保存・統計・帰属・乱数 |
| 最小件数ちょうどを不足にする | 14 | 実旧の境界結果 |
| active sessionを同fieldの別recordへ置換 | 8 | session参照一致 |
| 解決へ全FIFOを渡す | 13 | 区間だけの学習・候補・転送tuple |
| Python乱数を余分に消費 | 38 | 明示Randomの最終状態 |
| 解決前にFIFOを消費 | 25 | 保留位置と最終位置の不変性 |
| FIFO消費判断を逆転 | 46 | 旧の消費/保持結果とreadonly property |
| 元proposal位置を上書き | 9 | 候補sessionと転送metadata |

復元後: **64 passed、11.17s、exit0**。runtimeの実byte SHA256は変更前後とも`ec4ebfad03c614fc249c137a34a7e87f492da26dc2d5d7b0b002153a0658b7fa`。`git diff`もruntimeについて空だった。

外部証拠: `venv/refactoring-tests/alarm-buffer-response-mutation-evidence/report.json`、各変異のlog、restored-green.log。独立承認はreview.md/spec.jsonを参照する。全欠陥の検出を保証するものではない。
