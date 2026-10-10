# 命名表: linux-proposed-regression-golden

sourceの公開する名前は、足さない（sourceを変えない）。testのfileと、基準のfileの名前だけを載せる。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `test_linux_proposed_regression.py` | `tests/refactoring/` | Linuxでの、最終構成の旧実装の照合と、Linux用のgoldenの作成（`--update`） |
| `proposed_regression_golden_linux.json` | `tests/refactoring/` | Linux用のgolden（既存の`tests/proposed_regression_golden.json`と同じ形） |
| `LINUX_GOLDEN_PATH`・`load_linux_golden`・`build_linux_golden_payload` | 上のtestのfile | goldenの場所、読込み、作成 |

## 判断が必要な点

- 既存の回帰testの語（proposed regression、golden）に合わせ、環境を後ろに付ける（`_linux`）。既存のWindows用のfileの名前は、変えない。
- 置き場所は、`tests/refactoring/`（既存の`tests/`直下のfileは、固定旧実装の回帰testで、変更しない範囲。新しいtestは、ここへ足してきた）。
