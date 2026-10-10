# 命名表: environment-keyed-regression-goldens

sourceの公開する名前は、足さない。testのfileと、基準のfileの名前だけを載せる。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `proposed_regression_goldens/` | `tests/refactoring/` | 環境ごとの、最終構成のgolden（1環境1ファイル） |
| `<os>-<機種>-python<版>-numpy<版>-torch<版>.json` | 上のディレクトリ | その環境のgolden。名前は、`_env`から決まる |
| `test_environment_proposed_regression.py` | `tests/refactoring/` | 実行環境に合うgoldenの選択、旧実装の照合、goldenの作成（`--update`） |
| `make_golden_file_name`・`list_golden_paths`・`find_golden_path`・`build_golden_payload`・`write_environment_golden` | 上のtestのfile | 名前、一覧、選択、作成 |

## 判断が必要な点

- 前のspecの`test_linux_proposed_regression.py`・`proposed_regression_golden_linux.json`は、OSの名前を鍵にした名前なので、置き換える（前のspecの文書は、書き換えない）。
