# IMPROVE-032の手元検証

製品の既定を基点に、警報の後の「新しいモデル」の測定だけを差し替える。
製品の`src/`・既定の設定・goldenは変更しない。本検証の採否判断は別途行う。

| 指定する変種 | 扱い |
| --- | --- |
| `product-default` | 現在の製品の既定。各kの末尾を古い順に一から測る |
| `default-reverse-prequential-grid5` | R5。最新から1本の流れ、最小件数の倍数のk |
| `default-reverse-prequential-grid1` | R1。最新から1本の流れ、最小件数から窓長まで1件刻み |
| `default-reverse-prequential-grid1-position-cost` | R1＋位置。切替の仮説だけにln(候補数)を加える |

既存CLIの`default`は過去の設定に戻す変種なので、この比較の対照には`product-default`を使う。
最小変化区間は製品のまま5件で、件数不足の警報を保留する判定も変更しない。

## 測定と記録

`reverse_prequential.py`のcontext managerをrunの間だけ差し込む。
逆順の全窓を製品の`compute_*_learner_prequential_code_length`へ渡し、製品の
評価関数の戻り値を観測して途中和を保存する。共有部の固定・optimizer・batch・予測と更新の順序は
製品の関数そのもので、optimizerの状態を標本間で維持する。
初期化のseedは`client_id:alarm_sample_index`のUTF-8 SHA-256の先頭8 byteをbig endianで読む。
製品の乱数の隔離処理を使い、runのPython・NumPy・Torchの乱数は消費しない。
両学習者を全窓で測り、各kで短い方＋ln 2を使う。

保持モデルの損失評価と切替の基本費用は製品runtimeを使う。
比較部は製品と同じく、現在のままを最初とし、kの昇順・保有の順・新しいモデルの順に、
厳密に短いときだけ置き換える。位置の費用は基本の切替費用に足してから従来の順で加算する。
R5でも流れは窓全体を処理する（窓31件なら31回、候補の最大は30件）。

`results.json`の各runに次を保存する。

- `prequential_measurement.learner_update_count`: 測定用学習者の実際の`optimizer.step()`の合計。
- `prequential_measurement.learner_update_counts_by_name`: 初期化し直す学習者とwarm startの内訳。
- `prequential_measurement.selections`: 警報ごとのclient・標本番号・窓長・候補k・選択k・仮説・更新数。
- `alarm_interval_assignment_selection`: 製品の同名記録の全列。各列名は
  `alarm_interval_assignment_selection.change_interval_sample_count`等のまま。

更新数はMACとは別の計算量の指標で、測定用の更新に限定する。従来からの精度・上下通信byte・MAC・秒も保存する。
窓31件なら、既定は学習者ごとに105回（合計210回）、R系は学習者ごとに31回（合計62回）。
run全体の警報回数・窓長は変種によって変わり得るので、run合計だけで1警報あたりの費用を推定しない。

## 検査

worktreeルートのPowerShellで実行する。

```powershell
$env:PYTHONPATH = 'src;.'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
../../venv/Scripts/python.exe -m pytest tools/research_provisional_candidate_expert -q --override-ini 'python_files=checks_*.py' -p no:cacheprovider
../../venv/Scripts/python.exe -m pytest tests/test_regression.py -q -p no:cacheprovider
```

専用checksは、逆順の途中和と各kの独立測定の完全一致（2・4クラス、両学習者）、
R1の候補、位置費用の3仮説への作用、製品との比較・同値の優先順、実際の更新数、
乱数不変とpatchの復元を検査する。

## 本検証のコマンド（未実行）

4変種×4データ×2間隔×4seed＝128run。各runは10client×5000標本。
workersは`$workers`で指定する。以下は手元用の4workersの例。
出力先は毎回新しくする。途中から再開するときだけ、同じ出力先・同じ引数に`--resume`を加える。

```powershell
$workers = 4
$output = "results/fedsda-reverse-prequential/$(Get-Date -Format yyyyMMdd_HHmmss)_local-n5000"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:FDE_MNIST_DATA_DIR = (Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m tools.research_provisional_candidate_expert.run `
  --output $output --datasets mnist4 sine2 sea4 circle2 `
  --intervals 50 200 --seeds 0 1 2 3 --sample-count 5000 `
  --variants product-default default-reverse-prequential-grid5 `
    default-reverse-prequential-grid1 default-reverse-prequential-grid1-position-cost `
  --workers $workers
```

## 2026-10-29の検証記録

基点HEADは`ebdcc5a1c644b6b470124d1004d5aa039cbfcfd4`。変更は未コミット。
既存の`.tmp/`は今回の作業前から存在する。
研究ツール段階のため、steeringの`tech.md`に従いcc-sddのspecは作らない。
実装の独立CLIレビュー・検査・smokeの結果は以下へ追記する。

smokeの出力先は`results/fedsda-reverse-prequential/20261029_local-smoke/results.json`。
MNIST4・間隔50・seed0・10client×500標本。
通常CLIのProcessPoolはこのセッションのsandboxでWindows Pipeの作成が拒否されたため、
同じ`_execute_job`を4変種について同一processで順に呼んで保存する。並列CLIの完走確認は未実施。


4runすべて完走し、選択記録と診断のclient・警報標本番号・k・仮説が全件一致した。

| 変種 | 精度 | 測定用更新（合計） | 上りbyte | 下りbyte | 全MAC（推定逆伝播込み） | 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 既定 | 89.12% | 210 | 509463016 | 510987760 | 392122824576 | 66.76 |
| R5 | 89.12% | 62 | 509463016 | 510987760 | 391736670080 | 64.70 |
| R1 | 89.18% | 98 | 509293600 | 509293600 | 388828105344 | 63.44 |
| R1＋位置 | 89.18% | 98 | 509293600 | 509293600 | 388828105344 | 60.93 |

既定とR5は窓31件の警報1回で、どちらも新しいモデル・k=5。R1系は窓31件・18件の警報2回で、
新しいモデル・k=27、保持モデル・k=17。R1系の更新98回は2×(31＋18)。
全体精度は予測した5000標本の累積精度。秒は各runの実測で、同時に別の検査が走っており速度の比較には使わない。
小規模の機能確認なので、効果の採否は未判断。本検証128runは未実行。

専用checksは10件成功（4.78秒）。`tests/test_regression.py`は1件成功（117.69秒）。
新規2PythonファイルのRuff checkとformat checkも成功した。`git diff --name-only -- src tests environments`は空。
本検証用の`--print-plan`は128run・出力先未作成を確認した。本検証自体は実行していない。

独立レビューは未取得。NN数値・RNG・更新順序を扱うためHaiku 5.5を優先し、
`claude -p ... --model claude-haiku-5-5 --effort medium --output-format json --no-session-persistence --tools Read,Glob,Grep --permission-mode plan`
を起動したが、`is_error=true`・`modelUsage={}`・`ECONNREFUSED`で終了した。
sessionは`fd6dc7cd-4818-4c54-a1ca-47a2faebd569`。約182秒で、実際のモデルでのレビューは行われていない。
代替の`codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox read-only --ephemeral ...`も、
app-server clientの初期化がアクセス拒否（os error 5）で終了した。自己承認では代替しない。
Claudeが通常環境でこの未コミット差分をレビューしてからcommitする。

対象のSHA-256（上から実装・専用checks・既存CLI）:

```text
reverse_prequential.py         11E7BD3898948126EB4E9F9ECBD4D6A3365DE7E6B2691B52EB0058BBB40DF100
checks_reverse_prequential.py  1FA396D86BDBB6EC6F77EB001542CF404EE8916B4A2D203AF6D108010F0CBC2E
run.py                        A7F63C4797D6ABA05E9630675848D887304C23BEF2232BC6BF9BEF30AF6FEF1F
```

既存研究検査の通常実行は、当時の専用検査9件を含む82件中65件成功・17件setup error（613.67秒）。
errorは全て既存の`checks_loss_monitor_rebaseline.py`の`tmp_path`作成時のWinError 5だった。
`--basetemp`をworkspace内へ指定しても、pytestが作るパスの読取りが拒否された。
既存検査のsourceを変更せず、検査process内だけ`tmp_path`を、通常の`Path.mkdir`で作るworkspace内の一意な
ディレクトリへ差し替えて再実行する。権限設定・認証設定は変更していない。
その15件は全て成功（250.94秒）。既存検査は合計71件成功、並列2件は未検証。
現在の専用検査10件と合わせて81件成功（重複実行分は数えない）。
並列の2件は、別途smokeで実際に確認したPipeの作成拒否と同じ経路なので再実行しない。
対象は`test_parallel_workers_apply_variant_and_write_diagnostics`と
`test_combined_variant_product_sweep_records_limit_and_rebaselines`。この2件は通常環境での確認が残る。
