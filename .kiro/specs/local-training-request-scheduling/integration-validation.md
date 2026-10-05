# 統合検証: ローカル学習要求の保留と実行回数

## 判定範囲
全3tasks完了、Luna最終統合GOを主担当が採用した。全11条件にblockerはない。
学習要求一件の累積・間隔・試行予算算出・正常終了の外側確認後の全件消化だけを移植した。
設定とcounterは独立し、NN/標本/optimizer、自動callback、flush呼出し位置、並行実行、完全run設定は所有しない。
spec.jsonが承認/進捗の正本、review.mdが採否、naming revision1が正式名の正本。

## 環境とコマンド
2026-10-06（日本時間）、Windows CPU、共有../../venv: Python3.13.15、Torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。
基準環境と再構築はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache。

```text
python -m pytest tests/refactoring/test_local_training_request_scheduling.py -q -p no:cacheprovider
python -m pytest tests/refactoring/test_local_training_request_scheduling.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/training-schedule-final-20261006a --junitxml=../../venv/refactoring-tests/training-schedule-final.xml
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pyright --pythonpath <共有venvのpython.exe絶対パス>
python -m pip check
```

## 実測
- Task1 missingmodule collection error/3.53秒/exit1のRED後、37 passed/2.75秒/exit0。
  interval1/2/5×L0/1/3の旧BaseClient実train_step/flush列を比較し、各eventのpending/呼出し予算/順序が一致。
  不正設定/構築前改変/ack拒否・readonly/frozen/kwonly/defaultなし・巨大整数・学習例外の保留維持と明示retryを確認。
- Task2 test-only接続は70 passed/6.37秒/exit0、Luna独立70 passed。
  class2/4×Adam/SGD×interval1/3×L0/2×共有更新/凍結の32条件。
  実旧要求→共同更新と新schedule→実反復→成功確認で、各境界loss/全parameter/grad/optimizer/RNG/予算がexact一致。
  参加者なしの正常な空lossを失敗と扱わず保留消化することも確認。数値本体のmockなし。
- AST24禁止/6許可を先行してRED9 failed/21 passed/439 deselected/0.09秒。
  exact二module/publicsymbolのguardを一般stdlib許可より前へ追加し、GREEN対象+AST539 passed/10.30秒。
  EOF余分空行のLuna通知はRuff format後にgit diff --check成功を確認して解消。
- stdlib smokeは別processのpython -S/PYTHONPATH=srcで成功し、Torch/NumPy/旧packageがsys.modulesに存在しないことを確認。
- fresh CPU接続smokeは算出した4試行を実Adam反復へ渡し、CPU32/全個別step=4/parameter変化/成功後pending0/旧非importを確認、exit0。
  一時scriptはgit管理外の共有venv/refactoring-tests/training_schedule_smoke.py。
- Ruff check成功、format96 files already formatted、Pyright0 errors/0 warnings、pip check整合、diff-check成功。
- 最終全回帰は3821 passed/3 skipped/1既存warning/189.61秒/exit0、旧11/最終3goldenを含む。
  3skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。
- 固定旧748c3aaとの旧production/両golden/旧回帰test差分は空。値・許容差は未変更。
- 検証ソースhashはed52a4c35cbbf42e93d316a8fc1328c8015c5d1302ea100c960a1513833fa23a。
  tracked Pythonと両golden196パスをsortし、各UTF8相対パス+NUL+CRLFをLFへ正規化した内容+NULをSHA256集計。
- 要求/設計/命名の承認LF hash一致・UTF8/配置を確認。tasksの完成checkboxの内容hashはcurrent_sha256_lfへ別記する。
- Luna独立Task3 APPROVED、対象+AST539 passed/6.16秒。独立stdlib/new CPU4-step smoke成功と現在sourcehash一致も確認した。
- 全3task状態とroadmap同期後に、Luna最終feature GO。全11/11要件・実旧32条件・依存/配置/所有・検証hash・残る境界を確認し、blockerなしとの判定を採用した。

## 11要件の証拠
| 条件 | 実装/検証 |
|---|---|
| 1.1 | record+1/間隔判定、旧9条件要求列と32条件実NN接続 |
| 1.2 | pending×Lだけで試行予算、batchや参加者数を掛けない実旧比較 |
| 1.3 | 間隔未満の端数flush/空flush/二重flush、calculate無変更 |
| 1.4 | L0でも要求の累積/正常確認を行い旧range0と一致 |
| 1.5 | 外側実学習正常return後のsnapshot件数ackだけclear |
| 2.1 | metadata数値宣言とexact整数/値域/型/改変設定の構築拒否 |
| 2.2 | bool/派生/負/0/不一致/空の重複ackでcounter保持 |
| 2.3 | 実行例外ではackへ進まずpending保持、明示retry後clear |
| 2.4 | constructor0、frozen/kwonly/defaultなし不変設定とreadonly count |
| 3.1 | 実旧train_step/flush順・予算/counter/失敗/32条件NN exact |
| 3.2 | exactAST、stdlib/newCPU fresh非旧importとtest-only反復接続 |

## 旧所見と後続
今回の通常経路で新たな旧不具合は観測していない。既存LEGACY findingsと旧sourceの状態は変えない。
この同期counterは並行ticketではなく、同件数の過去batch誤確認までは判別しない。
設定の構築後object.__setattr__によるfrozen回避や、counterのprivate改変は通常契約の対象外。
失敗した学習の既済NN更新はrollbackしない。retryは外側が明示する。
完全run設定/CLI、要求を発行する標本処理位置、警報/ラウンドflush位置、新client/全体runは後続。
ローカルgoldenは旧実装固定値の確認、新旧同値性は部品対照による。新全体run golden完成とは主張しない。
GitHubホストのgolden診断はローカル基準の代替にせず、環境差でgoldenを更新しない。
