# リファクタリング前の回帰基準

## 目的と基準の分離

研究室サーバで実行した論文用実験の結果は、条件・実行環境・来歴を含む成果物として保存する。
開発用goldenは、小規模なローカル実行によって実装変更時の挙動を確認する基準である。
両者は用途が異なるため、サーバの結果をローカルgoldenへ流用しない。

数値を固定する前に、現在の実装・実行環境で既存goldenが通ることを確認する。
別OSでの不一致を理由に、既存goldenの上書きや許容誤差の拡大を行わない。

## 既存goldenと今回の補完

- `tests/test_regression.py` / `tests/regression_golden.json`:
  FedDrift、旧FedSDA、Obliviousなどの代表11ケース。旧goldenの値は変更していない。
  このケース集合の最後の追加は、Residual Adapter・Switching導入より前である。
  後続の機能テストは存在するが、最終提案構成の数値回帰への追加が追いついていなかった。
- `tests/test_proposed_regression.py` / `tests/proposed_regression_golden.json`:
  最終Residual Adapter＋ClassESR＋Switching構成の3ケース。構成の正本は
  [最終提案構成](../overview/proposed-method.md)。旧goldenとは独立して保存する。

| ケース | 規模（クライアント数×各クライアントのサンプル数） | 主な確認対象 |
|---|---|---|
| SINE-2 | 3×1500 | 候補採用・棄却、複数expert、切替、統合、FIFO再較正 |
| SEA-2 | 3×600 | 3次元入力、単一モデル時の実行、通信・計算量 |
| MNIST-2 | 2×100 | 784次元・10クラス、候補検証、モデル切替 |

seedは0、集約間隔は50、距離閾値は0.1。Adapter rank=8、joint mean、
forward_persistent（検証10件）、FIFO replay、average linkage、
class_functional_confidence、on_new_model、mergeを使用する。
学習・ドリフト生成条件を含む縮小設定は新goldenの`definition`に保存する。
これは性能評価用の実験規模ではない。

新goldenは33指標（精度・検出・通信量・parameter/byte数・計算量・再較正件数）と、
予測成否・モデルID・候補判定・モデル登録・統合・Switching leader等の離散列を比較する。
離散列は配列のshapeと正規化したJSONのSHA-256で固定し、浮動小数点の重み自体は固定しない。
数値比較は絶対誤差`1e-9`、相対誤差なし。実時間は回帰対象に含めない。

## 実行環境

再構築用の版固定・環境詳細・スクリプトは
[Windows CPU goldenの再現環境](../../environments/golden/windows-cpu/README.md)に保存する。
新規venvからの再現確認には、同資料の`recreate.ps1`・`verify.py`を使う。

2026-10-02に確認した基準はWindows / CPU / float32、Python 3.13.15、
NumPy 2.4.6、torch 2.12.1+cpu、pytest 9.1.1である。
既存goldenの生成メタデータにはPython 3.14.6と記録されているが、現在のWindowsでも11ケースが一致する。
以前確認したUbuntuでは既存goldenとの差が出たため、バージョンを揃えるだけで
OSをまたいだ一致が保証されるとは扱わない。

新goldenにはOS・アーキテクチャ・dtype・threads・ソースcommit・旧goldenのSHA-256・
使用したMNISTファイルのSHA-256も記録する。新テストのtorch実行は1スレッドに固定する。
環境が異なる場合は警告し、比較を実行する。自動skipやgoldenの自動更新は行わない。

MNISTの学習用画像・ラベルは`data/mnist/`に必要（git管理外）。
`FDE_MNIST_DATA_DIR`で別の保存先も指定できる。
不足時は既存のデータ読込み処理が取得するため、オフラインでは事前に配置する。

## Windowsでの検証コマンド

PowerShellで、リポジトリのルートから実行する。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR=(Resolve-Path venv/matplotlib-cache).Path
.\venv\Scripts\python.exe -m pytest --ignore=.venv `
  tests/test_option_schema.py tests/test_parameter_schema.py tests/test_metric_schema.py `
  tests/test_shared_backbone.py tests/test_provisional_model.py `
  tests/test_clustering_decision.py tests/test_fedsda_configuration.py `
  tests/test_regression.py tests/test_proposed_regression.py -q
```

`.venv`は復旧したLinux用環境であり、このWindows検証では`venv`を使う。
`--ignore=.venv`でLinux用ディレクトリへのpytest探索を避ける。

意図したアルゴリズム変更を承認した場合に限り、新しい最終構成goldenを更新する。

```powershell
.\venv\Scripts\python.exe tests/test_proposed_regression.py --update
```

更新前後の指標・イベント件数・離散列の差と理由をレビューする。
旧goldenはこのコマンドで変更されない。

## 実行環境ごとの最終構成golden

旧実装の最終構成の結果は、実行環境で変わる。確認した範囲（2026-10-10〜11）:

| 環境 | Windows用のgoldenとの違い |
|---|---|
| WSL Ubuntu（Python 3.14.4、NumPy 2.4.6、torch 2.12.1+cpu） | SINE-2は指標10・離散列15、SEA-2は指標1・離散列2が違う。MNIST-2は一致 |
| 研究室サーバ（Linux、Python 3.10.16、NumPy 2.2.6、torch 2.12.1+cpu） | SINE-2が、WSLとも違う（候補の棄却が2件。WSLは3件）。SEA-2とMNIST-2は、WSLと同じ |

そのため、goldenを、実行環境ごとに持つ（2026-10-08・10-11ユーザー決定）。
Windows用のgolden（`tests/proposed_regression_golden.json`）と`tests/test_proposed_regression.py`は、変更していない。

### 置き場所と選び方

- `tests/refactoring/proposed_regression_goldens/`: 環境ごとのgolden。1環境1ファイルで、名前は、
  その環境の記録から決まる（例: `linux-x86_64-python3.14.4-numpy2.4.6-torch2.12.1+cpu.json`）。
  形・条件の定義・使用したMNISTファイルは、Windows用と同じ。
- testは、実行環境の記録（OS、機種、Python・NumPy・torchの版、device、dtype、スレッド数）と、
  goldenの`_env`が**完全に一致する**ものを、自動で選ぶ。Windows用のgoldenも、同じ規則で選ぶ対象に含まれる。
  設定や引数は要らない。
- 一致するgoldenがない環境では、goldenとの照合だけをskipする（失敗にしない。skipの理由に、
  環境の記録と、足し方が出る。`-rs`で表示できる）。新実装と旧実装を、同じprocessの中で比べる照合は、
  goldenに依らないので、どの環境でも実行される。
- 版が1つ変わると、goldenは選ばれなくなり、照合がskipになる（失敗にならないので、気づきにくい）。
  これは、Windowsでも同じである: 新実装の照合は、Windowsの基準環境（Windows用のgoldenの`_env`）と
  版が違うWindowsでは、skipされる（既存の回帰test`tests/test_proposed_regression.py`は、これまでどおり、
  版が違っても、警告して比較する）。開発の環境（Windowsの基準環境とWSL）では、検証のときに、`-rs`で、
  skipの件数と理由を確かめる。skipの理由には、いまあるgoldenの環境の記録も出るので、どの版が違うかが分かる。
  Windowsの基準環境での全pytestのskipは、4件（POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件）。
  WSLでは、goldenに関わるskipは、ない。

照合するtest:

- `tests/refactoring/test_environment_proposed_regression.py`: 旧実装の3ケースを、実行環境のgoldenと照合する
  （基準は、既存の回帰testと同じ。イベント件数と離散列は完全に一致、指標は絶対誤差`1e-9`）。
  Windowsの基準環境では、既存の回帰testが照合するので、skipする。
- `tests/refactoring/test_fedsda_run_metric_derivation.py`: 新実装の3ケースを、実行環境のgoldenと照合する。

```bash
source .venv/bin/activate   # リポジトリ直下のLinux用環境
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests/refactoring/test_environment_proposed_regression.py \
  tests/refactoring/test_fedsda_run_metric_derivation.py -q -rs
```

### goldenを足す・消す・作り直す

足す（goldenのない環境で、1回だけ）:

```bash
python tests/refactoring/test_environment_proposed_regression.py --update   # ファイルを1つ書く
python -m pytest tests/refactoring/test_environment_proposed_regression.py -q   # 別のprocessで、再現を確かめる
git add tests/refactoring/proposed_regression_goldens/
```

できたファイルをcommitすれば、その環境で、goldenとの照合が行われるようになる。
goldenに記録する、旧の代表11ケースのgoldenのSHA-256（`legacy_golden_sha256`）は、checkoutの改行（LinuxはLF、
WindowsはCRLF）で値が違う。検査は、どちらの形のSHA-256も受け入れる。
消すときは、ファイルを消す（ほかに変える所はない。環境ごとのgoldenが1つもなくても、testは成功する）。
ディレクトリには、goldenのファイルだけを置く（名前を変えたり、ほかのファイルを置いたりすると、検査が失敗する）。

ふだんのspec（挙動を変えない移植・整理）では、goldenを作り直さない。照合が失敗したら、実装の差を調べる。
作り直すのは、意図したアルゴリズムの変更を承認して、結果が変わるときだけで、そのときは、
Windows用（`python tests/test_proposed_regression.py --update`）と、環境ごとのgolden
（各環境で`--update --overwrite`）を、どちらも作り直す。すでにgoldenがある環境では、`--overwrite`を
付けないと、作成は拒否される。Windowsの基準環境では、常に拒否される（上のWindows用のコマンドを使う）。

WSLからWindows側のworktreeを使うときは、WSLのgitがworktreeを読めないので、Windows側で求めた
commitを`--source-commit <40桁のhash>`で渡す。

### WSLで失敗するtest

WSLでは、次の3件が失敗する。どれも環境差であり、goldenやtestを変える理由にしない。

- `tests/test_proposed_regression.py`と`tests/test_regression.py`: Windows用のgoldenと比べる。
- `tests/refactoring/test_single_run_dependency_boundaries.py`の1ケース
  （`from __future__ import *`）: Python 3.14の構文解析の違い。

全体runの対照（`tests/refactoring/test_fedsda_stream_protocol_run.py`）の、sine2以外のdatasetの条件は、
新旧の全状態の一致を、どの環境でも確かめる。条件ごとの「通る経路」（複数モデルでの終了ほか）の期待は、
条件を選んだWindowsでだけ確かめる（Linuxでは、浮動小数点の差で、警報やモデルの登録の起き方が変わる）。

### mainのcheckoutしかない計算機（研究室サーバ）

リファクタリングの作業は、ブランチ`refactor/architecture`にある（`main`には、新実装`src/`と、上のtestがない）。
`main`を切り替えずに、別のディレクトリへworktreeを作って実行する。実行中の実験と、`main`の作業ツリーには、
影響しない。

```bash
cd <リポジトリのルート>                      # mainのcheckout
git fetch origin
git worktree add ../FedSDA-refactoring origin/refactor/architecture   # 初回だけ（detached HEAD）
source .venv/bin/activate                    # mainのcheckoutの環境を使う
python -c "import pytest; print(pytest.__version__)"   # なければ: pip install pytest==9.1.1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FDE_MNIST_DATA_DIR="$PWD/data/mnist"  # mainのcheckoutのMNIST（なければ、旧の読込みが、ここへ取得する）
cd ../FedSDA-refactoring
python -m pytest tests/refactoring/test_environment_proposed_regression.py \
  tests/refactoring/test_fedsda_run_metric_derivation.py -q -rs
```

2回め以降は、`cd ../FedSDA-refactoring && git fetch origin && git checkout --detach origin/refactor/architecture`で更新する。
不要になったら、`main`のcheckoutで`git worktree remove ../FedSDA-refactoring`。

研究室サーバのgoldenを足すとき（1回だけ）は、上の環境（`.venv`と環境変数）のまま、worktreeで次を実行する。
ブランチへ直接commitして、pushする。`git checkout -B`は、worktreeのローカルの`refactor/architecture`を、
取得した最新の`origin/refactor/architecture`の位置へ作り直す（このworktreeに、pushしていないローカルのcommitが
ある場合は、ブランチから外れる。上の手順で作ったworktreeには、ない）。

```bash
cd ../FedSDA-refactoring          # すでにworktreeにいるなら不要
git fetch origin
git log --oneline origin/refactor/architecture..refactor/architecture 2>/dev/null   # 何も出なければ、ローカルだけのcommitはない
git checkout -B refactor/architecture origin/refactor/architecture
python tests/refactoring/test_environment_proposed_regression.py --update
python -m pytest tests/refactoring/test_environment_proposed_regression.py \
  tests/refactoring/test_fedsda_run_metric_derivation.py -q -rs      # skipなしで成功することを確かめる
git add tests/refactoring/proposed_regression_goldens/
git commit -m "test: 研究室サーバの環境の最終構成goldenを追加"
git push origin refactor/architecture
```

研究室サーバでの照合は、ふだんのspecのたびに行う必要はない（開発中の照合は、Windowsの基準環境とWSLで行う）。
worktreeを更新（`git fetch origin`の後、`git checkout --detach origin/refactor/architecture`。上でブランチを
作った後なら、`git pull`）した後に、確かめたいときだけ実行すればよい。

## リファクタリングの進め方

同じGit履歴のブランチまたはworktreeから開始し、現状のcommitとgoldenを保存する。
責務の分割・配置変更を小さな単位で行い、その都度、上記の検証を通す。
最初はアルゴリズム・固定設定・乱数の消費順序を維持する。

goldenが通っても、全seed・全データセット・全オプションの同値性を証明したことにはならない。
変更した経路がこの3ケースに含まれない場合は、対応する機能テストや同一条件の比較を追加する。
論文用の全実験を最初から再実行する必要はない。

コミット保留の3資料と既存の`results/`は、リファクタリングのために削除・上書きしない。


## 新実装の重要動作golden（基準版1〜3）

旧基準とは別に、`tests/refactoring/behavior_regression_goldens/v1/`へ新APIの基準を保存する。
FedSDAの11構成・FedDrift・FedAvgの小さいrunと、重要イベント/境界の追加4ケースを扱う。
完全設定・全指標・全記録列のshape/dtype/digest・最終グローバルパラメータを固定する。
候補の採用/棄却/再利用、統合、予測重み再較正が実際に起きたかもケースごとに検査する。
これは性能評価でも全条件の網羅でもなく、検証済み移植の現在の挙動を後続変更から守る基準である。
13構成の小runは、構成の選択・設定・基本学習の回帰を守る。ADWINなど各方式固有の警報・候補検証・
統合経路をすべて通すものではない。追加の重要FedSDA runは最終提案構成の経路を扱い、
他の構成固有の動作は既存の実旧対照・部品testと組み合わせて検証する。

版は次の役割で区別する。全履歴はGitに残す。

| 項目 | 意味 | 変更する場合 |
| --- | --- | --- |
| `baseline_version` | 期待する動作の基準。現在は1 | LEGACY/IMPROVEなどで期待値を意図的に変えるとき |
| `schema_version` | このgolden JSONの形式。現在は1 | snapshotの記録形式を変えるとき |
| `environment` | OS・機種・Python/NumPy/torch・CPU/dtype/thread | 同じ動作基準を別環境で検証するとき |
| `source_commit` | 生成した生産コードの来歴 | 新しい基準や環境別goldenを生成するとき |

基準版は、次のとおり。照合のtestは、全部の版を、毎回、確かめる。

| 版 | 動作 | 設定 | ケース |
| --- | --- | --- | --- |
| 1 | 2026-10-13の移植完了時点。FedSDAの登録の扱いは、固定旧と同じ（送信保留を1つだけ保持し、確認で現行モデルを付け替える） | 最終構成の既定から、登録の扱いだけを、`keep_latest_pending_model_and_reassign_current_model_id`へ替えて実行する。保存した設定は、この項目を持たない（版1の後に足された）ので、値を確かめてから、除いて比べる | 17 |
| 2 | 採用したモデルを、それ自身のIDで登録する（LEGACY-016の修正。[spec](../../.kiro/specs/pending-model-registration-by-own-id/design.md)） | 最終構成の既定から、登録の扱いを`register_each_pending_model_by_its_own_id`に明示する（版3と同じ理由） | 版1の17（同じ設定）＋`fedsda-sine2-reference-reuse` |
| 3 | 短い警報のあと、現行モデルへ取り込んだ前区間を、保留から外す（LEGACY-002の修正。[spec](../../.kiro/specs/too-short-alarm-earlier-sample-release/design.md)） | 最終構成の既定から、登録の扱いを`register_each_pending_model_by_its_own_id`に明示する（版4で既定が変わったため。保存した設定はこの値を持つので、除かずに比べる） | 版2と同じ18 |
| 4 | 送信保留のモデルを、現行モデルとして待った後に登録する（LEGACY-016の扱いの見直し。[spec](../../.kiro/specs/pending-model-registration-after-waiting-as-current-model/design.md)。2026-10-14に一度既定にしたが、比較で精度がわずかに下がったので、既定は版3の扱いへ戻した） | 最終構成の既定から、登録の扱いを`register_pending_model_after_waiting_as_current_model`に明示する | 版3の18＋`fedsda-sine2-alternative-reference-reuse` |

版1・2は、短い警報のあとの保留の扱いも、固定旧と同じ値（`keep_all_pending_samples`）へ替えて実行する（保存した設定は、この項目も持たない）。版3と版2の差は、版2の扱いで、前区間のある短い警報が起きるケース（`fedsda-sine2-events`と`fedsda-sine2-reference-reuse`。同じ標本の2回目の取込みが、Windowsの基準環境で、487件と467件）に限る。ほかの16ケースは、版2と同じであることを、testが確かめる。版3の、FedSDAのケースの結果は、「同じ標本の2回目の取込みの数」（0であること）も持つ。

版4と版3の差は、版3の扱いで、待ちの間に現行でない送信保留がある（後に同期が続く待ちの進行の時点か、登録の確認の時点）ケース（`fedsda-sine2-events`の4件と`fedsda-sine2-reference-reuse`の1件。Windowsの基準環境）に限る。ほかの16ケースは、版3と同じであることを、testが確かめる。`fedsda-sine2-reference-reuse`（seed 4）は、版4では、別の保有モデルの再利用の判定を通らなくなるので、その経路は、版4で足した`fedsda-sine2-alternative-reference-reuse`（同じ設定で、seedが5）が守る（版4だけのケース。版3のgoldenは変えない）。版2・3は、登録の扱いを`register_each_pending_model_by_its_own_id`に明示して組み立てる（helperの「版の後に、既定の値が変わった項目」の表。既定は、2026-10-14に一度変わり、同日に戻った。現在の既定は、版3の扱い）。版4は、現行モデルとして待った後に登録する扱いを、同じ表で明示する。

版2と版1の差は、版1の扱いで、登録されないままの採用が起きるケース（`fedsda-sine2-events`の2件）に限る。ほかの16ケースは、指標・表・パラメータが、版1と同じであることを、testが確かめる。`fedsda-sine2-events`は、版2では、別の保有モデルの再利用の判定を通らなくなる（登録されずにclientへ残っていたモデルが、登録・統合されるため）ので、その経路は、版2で足した`fedsda-sine2-reference-reuse`（同じ設定で、seedが4）が守る。版2の結果は、「登録されないままの採用の数」（0であること）も持つ。

2026-10-13に、分類器の属性の名前を改めた（`residual_adapter`→`model_owned_feature_transform`。IMPROVE-014。[spec](../../.kiro/specs/classifier-structure-independent-naming/design.md)）。属性の名前は、パラメータ名の一部なので、版1〜3の、WindowsとLinuxの6ファイルの、最終グローバルパラメータのキーの名前だけを、機械的に書き換えた（版1は76、版2は100、版3は96のキー）。値（digest・shape・dtype）と、生成の記録（`source_commit`など）は、変えていない（実行し直していない）。書き換えの前後で、キーの名前を戻すと、内容が一致することを確かめた。動作の基準は変わらないので、版は増やしていない。

現在の版（新しい基準を生成するときの既定）は、helperの定数`BASELINE_VERSION`。別の版を生成・照合するときは、`--baseline-version`で指定する。

通常のcommitごとには基準版を増やさない。意図した動作変更は対象specで旧基準との差分理由を説明し、
レビューと検証後に新しい版directoryを追加する。現在の版はhelperで明示し、最新directoryを自動採用しない。
環境差だけで既存goldenを上書きしたり、許容誤差を増やしたりしない。未知環境は数値testだけを理由つきでskipし、
構造/定義検査は必須にする。CPU型番・数値ビルドの差まで完全一致を保証しないためホストCIは数値診断、固定ローカルは全回帰必須。

照合はworktreeルートで、固定環境・OMP/MKL各1thread・MNIST配置を用意して行う。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
../../venv/Scripts/python.exe -m pytest tests/refactoring/test_important_behavior_regression_goldens.py -q -p no:cacheprovider
```

生成は通常のpytestから呼ばれない。未使用のGit管理外出力を指定し、候補を確認してから対象版へ追加する。
検証用tempの置き場所とMNISTは、この文書と共通引継ぎ手順に従って準備する。

```powershell
$goldenSourceCommit=git rev-parse HEAD
../../venv/Scripts/python.exe tests/refactoring/behavior_golden_cases.py --output ../../venv/refactoring-tests/behavior-candidate-windows.json --source-commit $goldenSourceCommit
```

Linuxでは固定`.venv`をactivateして同じhelperを実行する。生成元commitはHEADとの一致と、srcの変更/未追跡ファイルなしを検査する。
Windows作成worktreeをWSLで使う場合は、対応するWindowsパスに対し`git.exe -C`で読取り検査する（Git設定は変えない）。
独立processで2回すべてのケースを実行し、全snapshot一致と必須経路を確認した後にだけ新規出力へ保存する。
既存先は上書きせず拒否する。MNISTの画像/ラベル2ファイルのSHA256も保存する。

数値は対応環境内で完全一致させる。未定義指標は公開serializerの定義に従いnull、
記録列のNaN/+Inf/-Infは区別したtagでcanonical JSONのdigestへ含める。
差分はケースと指標/設定のfield path、記録列名、モデルIDとparameter名までを示す。
digestの異なる元要素の位置は保存しないため、原因調査では同条件runの公開記録表を読む。
