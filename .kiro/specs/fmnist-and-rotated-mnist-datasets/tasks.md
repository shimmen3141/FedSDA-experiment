# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4。実装はGPT-6.1-sol（task 1〜4）。確認・全pytest・レビューの起動は主担当。

- [x] 1. 読み込みと生成器
  - 要求4.2の(i)〜(iv)のtestを先に書き、REDを確かめてから実装する。設計書の「回転の対応の確認」に、公式のcommit `18e7fbe`の該当の行を書き足す。
  - 完了条件: 新しいtestが通り、既存の`mnist2`・`mnist4`のtestと版1〜14のgoldenが値を変えずに通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

- [x] 2. 全手法の設定と掃引
  - 要求4.2の(v)(vi)(vii)のtestを先に書き、REDを確かめてから実装する。
  - 完了条件: 新しいtestが通り、既存の設定のtestと版1〜14のgoldenが通る。
  - _Requirements: 3.1, 3.2, 3.3, 4.1_

- [x] 3. 文書
  - 完了条件: データの説明に4データの定義・根拠の実装（公式FedDriftのファイルとcommit、IFCA・FedCCFAの論文）・開示する差があり、Fashion-MNISTの取得の手順（URLとMD5の照合）があり、
    `docs/reference/options.md`が直接編集されていない（必要なら再生成されている）。
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 4. 確認（主担当）
  - 全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回。

## Implementation Notes（2026-10-27、実装担当）

- 対象はtask 1〜3。開始時は`refactor/architecture`、HEAD `1bace5df`、追跡ファイルの差分なし。
  仕様4文書・spec.json・steering・命名規約を読み、`kiro-impl --review off`で実装した。task 4の全pytestと独立レビューは主担当が行う。
  commit・stage・pushは行っていない。
- 公式の根拠は指定の手元のFedDriftに対する`git show`で再確認した。
  `18e7fbe`の78行はk=0、79〜83行は180°の説明とk=2、181〜183行はreshape→rot90→flatten。
  `62ad4e2`の76〜77行は行列のkをそのまま渡し、168〜170行は同じ回転。ラベルも変更しない。設計の根拠との相違は無く、design.mdへ確認を追記した。
- 要求1・2: IDXの読込みを共通の非公開関数へ分け、Fashion-MNISTの環境変数・既定の置き場所・診断名を追加した。
  cacheは解決したディレクトリごとの既存の表を使う。既存MNISTの例外の形と検査を維持する。
  Fashion-MNISTは既存のラベル交換・画像選択を継承し、回転は同じ生成器で選んだ概念0の標本の画素だけを並べ替える。
  2概念はk=0/2、4概念はk=0/1/2/3。選択の乱数消費は同じ1回で、既存`MnistSampleGenerator`の処理は変更していない。
- 実装担当の判断: 生成器は`FashionMnistSampleGenerator`・`RotatedMnistSampleGenerator`へ型を分けた。
  既存のdatasetと生成器の対応検査がexact型＋概念数を使うため、同じ概念数でも元画像と変換を区別できる。
  `MnistTrainingData`はMNIST形式の配列を保持する同じ役割で再利用し、`digit_labels`はFashion-MNISTでも元のクラスラベルを表す。
  回転は公式のNumPy操作で作った添字表を使い回し、特徴のfloat objectの共有を保つ。モジュール・公開名・役割をnaming.mdへ追加した。
  MD5の読込み時の検査は既存MNISTに無いので足さず、取得時の照合を文書で必須にした。
- 要求3: 4データの定義と実行条件、3手法のモデル設定の表、掃引の重さの表を追加した。
  FedAvgの名前の前方一致は6つの画像データ名の明示の集合に変更した。
  3手法の完全設定はdataset名以外をMNISTと一致させ、FedDriftのδは0.10。
  掃引CLIのファイルの事前検査も明示の対応へ替え、回転のMNIST・Fashion-MNISTとも不足時は結果の作成前に拒否する。
- 要求4: 新規のtestは生成・読込み9件、設定・全体run・掃引28件の計37件。
  task 1は9 failedのREDを確認してから実装し、既存MNIST87件と合わせて96 passed。
  task 2は26 failed / 2 passedのREDを確認してから実装し、28 passed（全体run12件を含む、175.39秒）。
  全体runは新4データ×3手法、10 client・各100標本・集約間隔50で、既定の事前学習から指標の導出まで行った。
  Fashion-MNISTが無い環境では理由つきでskipする。拒否の理由と型の確認を補強した後の非slow分も16 passed。
  データの定義・設定metadataの6データ固定の期待値と依存許可表を更新し、設定検証1325 passed、依存境界3127 passed。
- 要求5: データ特性と掃引の文書に4データの定義、公式FedDriftのファイル・commit・行、IFCA・FedCCFAの論文への直接の参照、開示する差を追加した。
  Fashion-MNISTの公式URL・2ファイルのMD5と取得後の照合手順も記載した。
  task 3の文書検証scriptは追記前に必要な記述の欠落でRED、追記後にPASS。文書用の恒久testは増やしていない。
  `docs/reference/options.md`は旧実装の宣言からの生成物で今回の変更を含まないため、変更・再生成していない。
- 検証環境: `../../venv/Scripts/python.exe`、`PYTHONPATH=src;.`、OMP/MKL各1。
  このWindows sandboxではpytestのmode=0700の一時ディレクトリに書けないため、Git管理外の`.tmp/pytest_workspace_temp.py`で
  `tmp_path`・`tmp_path_factory`だけを通常の`Path.mkdir`でworktree内に作るfixtureへ差し替えた
  （実行時は`PYTHONPATH=src;.;.tmp`、`-p no:cacheprovider -p pytest_workspace_temp`）。権限設定・sandboxは変更していない。
  標準ライブラリの`TemporaryDirectory`とmultiprocessingには介入していない。
- 初回の対象検証は4884 passed / 9 failed / 3 errors（327.59秒、`.tmp/fmnist-target.xml`）。
  2失敗は上記の期待値・依存許可表で修正済み。3 errorsは一時fixtureがtest名の`<lambda>`をWindowsのディレクトリ名へ使っていたためで、固定名＋UUIDへ修正した。
  残り7失敗はmultiprocessingの名前付きパイプの権限拒否6件と、`tests/test_proposed_regression.py`の`TemporaryDirectory`の権限拒否1件。
  同じ環境制約での再試行はせず、task 4で通常環境の確認を要する。
- 品質検査: `ruff check src tests/refactoring`、変更・追加Python18ファイルの`ruff format --check`、`spec_checks.py names ... --base 1bace5df`が成功。
  Pyrightはpyproject.tomlのstrict範囲を維持し、固定venvのsite-packagesとrepo/testの探索先を明示した一時設定で実施した。
  変更source11ファイルと新規test2ファイルは0 errors。変更した既存testを含めた18ファイルでは28 errorsがあるが、
  基点`1bace5df`の該当4ファイルでも同じ28件であり、診断の内容・件数が一致することを照合した。依頼範囲外の既存型エラーは直していない。
- 最終の対象検証: 初回と同じ14ファイルを対象に、4877 passed / 19 deselected（145.18秒、exit 0、`.tmp/fmnist-target-final.xml`）。
  19件の内訳は、別実行で成功した全体run12件（再実行不要）と、上記の環境制約7件。
  既存MNIST・合成データ・定義・設定・掃引・依存境界・`tests/test_regression.py`を含む。新規37件はすべて成功し、skipは無い。
- 版1〜14のgolden検証: `tests/refactoring/test_important_behavior_regression_goldens.py`を全件実行し、
  542 passed / 3 failed（887.76秒、exit 1、`.tmp/fmnist-goldens.xml`）。Windowsの**全282 caseの数値照合は成功**し、skipは無い。
  3失敗は`test_generation_requires_two_matching_process_results[same/different/failure]`だけであり、
  golden生成器の`tempfile.TemporaryDirectory`内のファイルの書込み・後片付けがsandboxに拒否されたもの。
  Windows/WSL両環境の保存済みgoldenの構造・設定の照合は成功したが、WSLでの数値実行は行っていない。
- task 1〜3は実装・各taskの受入れ条件に対応する検証を完了した。機能全体の完了承認は行わず、phaseはimplementation、task 4は未完了のまま。
  通常環境で必要な残りの確認は、上記の環境制約10件、既存testの型エラー、全pytest、WSLの数値実行、独立実装レビュー。
- 未実施: 全pytest、WSL上でのgoldenの数値実行、独立の実装レビュー、本規模の研究実験、commit。
  `BEHAVIOR_BASELINE_VERSION`、既存golden、旧実装、保留3文書、調査の記録、画像データは変更していない。
  実装上の仕様からの逸脱は無い。環境制約による検証の不足と既存testの型エラーは、主担当のtask 4へ引き継ぐ。

### 変更・追加ファイル

追加:

- `src/federated_learning_experiments/data/mnist/fashion_mnist_sample_generation.py`
- `src/federated_learning_experiments/data/mnist/rotated_mnist_sample_generation.py`
- `tests/refactoring/test_fmnist_and_rotated_mnist_sample_generation.py`
- `tests/refactoring/test_fmnist_and_rotated_mnist_runs.py`

変更:

- `src/federated_learning_experiments/configuration/experiment_run_conditions.py`
- `src/federated_learning_experiments/data/dataset_definitions.py`
- `src/federated_learning_experiments/data/mnist/mnist_training_data.py`
- `src/federated_learning_experiments/data/observed_sample_generation.py`
- `src/federated_learning_experiments/runtime/fedavg_run_settings.py`
- `src/federated_learning_experiments/runtime/feddrift_default_run_settings.py`
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_command.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_execution.py`
- `tests/refactoring/test_dataset_definitions.py`
- `tests/refactoring/test_fedsda_final_configuration_run_settings.py`
- `tests/refactoring/test_run_settings_validation.py`
- `tests/refactoring/test_single_run_dependency_boundaries.py`
- `tests/refactoring/test_synthetic_sample_generation.py`
- `docs/experiments/dataset-characteristics.md`
- `docs/experiments/experiment-sweep.md`
- `.kiro/specs/fmnist-and-rotated-mnist-datasets/design.md`
- `.kiro/specs/fmnist-and-rotated-mnist-datasets/naming.md`
- `.kiro/specs/fmnist-and-rotated-mnist-datasets/spec.json`
- `.kiro/specs/fmnist-and-rotated-mnist-datasets/tasks.md`

Git管理外の`.tmp/`にはJUnit、一時検証設定・fixture・基点のtestのコピーを残した。
用途と再実行方法は`.tmp/README.md`。これらは成果物の来歴のために残し、stageしない。
