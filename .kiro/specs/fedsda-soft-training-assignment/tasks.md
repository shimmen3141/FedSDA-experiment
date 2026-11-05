# Implementation Plan

- [x] 1. 構成と学習の経路とtestを足す
  - 要求3.1〜3.3のtestを先に書き、REDを確かめてから、構成`fedsda-soft-training`と`fedsda-training-mass-diagnostics`、学習の量のコンセプト別の記録の設定の項目（既定は記録しない）、設定の項目（`fedsda`のclientの設定の学習の割り当ての方式。既定`single_model_assignment`は保存から省き、新しい値`responsibility_weighted_training`。`experiment_spec/options.py`のOptionSpec・ActivationRule。構成の定数は`sweep/experiment_configurations.py`の既存の変種と同じ形。design.mdの「設定の登録」）、共有の学習ストアと責任度、重み付きの共同更新、集約の重み、統合での責任度の列の扱い、柔らかい学習の量のコンセプト別の記録、純度の導出を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界・スキーマのtest、`fedsda`の変種のtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`、`tests/test_proposed_regression.py`が通る。Ruff・Pyright。新しい全体runのtestの所要時間を記録し、重いものは`slow`と`tests/conftest.py`の配り方へ登録する。
  - _Requirements: 1.1〜1.6, 2.1, 3.1〜3.4_
- [x] 2. 文書と検証の記録
  - 構成の一覧、設定参照の再生成（`python -m tools.generate_option_docs`）、naming.mdの追記。関係するtestの結果をImplementation Notesに記録する。
  - 主担当: 独立CLIの実装レビュー（GPT-6 Luna）を起動し、判定・指摘・採否をImplementation Notesとspec.jsonに記録する。全pytest（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）を流して結果を記録する。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 本規模の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、本規模の実行（2構成×200 run＝400 run）のコマンドを渡す。design.mdの集計をユーザーに示し、実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2

## Implementation Notes

### 実装前のストア経路調査（2026-11-05）

| 操作 | 既存の経路 | 共有ストア・責任度・純度への対応 |
|---|---|---|
| FIFOの容量超過分を確定 | released_pending_sample_assignment → assigned_training_sample_absorption | 追加時に局所モデルを再評価して責任度を固定。診断だけならone-hot。既知の概念だけ量を加算 |
| 警報区間より前を確定 | alarm_training_interval_preparation → assigned_training_sample_absorption | 同上 |
| 警報区間で保有モデルを再利用 | alarm_change_interval_resolution → assigned_training_sample_absorption | 同上。分母0時の行き先は選択したモデル |
| 候補採用で区間を確定 | alarm_change_interval_resolution → adopted_candidate_local_adoption | 候補登録後の集合で責任度を計算。既存標本の新列は0 |
| 一時IDを登録IDへ置換 | held_model_registration_confirmation → reassign_model_training_samples_id | 責任度の列と純度の行を同時に改名 |
| 配布で統合 | global_model_distribution_application → remap_model_training_sample_collections | 対応表で責任度の列・純度の行を加算。再正規化しない |
| 全標本を破棄 | clear_model_training_samples | FedSDAの実行経路には呼出しなし。全補助状態も空にする |
| 単一モデルの標本を破棄 | FedSDAには該当する操作なし | 固定入力の単体testで列除去・正の列が無い標本の除去を検査 |

`ModelTrainingSampleStore`はPythonのlistへextendするだけで、上限・古い標本の追い出しは無い。評価ストアとFIFOの上限は学習ストアとは別。共有の学習ストアも上限を設けない。既存の固いストアは監視・統合の判断のために維持し、柔らかい学習と集約は責任度のストアを参照する。


### Task 1・2の実装と検証（2026-11-05、独立レビュー前）

- Task 1の実装とTask 2の文書を実施。依頼どおり、レビューCLI・内蔵子エージェント・commit・push・stash・本規模実験・全pytestは実行していない。独立レビューと全pytestは主担当の担当。必須検証の一部に権限による未実施があるため、Task 1・2のcheckboxは未完了のままにした。
- TDD: 新しい式・ストアのtestは未実装moduleによるcollection error（1.25秒）を確認してから実装し、7 passed（1.32秒）。次に新構成と固定hashのtestを追加し、未登録構成による2 failed（1.26秒）を確認してから登録・接続した。全体runのspy追加時に、spyの差込み先・既存設定helperの引数の誤りを修正。大きい条件への拡張時に比較する2構成の標本数の指定が一時的に違ったため2件失敗し、同じ条件へ修正して再確認した。途中の失敗を成功へ数えていない。
- 設定は`FedsdaRunClientSettings.training_assignment_strategy`（既定`single_model_assignment`、追加`responsibility_weighted_training`）と`record_training_mass_by_concept`（既定False）。いずれも既定を保存から省く。柔らかい学習はFixed-Shareと共同更新に限定。既存35構成の順序・保存設定SHA-256・run識別子を、変更前のHEADで作った固定fixtureと照合した。
- ストアの操作と対応は上表。既存の固いストアは監視・統合の判断と既存の診断のために残す（Tensor payloadは参照を共有する）。柔らかい学習の抽出とサーバの集約には、固いストアの件数を使わない。解放時の局所モデルの順伝播は`no_grad`で行い、既存のモデルの計測の経路で推論として計数される。新しい実行時importは標準ライブラリと既存依存のtorchだけ。
- モデルの追加時の既存標本の新列は、辞書に存在しない値を0とする表現。登録時は一時IDを改名し、統合時は対応表で全列を加算。純度の量も同時に移す。FedSDAの実行経路に単一モデルの標本破棄は無い。列の破棄・非再正規化・正の列を失った標本の除去は固定入力のtestで確認した。
- `n_m`は小数なので、サーバの集約と、統合時の重みの型・検査を有限のint/floatへ広げた。既存の固い構成では引き続きintを渡す。小数を不正としていた既存testの事例を非有限値へ変更し、小数の加重平均のtestを追加した。
- 保存形式の版は変更不要と判断した。新しい表は設定で有効にした構成だけに加わり、既存構成の列・dtype・保存設定は不変。既存のreaderは表の名前ごとの配列をそのまま読む。FedSDA artifactの版2、FedDriftの版10、sweepの版10は維持。`test_fedsda_default_waited_registration_and_alarm_wait.py`・`test_experiment_sweep.py`・`test_feddrift_experiment_run_artifacts.py`のCLI／構成／版の固定期待値を読み、変更不要を確認。構成一覧と除外集合だけは`test_fedsda_configuration_variants.py`で末尾追加に追従した。
- `experiment_spec/options.py`へOptionSpec・ActivationRuleを登録し、`python -m tools.generate_option_docs`で参照を再生成。構成一覧とnaming.mdも追記した。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と、具体的なimportの依存境界にも登録。新しい全体runの4ケースはslowを付け、`tests/conftest.py`でdatasetごとに配る。

#### 検証の条件と結果

固定venv `../../venv/Scripts/python.exe`、`PYTHONPATH=src;.`、`OMP_NUM_THREADS=1`・`MKL_NUM_THREADS=1`・`OPENBLAS_NUM_THREADS=1`、`FDE_MNIST_DATA_DIR=C:\Users\yshin\Research\FedSDA-experiment\data\mnist`で実行した。

- 新規testは32件成功（全件の実行は86.02秒）。その後、診断だけの比較も柔らかい学習と同じ規模へ拡張して2件を再確認し、2 passed・64.16秒（SINE2 22.47秒、MNIST4 40.57秒）。最後の設定の組合せ検査を含む非slowの新規28件・依存境界・スキーマ・client設定は、2,444 passed・4 deselected・21.90秒。deselectedの4件は前記の全体runの4ケースで、別途成功済み。
- Windows用の基準版1〜16の数値golden: 206 passed（case合計407.95秒）。版1〜6は各5件、版7〜10は各17件、版11〜16は各18件。`tests/test_regression.py`: 1 passed・76.35秒。goldenのファイルの更新は無し。
- 関係testの初回: 754 passed・1 failed・252 skipped、218.47秒。不正入力から有限小数を外す既存testの期待値更新を、実行中のprocessはまだ読んでいなかった。変更後の当該testを別processで再確認して1 passed・5.13秒。現在の結果は755件成功。下表の該当行は再確認後の件数を示す。
- 異なるtest IDを重複除外して **3,288件成功**（新規32＋依存境界／スキーマ2,294＋関係755＋golden／旧回帰207）。再実行したtestを成功件数に重複計上していない。
- 252 skipの内訳: 一時ディレクトリを必要とする239件は、このsandboxでpytestの作った一時ディレクトリの`os.scandir`がWinError 5になったため、関係testの実行用pluginで明示的にskipした。残る13件は元からの「大きい全体runで照合する」coverage testのskip。前者239件は未検証。
- `tests/test_proposed_regression.py`: 1件を実行したが、TemporaryDirectory内の`sine2.npz`の保存とcleanupがPermissionErrorとなり失敗（12.54秒）。**成功扱いにしていない**。この1件を含む、権限で未検証のtestは計240件。xdistは一時ディレクトリの準備中にINTERNALERRORとなり0件実行。権限の変更やsandbox解除は行っていない。
- Ruff check成功、Ruff format --check成功（405ファイル）、Pyrightは0 errors / 0 warnings。通常の`--pythonpath`の実行ではこのsandboxからvenvのsite-packagesを見つけられず、numpy/torchの未解決importを報告した。プロジェクト設定をextendsし、venvPath/venvだけを明示した`.validation-soft-training/pyrightconfig.json`で同じ設定範囲を検査した。リポジトリのpyproject.tomlは変更していない。
- 命名の検査は`spec_checks.py names ... --base 6d3ca16a`で未登録0、progressの検査も成功。独立レビューを代替するものではない。
- 実行のログ・JUnit・権限制約を分ける実行用pluginは、git管理外の`.validation-soft-training/`へ置いた。`goldens.xml`、`related.xml`、`fractional-validation.xml`、`new-final.xml`、`diagnostics-confirmed.xml`、`final-focused.xml`が証拠。主担当の環境では、skip用pluginを使わず通常のpytestで未検証分を確認する。

#### 小さい全体runの実測

どちらもseed 0、2 client、集約間隔50、batch 32。SINE2は1,500標本/client、MNIST4は800標本/clientとした。既存のイベントを扱う条件を参考に規模を決め、警報・候補採用・複数モデルでの更新が働くことを確認した。規模の最小性を網羅探索したわけではない。小さい条件であり、本規模での重なりや精度差の推定には使わない。

| 柔らかい学習の検査・指標 | SINE2 | MNIST4 |
|---|---:|---:|
| 全体runのtestの秒数 | 12.21 | 24.96 |
| 警報の回数 | 8 | 2 |
| 共同更新の回数 | 2,870 | 1,476 |
| 学習へ参加したモデル数の平均 / 最大 | 1.705923 / 3 | 1.794715 / 2 |
| 追加時の責任度の実効モデル数の平均 / 最大 | 1.114588 / 2.741944 | 1.535799 / 1.999968 |
| batchの重なりの平均 / 最大 | 0.081636 / 0.90625 | 0.090895 / 0.6875 |
| client 0の純度 / 1モデルの基準 | 1 / 1 | 1 / 1 |
| client 1の純度 / 1モデルの基準 | 0.9988396472 / 0.5020408163 | 1 / 1 |
| 集約の検査の回数 | 30 | 16 |
| IDの列の対応付けの検査の回数 | 61 | 33 |
| 実際の列の統合の回数 | 0 | 0 |

参加の条件を満たす全モデルのbatch、概念固有部の勾配とheadの更新、集約へ実際に渡す概念固有部の重み`n_m`・共有部の重み`sum(n_m)`、共有ストアの標本数と固いストアの合計、責任度の総量、ID対応前後の純度の量をspyで検査した。予測・重み・検出・警報／選択の結果が最初の学習まで一致することは、別の100標本/clientのtest（SINE2 2.86秒、MNIST4 13.84秒）で確認。統合と標本の破棄は実runで起きなかったため単体testで確認。診断構成の比較は全ての既存の表（列・値・dtypeを含む）とrun_metricsを完全一致で照合し、純度の量が固いストアのモデル別の件数と一致することも確認した。

one-hotの同じbatchを差し込んだ更新のtestは、SGD・Adam standard・AMSGradで損失・パラメータ・grad・optimizerの状態を比較する。抽出のtestは、one-hotでは割当先の標本だけから抜くことと、一般の責任度で頻度と損失の平均が目的に近づくことを、固定Randomで確認する（復元抽出なので重複を禁止しない）。分母0・非常に小さい分母・不正値・欠損概念・欠損client／runの純度も検査する。

#### 関係testごとの結果（秒数はJUnitのtestcaseの合計）

| ファイル | 成功 | 未実施skip | 秒数 |
|---|---:|---:|---:|
| `test_fedsda_configuration_variants.py` | 11 | 90 | 40.37 |
| `test_experiment_sweep.py` | 27 | 53 | 0.10 |
| `test_experiment_sweep_resumption.py` | 0 | 31 | 0.00 |
| `test_fedsda_run_client_settings.py` | 122 | 0 | 0.17 |
| `test_fedsda_run_settings.py` | 10 | 0 | 1.30 |
| `test_settings_serialization.py` | 22 | 0 | 0.02 |
| `test_fedsda_default_waited_registration_and_alarm_wait.py` | 3 | 0 | 0.10 |
| `test_server_model_registration_and_aggregation.py` | 28 | 0 | 37.92 |
| `test_model_clustering_and_consolidation.py` | 21 | 0 | 62.88 |
| `test_adopted_candidate_local_adoption.py` | 103 | 0 | 1.19 |
| `test_assigned_training_sample_absorption.py` | 141 | 0 | 1.44 |
| `test_held_model_registration_confirmation.py` | 96 | 0 | 0.38 |
| `test_global_model_distribution.py` | 34 | 0 | 67.63 |
| `test_global_model_distribution_owner_operations.py` | 17 | 0 | 0.02 |
| `test_fedsda_experiment_run_artifacts.py` | 6 | 41 | 5.50 |
| `test_feddrift_experiment_run_artifacts.py` | 20 | 37 | 1.53 |
| `test_held_model_joint_training_iterations.py` | 94 | 0 | 0.65 |

#### 変更したファイル

- `.kiro/specs/fedsda-soft-training-assignment/naming.md`
- `.kiro/specs/fedsda-soft-training-assignment/tasks.md`
- `docs/experiments/experiment-sweep.md`
- `docs/reference/options.md`
- `federated_drift_experiment/experiment_spec/options.py`
- `src/federated_learning_experiments/learning/training/held_model_joint_training_iterations.py`
- `src/federated_learning_experiments/learning/training/model_training_sample_store.py`
- `src/federated_learning_experiments/learning/training/responsibility_weighted_training.py`
- `src/federated_learning_experiments/runtime/adopted_candidate_local_adoption.py`
- `src/federated_learning_experiments/runtime/alarm_change_interval_resolution.py`
- `src/federated_learning_experiments/runtime/assigned_training_sample_absorption.py`
- `src/federated_learning_experiments/runtime/fedsda_run_client.py`
- `src/federated_learning_experiments/runtime/fedsda_run_client_settings.py`
- `src/federated_learning_experiments/runtime/fedsda_run_record_tables.py`
- `src/federated_learning_experiments/runtime/held_model_training_request_handling.py`
- `src/federated_learning_experiments/runtime/model_clustering_and_consolidation.py`
- `src/federated_learning_experiments/runtime/released_sample_training_responsibilities.py`
- `src/federated_learning_experiments/runtime/server_model_registration_and_aggregation.py`
- `src/federated_learning_experiments/sweep/experiment_configurations.py`
- `tests/conftest.py`
- `tests/refactoring/behavior_golden_cases.py`
- `tests/refactoring/soft_training_existing_configuration_hashes.json`
- `tests/refactoring/test_fedsda_configuration_variants.py`
- `tests/refactoring/test_fedsda_soft_training_configuration.py`
- `tests/refactoring/test_model_clustering_and_consolidation.py`
- `tests/refactoring/test_single_run_dependency_boundaries.py`

#### 残る確認

主担当による独立実装レビュー、全pytest、上記240件の権限で未検証のtestの再確認。小さい全体runでは実際のモデルの統合が起きていないので、その操作の数値の根拠は固定入力の単体test。純度と重なりの本規模での比較・既存の研究結果との照合はTask 3の担当であり、まだ実行していない。

### 主担当の確認（2026-11-05）

- 新しいtestと`tests/test_proposed_regression.py`（実装担当の環境では一時NPZの権限で止まった）を主担当のWindowsで流し、33 passed（6 workers、65秒）。新しい実行時のmoduleのimportは標準ライブラリとtorchだけ（`requirements.txt`の範囲）。
- 実装レビュー（GPT-6 Luna medium）: FAIL（Major 1: `ResponsibilityWeightedTrainingStore.discard()`が実行時の経路から呼ばれない。Minor 1: pytestを権限で実行していない）。Majorは不採用: `fedsda`の学習の標本のストアには標本を捨てる経路が無く、配布の時の統合は`remap_model_training_sample_collections`で付け替えるだけ（`learning/training/model_training_sample_store.py:93-108`、柔らかい学習の共有ストアも同じ`remap`に追従する）。保有をやめたモデルの標本は`fedsda`でもストアに残り、学習は保有モデルからだけ抜くので使われない。柔らかい学習の抽出も保有モデルに限る（`learning/training/responsibility_weighted_training.py:180`）ので、扱いは`fedsda`と同じ。`discard()`は設計の表の「捨てる」に備えた単体のtestの範囲で、実行時に呼ぶ経路は無い。再レビューは行わない。
- 全pytest（主担当、Windows、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`-n 8 --dist loadfile tests`、891秒、`.validation-soft-training/full-pytest-claude.xml`）: 13,733 passed、8 skipped、2 failed。失敗は`test_held_model_training_request_handling.py`が共同更新の引数の集合を固定していて、新しい引数`responsibility_training_store`が無かったもの（実装担当の更新漏れ）。期待値に足し、`fedsda`の経路では`None`であることの確認を加え、該当ファイルを再実行して104 passed。全体の再実行はしていない。実装担当の環境で権限のため未確認だった239件は、この全pytestで実行された。
