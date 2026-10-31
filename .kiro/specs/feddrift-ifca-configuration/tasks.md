# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求2.1〜2.3のtestを先に書き、REDを確かめてから、設定の項目（割り当ての方式の新しい値、学習の窓、クラスタのモデルの初期化）、2構成、runtimeの分岐（$K$ 個のモデルの登録と構成ごとの初期化、損失が最小のモデルの選択、学習の窓、統合を行わない）、選択の記録を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDriftとOracleのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4_
  - _Depends:_ Oracleのspecの実装（`.kiro/specs/feddrift-oracle-concept-assignment-configuration/`のTask 1）
- [x] 2. 文書と検証の記録
  - 要求3.1の文書。実装レビュー（独立CLI）と、関係するtestの結果をImplementation Notesに記録する。全pytestは主担当が流す。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 本規模の実行と集計
  - commit・pushの後、主担当が内部確認をしてコマンドを渡す。結果をFedHASP・FedDrift・FedAvg・Oracleと対で集計し、実験の台帳に記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes

### Task 1・2の実装（2026-11-01、基点e2dae549、未コミット）

- この依頼では主担当が独立レビューを行う。実装担当はレビューCLI・内蔵子エージェントを起動していない。独立レビュー・全pytest・本規模実験・commit・push・stashは未実施。仕様承認の記録は変更していない。checkboxは、権限で実行不能のtestと主担当の確認が残るため未完了のままとする。
- `kiro-impl`のmanual mode、`--review off`でTask 1・2を実装。testを先に追加し、3 failed（未知の2構成、未定義の設定field）、4 deselectedのREDを確認してからsourceへ着手した。
- 構成は`feddrift-ifca-window`、`feddrift-ifca-window-pretrained-cluster0`。既定のFedDriftから、`model_assignment_policy=minimum_loss_cluster_assignment`、`training_sample_window=current_timestep`、`cluster_model_initialization=all_random`／`pretrained_cluster0`だけを変える。新fieldの既定は`all_timesteps`／`all_pretrained`。割当方式の既定は既存の`detection_batch_assignment`。既定値は保存設定から省き、既存の識別子`sine2_interval50_seed0_1fbf192c89f9`を確認した。
- Oracleと同じ`experiment_spec/options.py`へ新API専用の`OptionSpec`・`ActivationRule`・`ChoiceConstraint`を登録し、`python -m tools.generate_option_docs`で設定参照を再生成した。数値パラメータ・掃引軸は追加していない。runtimeは旧schemaをimportしない。
- モデル数はfactoryが`concept_count`から渡す。`assemble_feddrift_run_client`でseedとモデルIDから分けたCPU torch乱数を使い、全Linearの`reset_parameters`で初期化する。乱数モデルのoptimizerは新規、勾配は空。client間は同じID・seedで同じ値を持つ独立のコピー。共通の事前学習処理は生成系列を揃えるため実行し、全乱数版のクラスタにはそのパラメータを使用しない。torch・Python・NumPyの乱数状態の不変、両構成間のID 1以降の一致を部品testで確認する。
- runtimeの分岐: `feddrift_run_participant_factory.py`（Kの決定）、`feddrift_run_client.py`（初期化）、`feddrift_client_stream_processing.py:flush_detection_buffer`→`feddrift_detection_batch_processing.py:_assign_minimum_loss_cluster_samples`（最初は比較せずID 0、以後はID昇順で平均有界損失の最小を選ぶ。同値は小さいID。全モデルの標本を空にして今のバッチを追加）。`feddrift_run_server.py`は初期Kモデルの実パラメータをrepositoryへ設定し、`prepare_timestep`でID報告の後にクラスタリング・統合を省く。学習対象の和集合・件数の条件・更新予算・集約重み・通信は既存のFedDriftへ接続。IFCAの構成は1時刻1roundに限定する。
- 「平均損失」は既存の`assess_feddrift_detection_batch`の既定の有界損失として実装した。通常のFedDrift構成の比較基準である誤り率はIFCAの選択には使わない。この点を文書に明示し、部品testでは実モデルの有界損失と照合する。
- 記録はIFCAだけの`cluster_selection`表。列は`client_id`、`timestep_index`、`first_sample_index`、`last_sample_index`、`selected_model_id`、`model_ids`、`model_mean_bounded_losses`。後二者は行×K、ID昇順。最初の損失は未比較のNaN。既存の検出バッチ表は空。指標導出・保存前に選択件数・標本範囲・最小損失・予測モデルとの対応を検査する。新しい表の追加に伴い保存形式を版4へ上げた（Oracleの版3の列と既存構成の表は維持）。
- `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`、`test_fedsda_configuration_variants.py`の構成の一覧と実旧と照合しない集合、依存境界の許可に登録した。Oracleの専用testの「末尾がOracle」を「一覧にOracleを含む」に直した。
- 文書は`docs/experiments/experiment-sweep.md`・`differences-from-feddrift.md`・`docs/reference/options.md`（生成）、命名表を更新した。論文と共通の初期割当、1roundの選択、評価の時点・真のK・学習規則の差、事前学習版の位置付けを明示する。

### 検証（Windows、指定venv、各数値ライブラリ1 thread）

- 新規部品・設定・短い全体runと選択記録の破損の拒否: 最終の実行は11 passed / 4 deselected（`.validation-ifca/new-unit.xml`）。追加の最小全体runによる版4の保存・読込みは2構成とも132列が一致した（NaNを含む）。
- 既存構成についても通常のworkspaceの出力先を使った追加smokeを実行し、FedDriftの125列・Oracleの126列の保存・読込みが完全一致した。両構成に`cluster_selection`表が無く、既定の窓・初期化のfieldが保存設定から省かれることも確認した。権限で実行できないpytestを、この追加smokeの成功へ置き換えてはいない。
- 依存境界・設定/数値スキーマ・新規部品の実行: 2290 passed / 4 deselected（`.validation-ifca/contracts.xml`）。この後に追加した短い全体run2件は上記の11件の実行で確認。
- 基準版1〜16のgolden数値照合は206件すべて成功。golden関連ファイル全体は380 passed / 19 errors / 0 failures（`.validation-ifca/golden.xml`、422秒）。19件は構造の破損・生成CLI等のtestのtmp_path準備がWinError 5で拒否されたもの。数値の不一致は無い。終了時のtmpディレクトリ走査も拒否され、process exitは非0。成功扱いにしない。
- Ruff check、Ruff format --check（386ファイル）、git diff --checkは成功。Pyrightの通常の呼出しはvenvのsite-packagesを発見せず196 errors（import解決）。診断ログで検索パスの欠落を確認し、pyproject.tomlのinclude・strict・Python版・検査モードを保ち、venvのsite-packagesだけを検索パスへ追加した一時設定で0 errors / 0 warningsを確認（`.validation-ifca/pyrightconfig.json`・`pyright-explicit-paths.log`）。一時設定をリポジトリの正本へ反映していない。
- `spec_checks.py names ... --base e2dae549`は新規公開名の未登録なし。全pytestと独立実装レビューは主担当の担当として未実施。
- 旧回帰も実行し、`tests/test_regression.py`は1 passed、`tests/test_proposed_regression.py`は1 failed（TemporaryDirectoryの操作・cleanupがWinError 5）。後者は成功扱いにしない（`.validation-ifca/legacy-regression.xml`、計147秒）。

### 小さい全体run（要求2.3）

間隔50、seed 0、10 client、各2500標本。5000標本への増量は不要だった。`pytest tests/refactoring/test_feddrift_ifca_window_configuration.py -q -s -p no:cacheprovider`の初回全体実行は7 passed、1309.43秒（部品test3件と下記4 run。その後に追加した部品・短いrunは上記の11件で実行）。

| 構成 | データ | client内の概念の変化の合計 | IDごとの選択回数（0から順） | 学習窓で照合した標本数 | 初期登録数 | 統合数 |
|---|---|---:|---|---:|---:|---:|
| 全乱数 | SINE2 | 24 | 303, 197 | 25000 | 2 | 0 |
| ID 0だけ事前学習 | SINE2 | 24 | 303, 197 | 25000 | 2 | 0 |
| 全乱数 | MNIST4 | 22 | 500, 0, 0, 0 | 25000 | 4 | 0 |
| ID 0だけ事前学習 | MNIST4 | 22 | 500, 0, 0, 0 | 25000 | 4 | 0 |

- 全4 runで最初の10 clientの選択はID 0。2時刻目以降の490件/runは、記録したK個の平均有界損失のargminと一致した。同値の小さいIDの選択は、保有順を逆にした部品testでも確認した。
- 学習直前のownerを毎回観測し、選択モデルだけが50件を持ち、他モデルの標本集合が空であることを確認した。その50件の特徴量とラベルを、runの観測列の該当時刻の標本番号の入力に完全一致で照合した。各clientのID集合は毎時刻0〜K−1、予測時の保有数もK、サーバの同期後のモデル数もKだった。登録は最初のK個だけで時刻は−1、クラスタリング・統合の記録は0件だった。
- MNIST4でID 1以降が選ばれなかったことは結果として残す。仕様はID 1以降の選択を要求しない。これは挙動の検証であり、手法の優劣の比較ではない。全体精度・通信・MACの比較集計は未実施、計算の経過時間は検証suite全体の上記実測のみ。学習窓の観測・Tensor照合の診断費用も含まれる。

### 関連suiteの最終結果と未確認

- 初回の逐次実行は、新規の4 runの重複を含んでいたためCtrl-Cで中断し、件数を成功として集計していない。最終差分への確認は、新規ファイル（上で別途実行済み）を除いた同じ関係ファイルを8並列・`--dist loadfile`で実行し直した。
- 対象は`test_feddrift_*.py`（新規IFCAのファイルを除く）、`test_single_run_dependency_boundaries.py`、`test_fedsda_configuration_variants.py`、`test_experiment_sweep.py`、`test_experiment_sweep_resumption.py`、`tests/test_option_schema.py`、`tests/test_parameter_schema.py`。
- **3322 passed / 217 errors / 14 skipped / 0 failures**、1019秒（`.validation-ifca/related-final.xml`）。OracleのSINE2・MNIST4の全体runも両方成功した。217件すべてがtmpディレクトリの準備等の`PermissionError / WinError 5`であり、成功として扱わない。終了時のtmp走査も拒否され、process exitは非0。

| 権限で実行不能のtestファイル | 件数 |
|---|---:|
| `test_experiment_sweep.py` | 53 |
| `test_feddrift_experiment_run_artifacts.py` | 37 |
| `test_feddrift_experiment_sweep.py` | 16 |
| `test_fedsda_configuration_variants.py` | 80 |
| `test_experiment_sweep_resumption.py` | 31 |

- skipは、構成の小条件の全実行を前提とする確認10件（上記の実行不能があり全条件が揃わない）と、固定Linux CPU専用4件。成功件数に含めない。
- 最終のRuff check・format --checkは成功（`.validation-ifca/ruff.log`・`ruff-format.log`）。検査範囲を保った検索パス明示のPyrightも0 errors / 0 warnings（`pyright-final.log`）。新規構成の差分を3項目に限定するassertを強めた後、11件の部品・短いrunと対象fileのRuffを再実行して成功した。
- `spec_checks.py identity`へ部分JUnitの`contracts.xml`を渡した実行は、未コミット・固定旧差分・旧回帰を含まないJUnitの項目がNGであり、承認ゲート通過とは扱わない。全pytestと独立レビューは今回の担当範囲外で、主担当が実施する。
- 残りは、主担当の独立レビュー、権限で実行できなかった上記217件・golden関連19件・旧最終回帰1件の再確認、全pytest。Task 3は未着手。コード・golden・実行条件の値を、環境エラーを理由に変更していない。

### 変更ファイル

- `.kiro/specs/feddrift-ifca-configuration/`: `tasks.md`、`naming.md`。
- `src/federated_learning_experiments/sweep/experiment_configurations.py`。
- `src/federated_learning_experiments/runtime/`: `feddrift_run_client_settings.py`、`feddrift_run_participant_factory.py`、`feddrift_run_client.py`、`feddrift_client_stream_processing.py`、`feddrift_detection_batch_processing.py`、`feddrift_run_server.py`、`feddrift_run_metric_derivation.py`、`feddrift_run_record_tables.py`、`feddrift_run_artifact_files.py`。
- `src/federated_learning_experiments/learning/training/model_training_sample_store.py`。
- `federated_drift_experiment/experiment_spec/options.py`（新APIの宣言だけ）。
- `tests/refactoring/`: `test_feddrift_ifca_window_configuration.py`（新規）、`behavior_golden_cases.py`、`test_single_run_dependency_boundaries.py`、`test_fedsda_configuration_variants.py`、`test_feddrift_oracle_concept_assignment_configuration.py`、`test_feddrift_experiment_run_artifacts.py`。
- `docs/experiments/experiment-sweep.md`、`docs/experiments/differences-from-feddrift.md`、`docs/reference/options.md`（再生成）。
- 検証ログとJUnit、最小runの保存物はGit管理外の`.validation-ifca/`。作業開始前から存在した`.tmp/`・他の`.validation-*`は変更していない。

### 主担当の確認（2026-11-01）
- 実装レビュー（GPT-6 Luna、effort medium、対象は`e2dae549`の上の未commitの差分）: PASS（新しい指摘なし）。
- 全pytest（`tests`、Windows、12並列、1894秒、JUnit `../../venv/refactoring-tests/ifca-full-pytest.xml`）: 13,273 passed、10 skipped、失敗0。実装担当の環境で権限のため実行できなかったtestを含む。
- 小さい全体runで、MNIST4ではID 0だけが選ばれた（2構成とも500回中500回）。最初の時刻に全clientがID 0で学習し、一部のラベルの入れ替えでは学習済みのID 0の損失が乱数のモデルより小さいまま、という見込み（IFCAの性質で、仕様の範囲）。本規模の結果でも確かめ、論文に書く。
