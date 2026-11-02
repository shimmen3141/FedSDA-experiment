# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求4.1〜4.3のtestを先に書き、REDを確かめてから、学習率の更新の部品、設定の項目（単一のモデルの割り当ての方式、学習率の調整の設定とSGD）、2構成、掃引のオプション、runtimeの分岐（1個のモデルの登録、全標本のID 0への帰属、学習の窓、Adaptive-FedAvgのSGDと時刻ごとの学習率、集約の後の学習率の更新、検出・作成・統合を行わない）、学習率の記録を足す。Oracle・IFCAで足した項目と分岐を再利用する。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCAのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4_
  - _Depends:_ IFCAのspecの実装（`.kiro/specs/feddrift-ifca-configuration/`のTask 1、`7874c449`）
- [x] 2. 文書と検証の記録
  - 要求5.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 5.1_
  - _Depends:_ 1
- [x] 3. 調整・本規模の実行と集計
  - commit・pushの後、主担当が内部確認をして、Windowの本規模とAdaptive-FedAvgの調整のコマンドを渡し、$\eta_0$ を選んでからAdaptive-FedAvgの本規模のコマンドを渡す。結果をFedHASP・FedDrift・FedAvg・Oracle・IFCAと対で集計し、実験の台帳に記録する。
  - _Requirements: 5.2_
  - _Depends:_ 2

## Implementation Notes


### 実装担当の記録（2026-11-01、基点a790da55、独立レビュー前）

- Task 1・2を`kiro-impl`のmanual mode、`--review off`で実装した。依頼に従いcommit・push・stash・レビューCLI・内蔵子エージェント・本規模実験・全pytestは実行していない。独立実装レビューと全pytestは主担当が行う。仕様の承認記録は変更していない。Task 1のcheckboxは権限エラーのtestと主担当の確認が残るため未完了のままとする。
- 新規testを先に書き、学習率の部品がまだ無い`ModuleNotFoundError`（collection error 1件）でREDを確認してからsourceを実装した。公式の`AdaState`の`__init__`・`update`・`current_lr`をtestにそのまま複写した（整形のみRuff）。元checkoutの3ファイルは読み取りだけ。
- 追加構成は`feddrift-window`と`feddrift-adaptive-fedavg-window`。WindowはFedDrift既定から`model_assignment_policy=single_global_model_assignment`と`training_sample_window=current_timestep`だけを変更する。Adaptive-FedAvgはWindowから`adaptive_learning_rate_settings=AdaptiveFedavgLearningRateSettings()`だけを変更する。局所のSGDはその設定からruntimeが生成するので、事前学習の共通設定は変えない。
- 新fieldの既定は`adaptive_learning_rate_settings=None`。有効時は`initial_learning_rate=0.01`、`beta1=beta2=beta3=0.5`（固定）。既存fieldの既定`detection_batch_assignment`・`all_timesteps`・`all_pretrained`は変更しない。既定値は保存設定から省略し、既存FedDriftの識別子`sine2_interval50_seed0_1fbf192c89f9`を確認した。非既定の初期学習率でrun識別子が変わること、明示した0.01と省略時の一致もtestで確認した。
- **Validationの数値の不一致**: designのWindowの学習率「0.001のまま」は、現行の`_MODEL_DEFAULTS_BY_DATASET`（SINE2など合成データ0.01、MNIST系0.001）と一致しない。最初の全体testでSINE2が0.01であることを検出した。「FedDriftと同じoptimizer・学習率」「既存値を変えない」という要求と、構成の差を2項目に限定するdesignのArchitectureを優先し、Windowは現行のデータ別既定値を使う。testの期待値と文書をその値へ修正し、sourceの既定値やgoldenは変更していない。design.mdの承認済み本文は変更していない。
- `experiment_spec/options.py`へ`OptionSpec`・`ActivationRule`・`ChoiceConstraint`を登録。数値は`parameters.py`へ登録した。新APIの数値を旧configの未使用global設定へ複製しないため、`ParameterSpec.configuration_surface`（既定`config`）に`typed_settings`を追加し、数値宣言の参照先を実際の型のfieldとスキーマtestで照合する。runtimeは旧schemaをimportしない。
- 掃引は`ExperimentSweepPlan.adaptive_fedavg_initial_learning_rates`と`--adaptive-fedavg-initial-learning-rates`。Adaptive-FedAvgを含まない計画、重複・非float・非正・非有限値を拒否する。混在掃引では新しい軸はAdaptive-FedAvgだけを増やす。既存の閾値の軸の既存構成への適用は維持した。混在する2つの掃引軸で、既存FedDrift・Oracle・IFCAのrun設定と識別子の列が追加前と完全一致するtestも成功した。
- runtimeの接続: `feddrift_run_participant_factory.py`が単一モデルには1を渡す。`feddrift_run_client.py:assemble_feddrift_run_client`でID 0の共通事前学習モデルを複写し、AdaptiveだけSGD（momentum 0、weight decay 0、初期学習率η_0）を生成する。初期値の一致、独立コピー、torch・NumPy・Pythonの乱数状態不変をtestで確認した。
- `feddrift_client_stream_processing.py:flush_detection_buffer`から`feddrift_detection_batch_processing.py:_assign_single_global_model_samples`へ進み、検査後に学習標本を空にして、その時刻の全標本をID 0へ追加する。予測モデルもID 0のまま。検出の評価・新しいモデルの作成はしない。`feddrift_run_server.py:prepare_timestep`はIFCAと同じ固定モデル集合の分岐で、ID報告の後にクラスタリング・統合を省く。学習対象・件数条件・更新予算・重みは現行FedDriftの規則へ接続する。単一モデルの構成も1時刻1roundに限定する。
- `feddrift_run_server.py:_aggregate_and_distribute_models`で集約の候補を作り、配布の前に、state_dict順の全値をfloat64で連結して学習率を更新する。推定の候補は状態のコピーで計算し、非有限値なら通信やrepositoryの更新前に拒否する。配布する`SgdParameterOptimizerSettings`へ次時刻のηを渡す。学習率のスカラーは通信量に加算しない。
- 独立した部品は`methods/feddrift/adaptive_fedavg_learning_rate.py`。学習率の表はAdaptiveだけの`adaptive_learning_rate`で、`timestep_index`・`aggregation_count`・`variance_ratio`・`smoothed_ratio`・`learning_rate_for_next_timestep`の5列。時刻は0始まり、集約回数は1始まり。保存形式を版5へ上げ、既存構成の表と列は維持した。保存前に記録の検査と同期件数の一致を確認する。
- `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`、構成の一覧と実旧と照合しない集合、依存境界の実sourceの走査と許可集合に登録した。IFCAの専用testの一覧の位置を、新2構成を末尾に置いた位置へ変更した。保存形式testの現行版と未対応版の期待値を版5・版6へ更新した。
- 文書は`docs/experiments/experiment-sweep.md`と`differences-from-feddrift.md`、`python -m tools.generate_option_docs`で生成した`docs/reference/options.md`、`naming.md`を更新した。公式の最後から5つ目のroundの集約を手元の唯一の集約へ置き換える点、共通の事前学習・逐次評価・学習規則の違い、スカラーの通信、候補範囲から初期学習率を選ぶ規則を明記した。

### 検証（指定Windows venv、PYTHONPATH=src;.、数値ライブラリ各1 thread）

- 公式参照との部品照合: seed 51、P=7・T=30、乱数・途中で100を加える急変・一定3・全ゼロの4系列、η_0=0.1・0.001の2値、計8ケース・240回の更新が相対誤差1e-12以内で一致した。推定のoverflow・非有限入力・次元変更を拒否して状態が進まないことも確認。
- 最終の部品・設定・短いrun・CLI・依存境界・オプション/数値スキーマ: **2307 passed / 4 deselected**（`.validation-window/contracts-final.xml`）。うち新規testは26 passed。4 deselectedは別途実行する小さい全体run。
- 追加の保存smokeは2構成で成功。Windowの125列、Adaptive-FedAvgの130列を、版5で保存・読込みして全値が一致した（`.validation-window/artifacts.log`）。tmp_pathを使う既存pytestの権限エラーを、このsmokeの成功に置き換えていない。
- Ruff check・Ruff format --check（388ファイル）とgit diff --checkは成功。命名チェックは公開名の未登録なし。Pyrightの通常の起動はvenvのsite-packagesを発見できずimportエラーとなるため、元のinclude・strict・検査モード・Python版を保った一時設定へsite-packagesだけを追加し、0 errors / 0 warningsを確認（`.validation-window/pyrightconfig.json`・`pyright.log`）。この一時設定は正本へ反映していない。
- 旧回帰: 1 passed / 1 failed（`.validation-window/legacy.xml`、172.91秒）。`test_regression.py`は成功。`test_proposed_regression.py`は一時ディレクトリの操作・cleanupでWinError 5。成功扱いにしない。

### 小さい全体run（要求4.3）

間隔50・seed 0・10 client・各2500標本。標本数の増量は不要だった。各runは500回の学習直前にoptimizerと学習率を検査し、25,000標本の特徴・ラベルを観測streamと完全照合した。両構成とも学習窓はその時刻の50標本だけ、全clientの保有IDは常に{0}、サーバのモデル数は全時刻1、初期登録1件（時刻-1）、検出・統合0件。

| 構成 | データ | 全clientの概念変化の総数 | 学習率の照合 | ηの最小 | ηの最大 | 最初の概念変化より後のη上昇時刻数 |
|---|---|---:|---|---:|---:|---:|
| Window | SINE2 | 24 | Adam、0.01固定 | — | — | — |
| Adaptive-FedAvg | SINE2 | 24 | SGD、毎時刻の参照値と一致 | 0.00016728342919882194 | 0.01 | 12 |
| Window | MNIST4 | 22 | Adam、0.001固定 | — | — | — |
| Adaptive-FedAvg | MNIST4 | 22 | SGD、毎時刻の参照値と一致 | 0.00017770215437139074 | 0.01 | 11 |

上昇時刻数は、最初にいずれかのclientで概念変化があった時刻より後で、集約後のηが前時刻の集約後のηより大きい時刻の数。個々の変化への応答や改善の判定ではない。Adaptiveの各集約後のパラメータを公式参照へ渡し、記録した全50時刻のηと相対誤差1e-12以内で照合した。

- 小さい全体runを含む新規ファイルの再実行: **19 passed**（`.validation-window/new.xml`、946.01秒）。この実行の15件の部品testと最終2307件に含まれる26件の新規testは重複するため加算しない。新規testの最終の種類は26件の部品等と4件の小さい全体runの計30件。
- 基準版1〜16のgolden: **数値照合206件が全て成功**。golden関連ファイル全体は380 passed / 19 errors / 0 failures（`.validation-window/golden.xml`、742.27秒）。19件は全てtmp_path準備等のPermissionError / WinError 5。成功扱いにしない。旧golden・既存の設定値・基準版は更新していない。
- 保存形式の既存testの最終版の再実行: 20 passed / 37 errors / 0 failures（`.validation-window/artifacts-final.xml`）。37件はtmp_path準備の権限エラー。既存の関連testの一括実行と重複するので、総件数へ加算しない。独立した追加保存smokeは上記のとおり成功したが、pytestの権限エラーは残る。

### 既存の関連testの最終結果と残る確認

- FedDrift全関連file、Oracle、IFCA、構成・掃引・識別子・再開の一括実行は**1057 passed / 220 errors / 14 skipped / 0 failures**（`.validation-window/related.xml`、2366.62秒）。Oracle専用11件・IFCA専用15件は、それぞれ全件成功した（固定条件の全体runを含む）。
- 5 warningsは、実旧の1標本での比較testが旧実装`clients/base.py`の`torch.var`で出す自由度の警告。旧実装の値や処理は変更していない。
- 220 errorsは全てPermissionError / WinError 5。内訳: `test_feddrift_experiment_run_artifacts.py` 37、`test_feddrift_experiment_sweep.py` 16、`test_experiment_sweep.py` 53、`test_experiment_sweep_resumption.py` 31、`test_fedavg_experiment_sweep.py` 3、`test_fedsda_configuration_variants.py` 80。14 skippedの内訳はLinux CPU専用4件、全条件のrunを完走できなかった構成の経路網羅確認10件。後者は上記の権限エラーがある実行で、網羅を成功扱いにしないための既存fixtureのskip。
- JUnitのtestcase（module名とtest名）で重複を除いた今回の確認範囲は**4003件: 3749 passed / 239 errors / 1 failed / 14 skipped**。239 errorsはgolden補助19件と関連220件、1 failedは旧提案回帰の一時ディレクトリ操作。権限で実行不能・終了不能の240件を成功扱いにしていない。保存形式の37 errorsの再実行は重複のため加算しない。
- Task 2の文書・検証記録は確認済み。Task 1はコードと対象の数値確認を実施済みだが、上記の権限で未確認のtest、仕様内のWindowの学習率の不一致の主担当確認、独立実装レビュー・全pytestが残る。specのphaseはimplementationのまま、完了・GOとは判定しない。次は主担当による差分とこの記録の確認、独立実装レビュー、権限を満たす環境での未確認testと全pytest。

### 変更ファイル

- `.kiro/specs/feddrift-window-and-adaptive-fedavg-configuration/naming.md`
- `.kiro/specs/feddrift-window-and-adaptive-fedavg-configuration/tasks.md`
- `docs/experiments/differences-from-feddrift.md`
- `docs/experiments/experiment-sweep.md`
- `docs/reference/options.md`
- `federated_drift_experiment/experiment_spec/options.py`
- `federated_drift_experiment/experiment_spec/parameters.py`
- `src/federated_learning_experiments/methods/feddrift/adaptive_fedavg_learning_rate.py`
- `src/federated_learning_experiments/runtime/feddrift_client_stream_processing.py`
- `src/federated_learning_experiments/runtime/feddrift_detection_batch_processing.py`
- `src/federated_learning_experiments/runtime/feddrift_run_artifact_files.py`
- `src/federated_learning_experiments/runtime/feddrift_run_client.py`
- `src/federated_learning_experiments/runtime/feddrift_run_client_settings.py`
- `src/federated_learning_experiments/runtime/feddrift_run_participant_factory.py`
- `src/federated_learning_experiments/runtime/feddrift_run_record_tables.py`
- `src/federated_learning_experiments/runtime/feddrift_run_server.py`
- `src/federated_learning_experiments/sweep/experiment_configurations.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_command.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_plan.py`
- `tests/refactoring/behavior_golden_cases.py`
- `tests/refactoring/test_feddrift_experiment_run_artifacts.py`
- `tests/refactoring/test_feddrift_ifca_window_configuration.py`
- `tests/refactoring/test_feddrift_window_and_adaptive_fedavg_configuration.py`
- `tests/refactoring/test_fedsda_configuration_variants.py`
- `tests/refactoring/test_single_run_dependency_boundaries.py`
- `tests/test_parameter_schema.py`

### 主担当の確認（2026-11-02）
- 実装レビュー（GPT-6 Luna、effort medium、対象は`a790da55`の上の未commitの差分）: PASS（Major以上なし）。
- 全pytest（`tests`、Windows、12並列、2718秒、JUnit `../../venv/refactoring-tests/window-adafedavg-full-pytest.xml`）: 13,305 passed、8 skipped、2 failed。実装担当の環境で権限のため実行できなかったtestを含む。
- 2件の失敗は、掃引の計画の記録の項目（`test_experiment_sweep.py`）と掃引のコマンドの引数の一覧（`test_fedsda_default_waited_registration_and_alarm_wait.py`）の期待値に、新しい`adaptive_fedavg_initial_learning_rates`を足していなかったため。主担当が期待値に足し、2ファイルの81件が通ることを確かめた。
- Windowの学習率: 仕様に「0.001」と書いていたが、`feddrift`の既定はデータごとに違う（SINE2は0.01、MNIST4は0.001）。実装担当は既存の値を保った。仕様の文面を直した。
- 論文での表記は「FedAvg+Window」（2026-11-01のユーザー判断）。構成の名前は`feddrift-window`のまま。

### η0の調整の結果（2026-11-02、主担当）
- `results/adaptive-fedavg-lr-tuning-v16/20261102_083406_lr-grid-seeds100-102`。480 run完了・失敗0。集計は主担当のscratchpadの`adafedavg_select.py`（Git管理外。既定の0.01は保存設定から省かれるので、項目が無いrunを0.01として読む）。
- 40セル中39で上端の0.1が最良（残りはCIRCLE2・間隔500の0.01）。頭打ちではなく、0.01→0.1で、画像データは+1〜+20pp上がった（rotmnist2・間隔50で58.3%→74.6%、rotmnist4・間隔50で41.8%→60.6%）。
- 原因の見込み（確かめていない）: 学習率は $\eta_0\hat\gamma_t/t$ で集約の回数 $t$ に反比例して下がる。手元は間隔50で100回集約するので、学習率は最後に初期のおよそ1/100になる。FedDriftの論文の設定は時刻が10ほどで、下がり方が小さい。そのため、手元では大きい $\eta_0$ ほど有利になる。間隔が長い（集約が少ない）ほど精度が高いことも、これと合う。
- 設計では「端でも広げない」としていたが、FedDriftのδの上端（頭打ち）と違い、大きく上がり続けている。比較手法を弱く扱っていると読まれうるので、上へ広げる（0.3・1.0、240 run）かをユーザーに確かめる。

### FedAvg+Windowの本規模の結果（2026-11-02、主担当）
- `results/method-comparison-window-v16/20261102_064312_window-seeds0-9`（uranium）。400 run完了・失敗0。
- FedHASP−FedAvg+Window: −0.03〜+39.02pp、平均+14.42、FedHASPが上39/40セル（全seedで上37）。下はSEA2・間隔200の1セル（−0.03pp）。
- FedAvg+Window−FedAvg（集約ごと）: 平均−4.38pp、上は5/40セル（SINE2の4セルとCIRCLE2・間隔50）。画像データでは大きく下（rotmnist2・間隔50で−16.3pp）。窓1時刻は「間隔分の標本だけ」で、間隔50では1時刻50標本しか学習に使わない（FedDriftの論文の設定は1時刻500標本）。
- IFCAのID 0だけ事前学習した版は、SINE2以外の36セルで10 seedの値がFedAvg+Windowと完全に一致した（IFCAはSINE2以外でID 0だけを選んだ。選択の割合を間隔50・500で集計して確かめた）。FedDriftの論文の付録の「IFCA-WはSEA・MNISTでWindowに退化した」と同じ。
- 通信のFedAvg+Window/FedHASPは、間隔50で0.46〜0.94。

### η0の調整の確定（2026-11-02、主担当とユーザー）
- 上端の0.1が39/40セルで最良で精度が上がり続けたので、ユーザーの判断で上へ広げた。0.3・1.0を全40セル（`20261102_175330_lr-extend-up-seeds100-102`、240 run）、1.0がまだ上端だった11セルに3.0（`20261102_182941_lr-extend-up2-images`・`-sine2`・`-mnist2`、33 run）。どれも失敗0。
- 3.0は11セルすべてで1.0より下がった（例: rotmnist4・間隔50で76.07%→57.70%、SINE2・間隔100で61.09%→55.81%）。全40セルで最良が格子の内側に入った。
- 選んだη0: 0.01が1セル（CIRCLE2・間隔500）、0.1が8セル（CIRCLE2の間隔100・200、SEA2の間隔100・500、SEA4の間隔100・200・500）、0.3が20セル、1.0が11セル（SINE2の全間隔、MNIST2・間隔50、MNIST4・rotmnist2・rotmnist4の間隔50・100）。表は`results/adaptive-fedavg-lr-tuning-v16/selected-initial-learning-rate.csv`（候補0.0001〜3.0）。
- 短い間隔（集約が多い）ほど大きいη0が選ばれた。学習率が $\eta_0\hat\gamma_t/t$ で集約の回数に反比例して下がることと合う。論文では、元の論文の候補（$10^{-4}$〜$10^{-1}$）の上端が最良で上がり続けたので、上へ0.3・1.0・3.0を足し、全セルで内側の最良を選んだと書く。
- 本評価: seed 0〜9、12本のコマンドで400 run（`--study method-comparison-adaptive-fedavg-v16`）。

### Adaptive-FedAvgの本規模の結果（2026-11-03、主担当）
- `results/method-comparison-adaptive-fedavg-v16`の12フォルダ（uranium）。400 run完了・失敗0。各runの初期の学習率が、選んだ値（`selected-initial-learning-rate.csv`）と一致することを集計の中で確かめた。
- FedHASP−Adaptive-FedAvg: +0.08〜+38.91pp、平均+11.41、40/40セルで上（全seedで上38）。
- Adaptive-FedAvg−FedAvg+Window: 平均+3.02、上33/40（学習率の調整の効果）。Adaptive-FedAvg−FedAvg（集約ごと）: 平均−1.37、上8/40。
- Oracleを除く比較手法の中でFedHASPが最良のセルは35/40で変わらない（残りはSEAの5セル。FedDriftが4、FedAvgが1）。
