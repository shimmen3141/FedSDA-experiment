# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求2.1〜2.3のtest（分割の判定の部品、分割が決定的に起きる入力でのサーバの集約の経路、小さい全体run）を先に書き、REDを確かめてから、設定の項目（割り当ての方式の新しい値、$\gamma$）、構成、サーバの分割の判定と集約の省略、割り当ての保持、記録を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・DriftSurfのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4_
- [x] 2. 文書と検証の記録
  - 要求3.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と本規模の実行と集計
  - commit・pushの後、主担当が内部で計画（run数と重複先）を確かめ、予備の実行（24 run）のコマンドを渡す。分割の回数・時刻・モデルの数・精度を集計してユーザーに示し、本規模へ進むかをユーザーが決める。進むなら本規模（400 run）のコマンドを渡す。結果を論文の表1の手法と対で集計し、開始と結果を実験の台帳に記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes

### 2026-11-03 Task 1・2の実装（独立レビュー前）

- 基点: `refactor/architecture`、HEAD `5d4598d4`。この作業ではcommit・push・stash、レビューCLI、予備・本規模の研究実行をしていない。既存の未追跡資料・検証ディレクトリは変更していない。
- 主担当からの実装委任に従い、`kiro-impl`の手動実行・`--review off`で進めた。独立実装レビューと全pytestは主担当が行う。仕様レビューの過去の承認は変更していない。
- Task 1のコードとtest、Task 2の文書は作成済み。ただし必須の保存・掃引testの一部が権限で実行できないので、Task 1のcheckboxは未完了のまま。最終承認・実装レビューを済ませたという意味ではない。

#### 設定と実行経路

- 構成名`feddrift-cfl-window`を末尾へ追加。`model_assignment_policy="cfl_bipartition_assignment"`、`training_sample_window="current_timestep"`、`cluster_model_initialization="all_pretrained"`。
- 新しい数値項目`cfl_split_gamma: float = 0.1`。有限の0〜1、CFL以外で既定と異なる値は拒否。OptionSpec・ActivationRule・ChoiceConstraintとParameterSpec（`run_client_settings.cfl_split_gamma`、`γ_CFL`）へ登録。CLI・掃引軸は追加していない。
- 既存構成の既定は`detection_batch_assignment`・`all_timesteps`・`all_pretrained`のまま。γの既定値は保存設定から省く。FedDriftの識別子`sine2_interval50_seed0_1fbf192c89f9`を照合した。
- `feddrift_run_participant_factory.py`: 初期モデルは1つ、上限はdatasetの`concept_count`をサーバへ渡す。CFLも1時刻1roundを検査する。
- `feddrift_client_stream_processing.py`: 各時刻の最初の観測の前に、前の分割後の割り当てを記録へ写す→全モデルの学習標本を空にする。時刻末に今のバッチを現在の割り当てへ追加する。予測は既存の`current_training_model_state`を使う。損失比較・検出・検出によるモデル作成へ入らない。
- `feddrift_run_server.py`: 既存の`_fixed_model_pool`のprepare分岐を使い、交差評価・統合をしない。集約直前に`split_cfl_training_round`へ接続。
- `runtime/cfl_training_round.py`: 学習標本がある割当clientだけについて、`state_dict`順に`local[k]-distributed[k]`を平らに連結する（snapshotはbufferも含むstate_dict）。判定が1つでも分割なら、新IDを登録し、元モデルと新モデルをθ0へ戻す。既存の配布部品に`active_ids=()`・`update_model_weights=False`を渡して全モデルの集約を省き、全モデルを配布する。学習の上り通信は記録し、集約MACは足さない。配布後にcl2のclientの割り当てを新IDへ変える。
- `methods/feddrift/cfl_cluster_splitting.py`: run全体の`cfl_norm`・`eps1`・`eps2`・次IDを保持。クラスタID昇順、閾値の更新、厳密な不等号、余弦類似度、alpha_cross、上限Kを公式の順序で判定する。更新がないクラスタは飛ばす。
- 完全連結2分割は近隣鎖→距離の安定ソート→内部節点IDの割当→sklearnと同じheapによる木の切り方。runtimeはscikit-learn・SciPyへ依存しない。開発用に`scikit-learn==1.9.1`を追加し、goldenの再現環境の版固定へ含めない旨を書いた。
- 保存形式を6から7へ上げ、CFLだけ`cfl_assignment`・`cfl_split_check`を追加した。列と欠損値の意味はnaming.mdと`differences-from-feddrift.md`に記載。

#### 検証の証拠

- RED: 新testを先に追加し、未実装`methods.feddrift.cfl_cluster_splitting`の`ModuleNotFoundError`で収集エラー1件を確認した。
- 部品: 乱数の対称行列30個と同点を含む3行列で、scikit-learn 1.9.1の`AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete")`とラベルを完全一致で照合。ラベルの代替規則は採用していない。
- 公式の写しから作った参照で、閾値更新、分割、上限、空の更新、単一client、クラスタID順による判定の違い、固定seedの10client×20次元×20round、alpha_crossが0.98より大きく分割しない円周の入力を照合した。参照内のモデル再初期化だけは両モデルのθ0へ変えた。
- サーバ経路: 既存2クラスタを用意し、時刻1で片方を決定的に分割。判定とcl1/cl2の対応、元モデルと新モデルのθ0、他クラスタを含む全モデルの集約省略、全clientへの配布、次時刻の割り当て・予測・空の学習窓を同じtestで照合。
- 小さい全体run（間隔50、seed 0、10client×2500標本）: SINE2は概念変化24件、分割0回（時刻なし）、最終1モデル。MNIST4は変化22件、分割0回（時刻なし）、最終1モデル。両方、各clientの割り当ての保持、予測モデル、25,000標本の窓の件数・特徴量・ラベル、ノルムと閾値からの判定、上限、検出0・統合0を照合。2 passed（407.43秒、最初の実行）。これを予備実行24 runの代わりにはしていない。
- 関連test（構成・掃引・識別子・依存境界・Oracle・IFCA・Window・Adaptive-FedAvg・DriftSurfの非slow、権限エラーの87件を除外した再実行）: **2414 passed, 105 deselected**。除外は既知の権限エラー87件とslow18件。`.validation-cfl/focused-final.xml`。
- スキーマとFedDriftのclient・server・学習・stream・factory・公式整合: **498 passed**。`.validation-cfl/schema-feddrift-final.xml`。文言修正後のスキーマ3ファイルは別に**28 passed**（`schema-final.xml`）。
- 構成の一覧と非旧照合集合: `test_fedsda_configuration_variants.py::test_configuration_names_build_run_settings_that_differ_only_in_the_changed_element`、**1 passed**（他100件は選択しない）。`configuration-registry.xml`。
- 基準版1〜16（Windows）: **380 passed, 19 errors**、失敗0。うち実runとgoldenの照合は206件すべて成功。19件はgolden保存形式などを調べるtmp_path依存のtestで、すべてsetup時のWinError 5。pytest全体の終了もtmpの後処理で権限エラーなので、コマンド全体を成功とは扱わない。`.validation-cfl/golden.xml`。
- `tests/test_regression.py`: **1 passed**（73.21秒）。`legacy-regression.xml`。goldenの値は変更していない。
- Ruff: `python -m ruff check src tests/refactoring`成功、`python -m ruff format --check src tests/refactoring`は395ファイル整形済み。
- Pyright: 通常の`--pythonpath ../../venv/Scripts/python.exe`ではvenvのsite-packagesが検索対象に入らず、numpy/torch等の未解決202エラー。同じinclude・strict・typeCheckingModeを維持し、venvのsite-packagesだけをextraPathsへ足した一時設定`.validation-cfl/pyright-config.json`では**223 source files、0 errors / warnings**（`pyright-final.log`）。リポジトリのPyright設定は変更していない。
- `spec_checks.py names ... --base 5d4598d4`: 新規の公開名4つはすべて登録済み、未登録なし。`git diff --check`成功。
- 参照のdtypeの精査で、float32のノルムを閾値の計算前にPython floatへ変えるとNumPy 2.4の公式の計算と末尾の桁がずれることを検出。専用testでRED（`0.24596748352050782 != 0.24596747756004333`）を確認し、NumPyの元のdtypeで`/10`と`*6`を計算してから記録用のfloatへ変換する形へ修正した。参照も公式どおりの型で計算する。修正後のCFL非slowは**38 passed, 2 deselected**（`cfl-components-final.xml`）。

#### 実行できなかった検証と引継ぎ

- 保存・掃引・artifactの**87件**とgolden構造の**19件**、合計**106件**がpytestの一時ディレクトリへの`PermissionError: [WinError 5]`でsetup失敗。アクセス権やsandbox設定は変えていない。対象nodeと理由は`.validation-cfl/focused.xml`・`golden.xml`に記録。権限がある主担当の環境で再実行が必要。
- 独立実装レビュー、全pytest、Task 3の予備・本規模実行は未実施。主担当の担当。
- 既存slowの実行はOracleのSINE2・MNIST4が通過した表示を確認したところで中断した（コマンド終了コード1、JUnitなし。suite全体の成功とは扱わない）。SINE2は変化24件・2モデル、MNIST4は変化22件・4モデル。IFCA 4件、Window/Adaptive-FedAvg 4件、DriftSurf 3件と、掃引のslow 3件は主担当の全pytestで検証する。これら14件を成功扱いにしない。
- float32の修正と判定照合の強化後、CFL test全体は**40 passed、403.89秒**（`.validation-cfl/cfl-final-float32.xml`）。うち非slow38件、必須の小さい全体run2件。全体runの結果は最初の実行と同じく、SINE2は変化24件・分割0回・最終1モデル、MNIST4は変化22件・分割0回・最終1モデル。各25,000標本の窓・予測、判定・割当・上限・検出0・統合0の照合に成功した。
- 最終の成功JUnit（関連2414・schema/FedDrift 498・golden 380・CFL 40・構成一覧1・旧回帰1）の重複を除いた成功caseは3296件。これに、コマンド中断前に通過を確認したOracleの全体run2件がある。権限エラー106件と未実施slow14件をこの成功件数へ含めない。

#### 変更ファイル

- 新規: `src/federated_learning_experiments/methods/feddrift/cfl_cluster_splitting.py`、`src/federated_learning_experiments/runtime/cfl_training_round.py`、`tests/refactoring/test_feddrift_cfl_window_configuration.py`。
- runtime既存: `feddrift_run_client_settings.py`、`feddrift_run_client.py`、`feddrift_client_stream_processing.py`、`feddrift_run_participant_factory.py`、`feddrift_run_server.py`、`feddrift_run_record_tables.py`、`feddrift_run_artifact_files.py`。
- 登録: `src/federated_learning_experiments/sweep/experiment_configurations.py`、`federated_drift_experiment/experiment_spec/options.py`・`parameters.py`、`requirements-dev.txt`。
- test既存: `tests/refactoring/behavior_golden_cases.py`、`test_experiment_sweep.py`、`test_feddrift_experiment_run_artifacts.py`、`test_feddrift_ifca_window_configuration.py`、`test_feddrift_window_and_adaptive_fedavg_configuration.py`、`test_fedsda_configuration_variants.py`、`test_fedsda_default_waited_registration_and_alarm_wait.py`、`test_single_run_dependency_boundaries.py`、`tests/test_parameter_schema.py`。
- 文書: `docs/experiments/experiment-sweep.md`・`differences-from-feddrift.md`、生成した`docs/reference/options.md`、対象specの`naming.md`・`tasks.md`・`spec.json`。検証出力はgit管理外の`.validation-cfl/`。

### 主担当の確認（2026-11-03）

- 独立実装レビュー（GPT-6 Luna、effort medium、`results/_sol_prompts/cfl-impl-review.out`）: PASS（Major以上なし）。
- 全pytest（主担当、Windows、`-n 8 --dist loadfile`）: 13,378 passed、8 skipped、失敗0・エラー0（2,085秒）。JUnitは`.validation-cfl/full-pytest-claude.xml`。実装担当の環境で権限のため実行できなかった106件とslow 14件も、この中で通った。
- Task 1のcheckboxを完了にした。Task 3（予備の実行）はユーザーの指示（2026-11-03）で手元で流す。

### 予備の実行の結果（2026-11-03、手元）

- `results/cfl-window-pilot-v16/20261103_161325_split-check-seeds100-102`、24 run、失敗0。分割は全runで0回。「平均の更新ノルム<eps1」が1回も成り立たない（1時刻1roundでは平均の更新が小さくならない）。design.mdの判断の材料のうち「分割が1回も起きない」に当たる。本規模へ進むか、round数を増やす派生の構成を作るか、比較から外すかはユーザーの判断待ち。台帳`docs/experiments/experiment-log.md`に記録。
