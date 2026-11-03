# Implementation Plan

- [x] 1. 1時刻のround数の設定と、CFLの時刻の途中の分割を足す
  - 要求3のtest（既定の不変、掃引と制約、FedAvg+Window $R=3$ の小さい全体run、CFLの $R=3$ のサーバの経路、CFLの $R=3$ の小さい全体run）を先に書き、REDを確かめてから、掃引の値・設定の登録・組み立ての検査・CFLの分割の後の標本の付け替え・roundごとの対象の集合・記録の列`round_index`（保存形式の版8）を足す。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・Window・Adaptive-FedAvg・DriftSurf・CFLのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_
- [x] 2. 文書と検証の記録
  - 要求4.1の文書（設定の参照の再生成、`differences-from-feddrift.md`のD11とCFL+Windowの節、構成の一覧）とnaming.mdの追記。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、予備の実行（96 run）のコマンドを渡す。design.mdの「予備の実行」の集計をユーザーに示し、本規模へ進むかと $R$ の決め方をユーザーが決める。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2

## Implementation Notes

### Task 1・2の実装（2026-11-03、未コミット）

- 基点は`d734a8b2`、branchは`refactor/architecture`。ユーザーの分担に従い、`kiro-impl`の手動実行・`--review off`相当でTask 1・2を実装。独立レビューCLI、commit・push・stash、Task 3の予備実行は行っていない。独立した実装レビューと全pytestは主担当が担当する。
- RED: 新規testを先に実行し、31 failed・8 passed（40.21秒）。CLI引数・計画の項目の未登録と、Window/CFLのR=3の組み立て拒否で失敗した。既定27構成の保存設定・run識別子のSHA-256を実装前に採取し、testへ固定した。
- 設定: CLIは`--feddrift-local-training-round-counts`、計画は`feddrift_local_training_round_counts`（`None`で`(1,)`）、1 runは既存の`server_settings.local_training_round_count`（既定1）。OptionSpec・ActivationRule・ChoiceConstraint・ParameterSpec・SweepAxisを登録。新APIの軸の宣言は旧SweepPlanの実行へ混ぜない。2窓構成以外を含む計画で既定以外を指定すると拒否する。IFCA・Adaptive-FedAvg・DriftSurfの組み立ては各構成名を示してR≠1を拒否し、既存のFedDrift本体のR>1の経路は維持する。
- runtime: `feddrift_run_server.py`でroundの直前に対象集合を再計算し、集合が変わった場合だけ追加通知を1通/client計上する。集約側は学習後・分割前の同じ割り当てから対象集合を計算する。`cfl_training_round.py`で各roundの学習後・集約前に既存の分割判定を行い、分割時は既存どおり全モデルの集約を省きθ0を配布してから`cl2`の割り当てと標本列を新IDへ移す。移動先の不在は判定状態・記録を更新する前に検査する。件数・step数・帰属診断のstoreは移さない。
- 記録: `_cfl_records`の公式照合用tupleは維持し、同じ順の`_cfl_record_round_indices`を`cfl_split_check.round_index`として保存する。runの保存形式と掃引記録の版は8。ほかの構成の表・列は変えない。
- 文書: `docs/reference/options.md`は`python -m tools.generate_option_docs`で再生成。`differences-from-feddrift.md`のD11・CFL節と`experiment-sweep.md`の構成一覧、naming.mdを更新。
- **設計の前提との不一致（主担当の確認が必要）**: design.mdの「状態を持たないSGD」は既存コードと一致しない。`feddrift_default_run_settings.py`はAdam（AMSGrad、weight decay 0.001）を作り、`experiment_configurations.py`のWindow/CFL分岐はoptimizerを替えない。既存Windowの実体testもAdamを要求する。既定設定不変と既存の局所学習の再利用を守り、optimizerと配布時の作り直しは維持した。そのためR>1では公式の時刻内の状態保持とのD05の差が残ることを文書に明記した。承認済みdesign.mdは変更していない。
- 検証の途中経過: 新規testの軽い範囲40 passed、設定・掃引スキーマと依存境界2,286 passed。Ruff check・format check成功。Pyrightの通常の`--pythonpath`指定はvenvの探索に失敗し202件の依存importエラーとなったが、共有venvを`venvPath`・`venv`で指定した検証用設定（`.validation-rounds/pyrightconfig.json`、既存pyproject.tomlをextends）では0 errors・0 warnings。共有設定・依存関係は変更していない。
- pytest-xdistは一時ディレクトリへのWinError 5で収集前に停止（実行0件）。逐次実行は時間が長いため途中で停止し、最終集計には使わない。通常のPython子processは起動できることを確認し、既存の`tests/conftest.py::select_parallel_test_scope`で同じ分配単位を作り、12 processで関連testを検証中（補助はGit管理外の`.validation-rounds/parallel_checks.py`）。同一ファイルを分割せず、goldenだけ既存と同じケース別の単位にする。検証済みの新規43件・スキーマ29件・依存境界2,257件は除外し、残り1,652件を31組に分けた（合計3,981件）。一時ディレクトリの権限・設定は変更していない。権限エラーは成功扱いにしない。最終の件数と結果を追記するまでTask 1・2のcheckboxは保留する。
- 最終版の新規testは43 passed（1471.05秒、`.validation-rounds/new-final.xml`）。分割直後の全clientのθ0と次時刻の標本の不在を補強した後の軽い範囲も40 passed（10.56秒、`.validation-rounds/fast.xml`）。CFL R=3の小さい全体runはSINE2・MNIST4とも分割0回、時刻`[]`・round`[]`。決定的なサーバ経路では時刻0・round 0で1回分割し、対象集合は`{0}`→`{0,1}`、残り2 roundで標本を移した新IDの件数重み付き集約を確認した。4 client・2時刻×3 roundの下り通知の累計は`[4,8,8,12,12,12]`。FedAvg+Window R=3は学習呼出し1,500回（50時刻×10 client×3）、全roundで同じ窓を再利用し、上り・下りのモデル転送数・パラメータ値数・byteはすべてR=1の3倍、通知の件数は不変。
- スキーマ・掃引・依存境界の検査は2,286 passed（5.61秒、`.validation-rounds/schema-boundaries.xml`）。宣言の並びを既存のtraining群へそろえて文書を再生成した後のスキーマ3ファイルも29 passed（3.33秒、`.validation-rounds/schema-final.xml`）。Ruff check・format check、Pyright（上記のvenv明示設定）、`pip check`、`git diff --check`が成功。`spec_checks.py names --base d734a8b2`は公開名の未登録なし。

変更ファイル（この実装分）:

- `src/federated_learning_experiments/runtime/`: `cfl_training_round.py`、`feddrift_run_artifact_files.py`、`feddrift_run_participant_factory.py`、`feddrift_run_record_tables.py`、`feddrift_run_server.py`。
- `src/federated_learning_experiments/sweep/`: `experiment_sweep_command.py`、`experiment_sweep_plan.py`、`experiment_sweep_records.py`。
- `federated_drift_experiment/experiment_spec/`: `options.py`、`parameters.py`、`sweep.py`（宣言のみ。旧学習経路は変更しない）。
- `tests/refactoring/`: 新規`test_feddrift_window_multiple_training_rounds.py`、既存の形式版の期待値だけを更新した`test_experiment_sweep.py`・`test_feddrift_experiment_run_artifacts.py`。設定宣言の照合を追加した`tests/test_parameter_schema.py`。
- `docs/experiments/differences-from-feddrift.md`、`docs/experiments/experiment-sweep.md`、再生成した`docs/reference/options.md`。
- 対象specの`naming.md`、`tasks.md`、`spec.json`（実装の状態を追記。承認・仕様レビューの履歴は維持）。

### 最終の関連検証（2026-11-03）

- 重複を除いた対象は**3,981件: 3,852 passed・125 errors・4 skipped、assertion failure 0**。集計は`.validation-rounds/final.xml`、内訳は`.validation-rounds/summary.json`・`summary.md`。新規43件と軽い範囲40件、スキーマ29件、依存境界2,257件、通常の別processによる残り1,652件のJUnitを、同じtestを二重に数えず統合した。TDDのRED、途中の再実行、停止した逐次実行はこの件数に加えていない。
- 125 errorsはすべて一時ディレクトリのセットアップでの`PermissionError: [WinError 5]`。内訳は掃引53件、FedDrift掃引16件、保存・読込37件、golden共通検査19件。test本体を実行できておらず、成功とは扱わない。4 skippedは固定Linux CPU環境専用の検査（Windowsでは対象外）。権限設定の変更・sandbox解除は行っていない。
- 基準版1〜16の**数値golden 206件がすべて成功**。既存の対象ケースの選別をそのまま使い、版1〜6は各5件、版7〜10は各17件、版11〜16は各18件。goldenファイルは変更していない。goldenのtestファイル全体は380 passed・19 errors（上記の共通検査）。`tests/test_regression.py`も1 passed。
- 既存CFL 40件、IFCA 15件、Oracle 11件、Window/Adaptive-FedAvg 30件、DriftSurf 26件はすべて成功。新規R=3の全体run・決定的分割経路の結果は上記のとおり。
- Task 1・2は実装・文書更新済みだが、未実施125件、optimizerの設計前提の確認、主担当による独立した実装レビュー・全pytestを残すため、checkboxとspecの完了扱いは保留する。Task 3は未着手。commit・push・stashは行っていない。
- 最終のRuff check・format check（396ファイル）、Pyright（venvを明示した検証用設定）は成功。`spec_checks.py names`は未登録なし、`progress`はphase implementation・0/3で整合。`identity --junit .validation-rounds/final.xml`はNGであり、成功扱いにしない（`.validation-rounds/identity.log`）。理由は固定旧748c3aaとの既存のcommit差分と作業ツリー差分、ユーザー指定どおりの未コミット状態、JUnitの125 errors、今回の対象に含めていない`tests/test_proposed_regression.py`。旧の実行経路は変更せず、今回の旧パッケージの変更は要求された宣言3ファイルだけである。HEADは`d734a8b2`、branchは`refactor/architecture`、stageされた差分は空。

| testファイル | passed | errors | skipped |
|---|---:|---:|---:|
| `test_experiment_sweep.py` | 27 | 53 | 0 |
| `test_feddrift_cached_cluster_consolidation.py` | 90 | 0 | 0 |
| `test_feddrift_cfl_window_configuration.py` | 40 | 0 | 0 |
| `test_feddrift_client_round_training.py` | 63 | 0 | 0 |
| `test_feddrift_client_stream_processing.py` | 56 | 0 | 0 |
| `test_feddrift_computation_cost_summary.py` | 61 | 0 | 0 |
| `test_feddrift_detection_batch_assessment.py` | 62 | 0 | 0 |
| `test_feddrift_detection_batch_processing.py` | 75 | 0 | 0 |
| `test_feddrift_driftsurf_configuration.py` | 26 | 0 | 0 |
| `test_feddrift_experiment_run_artifacts.py` | 20 | 37 | 0 |
| `test_feddrift_experiment_sweep.py` | 30 | 16 | 0 |
| `test_feddrift_ifca_window_configuration.py` | 15 | 0 | 0 |
| `test_feddrift_linux_core_metrics.py` | 0 | 0 | 4 |
| `test_feddrift_measured_run_execution.py` | 34 | 0 | 0 |
| `test_feddrift_model_diagnostic_metrics.py` | 59 | 0 | 0 |
| `test_feddrift_official_algorithm_alignment.py` | 29 | 0 | 0 |
| `test_feddrift_oracle_concept_assignment_configuration.py` | 11 | 0 | 0 |
| `test_feddrift_run_client.py` | 55 | 0 | 0 |
| `test_feddrift_run_metric_derivation.py` | 34 | 0 | 0 |
| `test_feddrift_run_participant_factory.py` | 43 | 0 | 0 |
| `test_feddrift_server_synchronization.py` | 149 | 0 | 0 |
| `test_feddrift_stream_finalization.py` | 89 | 0 | 0 |
| `test_feddrift_stream_run_execution.py` | 44 | 0 | 0 |
| `test_feddrift_window_and_adaptive_fedavg_configuration.py` | 30 | 0 | 0 |
| `test_feddrift_window_multiple_training_rounds.py` | 43 | 0 | 0 |
| `test_important_behavior_regression_goldens.py` | 380 | 19 | 0 |
| `test_option_schema.py` | 21 | 0 | 0 |
| `test_parameter_schema.py` | 3 | 0 | 0 |
| `test_regression.py` | 1 | 0 | 0 |
| `test_single_run_dependency_boundaries.py` | 2,257 | 0 | 0 |
| `test_sweep_plan.py` | 5 | 0 | 0 |

### 主担当の確認（2026-11-03）

- optimizerの前提: design.mdの「状態を持たないSGD」は誤り。既定はAdam（AMSGrad、weight decay 0.001）で、公式の既定（`main_fedavg.py`の`--client_optimizer adam`・`--wd 0.001`、`FedAvgEnsTrainerClusterFL.py`の`amsgrad=True`）と同じ。公式はモデルごとのAdamを初めに1度作り時刻内のroundで状態を保つが、手元は配布のたびに作り直す（D05）。全手法に共通の学習規則で、同じRのFedAvg+Windowにも同じく生じるので許容し、design.mdとdifferences-from-feddrift.mdの該当文を直した。
- 実装レビュー: GPT-6 Luna（medium、read-only）PASS、Major以上なし（`results/_sol_prompts/rounds-impl-review.out`）。Lunaのpytestは一時ディレクトリの権限で収集前に停止し、独立には再現していない。Ruff checkは独立に成功。
- 全pytest（主担当、Windows、`-n 8 --dist loadfile`、2,442秒、`.validation-rounds/full-pytest-claude.xml`）: 13,417 passed、8 skipped、4 failed、8 errors。4 failedは版7→8・CLI引数・計画の項目の追加を既存testの期待値が反映していなかったもの（実装担当の環境では一時ディレクトリの権限で未実行だった）で、期待値だけを直し、該当4ファイルを再実行して110 passed（`.validation-rounds/rerun-4files.xml`）。8 errorsは対象`tests`を付け忘れ、Git管理外の`results/implementation-audits/fedavg-20261021/test-results.txt`がdoctestとして集められたもので、testではない。全体の再実行はしていない。
- 予備の実行の計画（内部確認）: 96 run、既存の結果との重複なし。
