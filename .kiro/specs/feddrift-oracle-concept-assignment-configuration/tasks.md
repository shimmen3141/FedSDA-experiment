# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求2.1・2.2のtestを先に書き、REDを確かめてから、設定の項目、構成、runtimeの分岐（$K$ 個のモデルの登録、標本ごとの振り分け、予測のモデル、統合を行わない）、記録を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3_
- [x] 2. 文書と検証の記録
  - 要求3.1の文書。実装レビュー（独立CLI）と、関係するtestの結果をImplementation Notesに記録する。runtimeを変えるので、全pytestも主担当が流す。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [x] 3. 本規模の実行と集計
  - commit・pushの後、主担当が内部確認をしてコマンドを渡す。結果をFedHASP・FedDrift・FedAvgと対で集計し、実験の台帳に記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes

### Task 1・2の実装（2026-11-01、基点`2f2a7152`、未コミット）

- 実装と文書を追加。独立CLIの起動・commit・push・stash・Task 3の本規模実験は依頼に従い未実施。独立実装レビューと全pytestは主担当が行う。権限で未確認の検証もあるため、checkboxは完了判定前のまま残す。
- RED: 新しい構成を使う3件が`unknown experiment configuration`で失敗することを確認してから実装。接続後、指標の導出が検出記録を同期回数だけ要求してOracleを拒否することを6標本のtestでも再現し、Oracleでは検出記録0件を検査するよう修正した。
- 設定: `FeddriftRunClientSettings.model_assignment_policy`。既定`detection_batch_assignment`、Oracleは`true_concept_assignment`。`experiment_spec/options.py`に`feddrift_model_assignment_policy`を`OptionSpec`・`ActivationRule`で登録（新APIだけの設定であることを明記）。数値パラメータ・新しい掃引軸は追加していない。既定値はfieldのmetadataに従って保存設定から省き、既存の設定辞書とrun識別子を維持する。基点で採取したFedDriftの識別子`sine2_interval50_seed0_1fbf192c89f9`との一致も専用testで確認する。
- 接続: participant factoryがデータのKを渡し、client assemblyとglobal repositoryが共通の事前学習モデルの写しをID 0〜K−1に登録する。stream processorは予測の後だけ次の予測モデルを真概念へ替え、完成バッチを標本の真概念で分けて学習storeへ追加する。serverはclientごとのその時刻の帰属ID集合の和集合を学習対象とし、同期後に集合を空にする。既存の件数条件・R round・累積標本数による集約重みをそのまま使い、prepareではID集合の報告1通/clientだけ数えて検出・作成・統合を実施しない。
- 記録: Oracleだけに`sample_prediction.training_model_id`を追加。FedDriftの保存形式を版3へ上げ、読込みは版1〜3。既存構成の記録表とgoldenは変更しない。構成の専用testを`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ、追加のimportを依存境界へ登録した。
- 文書: `docs/experiments/experiment-sweep.md`、`differences-from-feddrift.md`、`naming.md`を更新。`docs/reference/options.md`は`python -m tools.generate_option_docs`で再生成。
- 固定条件の全体run: SINE2/MNIST4、間隔50、seed 0、10 client、2500標本/client。両方成功し、5000標本への拡大は不要。SINE2は概念変化24回、モデル別の延べ学習標本数はID 0が800,000、ID 1が504,000。MNIST4は概念変化22回、ID順に800,000・78,400・217,600・363,200。各runの全25,000標本で学習先＝当該標本の真概念、予測先＝直前にラベルを見た標本の真概念（最初は0）を照合。各時刻のserver/clientモデル数はK（SINE2は2、MNIST4は4）、保有IDも0〜K−1、初期登録K件のみ、検出・新規作成・統合は0件。Oracleの専用test全11件成功（1155.49秒）。これらは受入検証で、手法の優劣を比較する研究実験ではない。
- 最新の軽い契約検証: スキーマ・構成・識別子・依存境界・設定保存など2,314 passed（重い1件はこの実行では対象外）。Oracleの軽いtestは9 passed。初期モデルの値と参照の独立性、生成でtorch乱数を消費しないこと、不正概念のblock全体の拒否、時刻内の複数帰属、モデルごとの閾値、R round、過去の標本を持ってもその時刻に対象でないモデルを学習しないこと、終端の残りの帰属も確認。
- 既存FedDrift・掃引・保存の最新検証: 1,019 passed、131 error（すべて`PermissionError: [WinError 5]`によるtmp_pathの準備失敗）、4 skipped（Linux CPU専用の照合）。errorの内訳は`test_experiment_sweep.py` 50、`test_experiment_sweep_resumption.py` 31、`test_feddrift_experiment_run_artifacts.py` 37、`test_feddrift_experiment_sweep.py` 13。権限エラーを成功扱いにしていない。
- 基準版1〜16: 最初のgolden全ファイルの実行は380 passed / 19 error（補助testの一時ディレクトリへのアクセス拒否）。数値goldenの206件はすべて成功した。指標のOracle分岐の修正後も、数値golden 206件のassertionがすべて成功（JUnit: tests=206、failures/errors/skipped=0）。ただしpytest終了処理で`numerical-temp`の走査がアクセス拒否となり、コマンドのexitは1。数値の照合成功とコマンドの正常終了は区別する。
- 品質: Ruff check・format check成功。指定の`pyright --pythonpath`はPythonのsite-packagesの探索に失敗し195 error。`pyproject.toml`のinclude/strict/extraPathsを維持し、同じvenvを明示した一時設定で再検査し0 error / 0 warningを確認。`spec_checks.py names`も追加の未登録名なし。
- fresh process: 共用`fresh_process_smoke.py`へOracleの短い全体runを追加。そのOracle部分を新processから実行して成功し、旧実装・test moduleの非importを確認。共用script全体は既存の`TemporaryDirectory`へのアクセス拒否で完走できず、全体の成功とは扱わない。
- 検証ログ/JUnitはworktreeのGit管理外`.validation-oracle/`（`contracts.xml`、`existing-final.xml`、`golden.xml`、`numerical-final.xml`、`oracle.xml`等）。外部venv配下はこのsandboxの書込み範囲外なので使用していない。全pytestは未実施。
- 最終の関係検証は上記の契約2,314件、既存1,019件、数値golden 206件、Oracle 11件で、計3,550件のassertionが成功。既存の131 error・Linux専用4 skippedと、別実行のgolden補助19 errorを成功数へ含めない。既存FedDriftの重い掃引・CLI 3件（tmp_pathを使う）は権限制約が既知なので最終実行では対象外。修正前の重複した長い実行は打ち切り、完了した最新の実行の件数だけを採用した。
- 作業中に別作業のIFCA仕様5ファイルだけがcommitされ、HEADは`77b95583`へ進んだことを確認。本担当はcommitしておらず、Oracleの実装差分は未コミットのまま。

変更ファイル（24件。下記のprefixはそれぞれの箇条書き内で共通）:

- `src/federated_learning_experiments/`: `core/settings_serialization.py`、`methods/fedsda/model_registration/global_model_repository.py`、`sweep/experiment_configurations.py`。
- `src/federated_learning_experiments/runtime/`: `feddrift_run_client_settings.py`、`feddrift_run_client.py`、`feddrift_run_participant_factory.py`、`feddrift_client_stream_processing.py`、`feddrift_detection_batch_processing.py`、`feddrift_run_server.py`、`feddrift_run_metric_derivation.py`、`feddrift_run_record_tables.py`、`feddrift_run_artifact_files.py`。
- `tests/refactoring/`: `test_feddrift_oracle_concept_assignment_configuration.py`（新規）、`test_single_run_dependency_boundaries.py`、`behavior_golden_cases.py`、`fresh_process_smoke.py`、`test_feddrift_experiment_run_artifacts.py`、`test_fedsda_without_warm_started_learner_configuration.py`。
- `federated_drift_experiment/experiment_spec/options.py`。
- `docs/`: `experiments/experiment-sweep.md`、`experiments/differences-from-feddrift.md`、`reference/options.md`（再生成）。
- 対象spec: `naming.md`、`tasks.md`。

### 主担当の確認（2026-11-01）
- 差分の照合: 実装担当の報告の変更ファイル24件が作業ツリーにそろうことを確かめた（作業中に主担当がIFCAのspecの文書をcommitしたとき、commitの前の検査が未commitの変更を退避・復元したが、欠けは無い）。
- 実装レビュー（GPT-6 Luna、effort medium、対象は`77b95583`の上の未commitの差分）: PASS（Critical・Major無し）。
- 全pytest（Windows、12並列、1608秒、JUnit `../../venv/refactoring-tests/oracle-full-pytest.xml`）: 13,256 passed、8 skipped、2 failed、12 errors。errorsの12件は、リポジトリの直下から流したため`results/implementation-audits/fedavg-20261021/test-results.txt`をtestとして集めようとしたUnicodeDecodeErrorで、コードの不具合ではない。failedの2件はtestの期待値の更新漏れで、主担当が直した: (1) `test_fedsda_configuration_variants.py`の構成の一覧と、実旧と照合しない構成の集合に`feddrift-oracle-concept-assignment`を足した。(2) `test_fedsda_warm_started_learner_and_cross_validated_initial_statistics.py`の「全FedSDAの構成がwarm startの学習者を使う」で、`fedsda-without-warm-started-learner`だけは初期化し直した学習者を期待するようにした（この失敗は`2708b3bb`から残っていた。そのspecでは全pytestを流していなかった）。直した2ファイルの再実行は100 passed、12 skipped（JUnit `../../venv/refactoring-tests/oracle-fix-two-tests.xml`）。Ruff成功。

### 本規模の結果（2026-11-02、主担当）
- `results/method-comparison-oracle-v16/20261101_194246_oracle-seeds0-9`（uranium、commit `cc921198`、基準版16）。400 run完了・失敗0。集計は主担当のscratchpadの`oracle_compare.py`（Git管理外）。比べる相手は論文の表1と同じ結果（新しい4データのFedDriftは、調整したδの本評価が届くまで既定のδで仮に比べた）。
- Oracle−FedDrift: 40セルすべてでOracleが上（+1.46〜+37.38pp、平均+6.25）。
- FedHASP−Oracle: −2.78〜+1.30pp、平均−0.72。FedHASPが上のセルは13/40（SINE2の間隔200（+0.04、10 seed中5）、MNIST2の間隔50〜200、MNIST4の全間隔、fmnist2の間隔50・100、fmnist4の間隔50〜200。うち10 seed中8以上で上が12セル）。Oracleが大きく上はSEA2・SEA4（2.1〜2.8pp）、CIRCLE2・rotmnist4の間隔500（2.1〜2.2pp）。
- 通信のOracle/FedHASPは0.74〜2.96（SINE2・CIRCLE2は1未満、4コンセプトの画像データは約2.8〜3.0）。clientのMACのOracle/FedHASPは0.49〜1.46。
