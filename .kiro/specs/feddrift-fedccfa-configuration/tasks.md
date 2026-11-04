# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求3.1〜3.3のtestを先に書き、REDを確かめてから、設定の項目（割り当ての方式・窓・整合の開始の新しい値、$\epsilon$・$\gamma$・温度のパラメータと $\epsilon$ の掃引の軸）、4構成、runtimeの分岐（表現部の共有と個人の分類層、診断用の分類層、クラスごとのMADDとDBSCAN、組ごとの平均、アンカーと対照の整合、欠けたクラスの規則、2クラスの2出力の分類層）、round数の掃引の値の許可、通信と計算量の計数、記録の表（保存形式の版を上げる）を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。公式の式の抜粋`tests/refactoring/fedccfa_official_excerpt.py`（MIT Licenseの表記つき）を作る。
  - 完了条件: 新しいtest（公式の式との照合、元の条件での挙動の確認、小さい全体run）、構成・掃引・識別子・依存境界・スキーマのtest、FedEM・IFCA・Window・CFL・DriftSurf・FedDrift・Oracleのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`が通る。Ruff・Pyright。新しい全体runのtestの所要時間を記録し、重いものは`slow`の印と`tests/conftest.py`の配り方へ登録する。
  - _Requirements: 1.1〜1.10, 2.1, 3.1〜3.4_
- [x] 2. 文書と検証の記録
  - 要求4.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）とnaming.mdの追記。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、予備の実行（192 run）のコマンドを渡す。design.mdの集計と選ぶ手順の結果をユーザーに示す。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2
- [ ] 4. $\epsilon$ の調整の実行と本規模の実行
  - 選んだ構成と $R$ で $\epsilon$ の調整の実行（480 run）、続いて本規模の実行（400 run）のコマンドを渡し、集計する。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 3

## Implementation Notes


### 2026-11-05 Task 1・2の実装（主担当の確認・独立レビュー待ち）

- 基点は`refactor/architecture`の`743b6600`。実装担当が直接調査・実装した。レビューCLI・内蔵子エージェント・commit・push・stash・予備の実行は行っていない。既存の承認記録とgoldenは変更していない。
- **Task 1の受入は未了**。公式の実コードをそのまま動かした照合が、アンカーの参照共有による副作用のため不合格（下記）。設計の等重み平均を維持し、公式の副作用は移植していない。Task 2の文書・命名・設定参照は作成済みだが、Task 1の受入未了のため両方のcheckboxを保留する。仕様の承認を実装の承認として扱わない。
- 通常のpytestの残る失敗は無い。ただし、権限で未確認のtestを含むため全検証完了とはしない。全pytestと独立した実装レビューは依頼どおり主担当が行う。

#### REDと実装の内容

公式の式・構成・整合の開始のtestを先に書き、未実装moduleの`ModuleNotFoundError`によるRED（収集エラー1、5.11秒）を確認してから実装した。配り方のtestも、parametrizeの無い重いtestがファイルの組へ残るRED（1 failed、3.46秒）を確認してから`tests/conftest.py`を修正した。

4構成は既存31構成の末尾へ、`feddrift-fedccfa-window`、`feddrift-fedccfa-window-scaled-alignment`、`feddrift-fedccfa-window200`、`feddrift-fedccfa-window200-scaled-alignment`の順に追加した。設定は`model_assignment_policy=fedccfa_classifier_clustering`、`training_sample_window=current_timestep/recent_observed_samples`、`fedccfa_alignment_start=fixed_round_20/scaled_to_run_rounds`、`fedccfa_dbscan_eps=0.1`、`fedccfa_alignment_gamma=20.0`、`fedccfa_alignment_temperature=0.1`。新しいfieldの既定値を保存から省く規約で、既存構成の保存設定を維持する。

`OptionSpec`・`ActivationRule`・`ChoiceConstraint`・`ParameterSpec`へ登録した。`FEDCCFA_DBSCAN_EPS_AXIS`を`SweepAxis`として宣言し、新APIの計画とCLIに`--fedccfa-dbscan-eps`を接続した。Rの軸も4構成で許可する。ε×Rの16 runの計画と識別子の区別をtestで確認した。`python -m tools.generate_option_docs`を実行済み。

HEADの構成生成関数を読み取りだけで別namespaceへロードし、同じ条件で既存31構成の順・保存設定・識別子を照合して一致した。その値の固定hashを新しいtestへ保存した（SHA256 `ec7f74c1415a33c41ed328e8e461c48e00bd75ccc6b7b0e787d14ecc48d177c0`）。既存の27構成のhashのtestも維持する。

| 部分 | 実装の場所 |
|---|---|
| 個人の分類層、初期の分類層、診断、H/γ、均衡抽出、対照の整合、局所アンカー | `runtime/fedccfa_classifier_clustering.py`の`FedccfaClientState`・`train_round` |
| MADD・DBSCAN・入力を変更しない平均 | 同moduleの`_madd`・`_label_groups`・`_weighted_mean` |
| 2クラスの2出力への初期値変換 | 同moduleの`_convert_binary_classifier`。`ModelArchitectureClassifier.binary_softmax_output`がsoftmaxのクラス1確率を共通の二値予測APIへ返す。独立モデルの構造検査も対応 |
| ラベル前の予測→観測直後の窓への追加、時刻先頭のwindowの消去 | `feddrift_client_stream_processing.py`。学習storeへの接続はflushで行う。集約の間の学習は無い |
| 通算roundの整合の開始 | `_alignment_start_round`と`feddrift_run_participant_factory.py`。clientの通算番号と時刻内round番号を分ける |
| 段ごと・roundごとのAdam（AMSGrad、weight decay 0.001）、分類層の学習率10倍、予算 | `FedccfaClientState.train_round`と`feddrift_run_client.py`。窓が通常の学習バッチ未満なら個人・表現部の更新を行わず、診断は均衡の全体で5更新 |
| 表現部の件数重み平均、クラスの組ごとの個人の行とアンカーの等重み平均 | `runtime/fedccfa_training_round.py`。グローバルなモデルは0だけ、サーバの初期の分類層は更新しない |
| 通信 | `feddrift_run_server.py`の初回配布と`fedccfa_training_round.py`の部品別の毎roundの転送。時刻境界の通知はしない。欠けたクラスのアンカーをアップロードしない。単独・欠けたクラスへの行・アンカーの配布は数えない |
| 計算量 | 既存のmodule/optimizer hookで診断・個人・表現部・アンカーを計測。学習件数は診断の実際の均衡件数と、個人・表現部のバッチ件数を分けて数える。サーバの表現部・行・アンカーの平均MACも加算 |
| 記録・保存形式 | `feddrift_run_record_tables.py`に`fedccfa_label_cluster`・`fedccfa_round`・`fedccfa_communication`。FedDriftのrun形式と掃引記録の形式を9→10へ上げた |
| 期待値の固定・保護 | `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界に登録。構成一覧・除外集合、版、CLI・計画の固定を持つ既存testを読み、必要な期待値を更新 |

欠けたクラスの単体testは固定した窓で、存在するクラスだけの均衡抽出・H、アンカー保持、未保有のアンカーのロジット除外、欠けたクラスのclientの組からの除外と行の保持を確認する。仕様に明記されていない境界は暫定的に次のように実装したため、主担当の確認が要る: (1) MADDの参加者が2人以下なら比較相手が無いため距離0。(2) ゼロベクトルのcos類似度は0。(3) 初めて現れたクラスに直前のアンカーが無ければ、その標本をアンカー取得まで対照損失から除く。(4) `alignment_active`は閾値の後でもH=0、または現在の窓のラベルに対応する直前のアンカーが無い場合はfalse（整合を実行しない）。元の条件での確認ではどの窓もH>0で、この境界の影響は無い。

小さい全体runは3 client・250標本・間隔50。3 clientはMADDの比較相手を持てる最小で、5時刻はwindow200が古い標本を捨てる境界を含む最小。SINE2/MNIST4×window/window200-scaledの4条件に、SINE2の両方の窓のR=3を加えた6条件。検査(a)〜(i)に加え、時刻内のパラメータ不変、サーバの初期の分類層の不変、部品別の通信個数、実optimizerの更新数を確認する。関数全体で20秒を超えるためslowにし、各ケースを別processへ配る。MNISTの挙動の確認もslow・単独のprocessへ配る。

#### 公式の実装との照合（**不合格**）

追跡しないscriptは[official_round_equivalence.py](../../../.validation-fedccfa/official_round_equivalence.py)。公式cloneの`entities.FedCCFA`をimportし、実メソッドを呼ぶ。公式cloneは変更していない。固定venvにTensorBoard・pandas・torchvisionが無いため、使用しないログ・CSV・画像datasetのimportだけをスタブ化した（呼ばれたら失敗）。`ClientDataset`と学習・集約・MADD・DBSCANは公式の実コード。初期モデルの構造は8→16→10、3 client、client2だけクラス1・2を交換する。クラスごとの固定特徴（seed12）と小さいノイズにより、1・2ではclient0/1の組と単独client2を実際に作る。

Aは各クラス5件、Bは`2+((c+i)%5)`件。Bの均衡抽出は公式の`np.random.choice`で選んだ添字を手元にも注入し、各クラス2件と抽出した添字を照合。全段1ミニバッチ、更新(診断5,個人1,表現5)、SGD(momentum0.9,weight decay1e-5)、表現部0.01/分類層0.1。これは`train_round`の照合専用引数で、評価runの設定・CLIから指定できない。初回の空のp_clf_paramsの扱いもこの入口だけ公式に合わせる。round0〜2を整合開始0/20の両方で実行。集約は手元の**実際の**`aggregate_fedccfa_training_round`を呼び、別の集約式を照合対象にしていない。

| 入力 | 整合開始 | 全体の最大絶対差（アンカー） | 診断の最大差 | 集約前の表現部の最大差 | 個人の分類層（集約前/後の大きい方） | 局所アンカーの最大差 | 組・均衡抽出 |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 0 | 4.7949853 | 1.96084e-4 | 1.96576e-4 | 7.27177e-6 | 4.43733e-3 | 一致 |
| A | 20 | 4.7949853 | 1.19209e-7 | 2.98023e-8 | 8.94070e-8 | 9.53674e-7 | 一致 |
| B | 0 | 4.8386865 | 2.63140e-4 | 2.67036e-4 | 3.09050e-5 | 5.02491e-3 | 一致 |
| B | 20 | 4.8386865 | 8.94070e-8 | 5.96046e-8 | 8.94070e-8 | 9.53674e-7 | 一致 |

各条件とも、最初の集約の前は全段で最大9.53674e-7以下。全roundで、クラス1・2は組{0,1}と単独client2。単独の行・アンカーの比較も各条件6箇所で行った。最終の全値・組・比較結果は[official_round_results.json](../../../.validation-fedccfa/official_round_results.json)。rawのscriptは終了コード1で、許容差1e-5に対して4条件すべて不合格。

原因は公式の`aggregate_protos`（299〜332行）と`aggregate_label_protos`（448〜476行）のTensor参照共有。前者は最初のclientのlocal_protosへの参照で足し合わせ、後者も先頭のlocal_protosをin-placeで変える。公式メソッドへアンカー[1,2,3]を渡した直接確認では、最初の集約後のlocal_protosが[6,2,3]となり、組平均は期待2に対して3.6666667だった。開始20では整合前なので表現部・個人の層が一致し、開始0では前roundの異なるアンカーが表現部の学習へ伝わる。

原因の対照として同scriptの`--isolate-alias`では、その参照共有だけをコピーで隔離する。この対照はA/B×開始0/20の4条件すべてが最大9.53674e-7、組・均衡抽出・単独の行/アンカーも一致した（[isolated_alias_results.json](../../../.validation-fedccfa/isolated_alias_results.json)）。**この対照の成功で、rawの公式照合を成功扱いにはしない。** 公式の意図した平均を優先するか、副作用まで再現するかを主担当が判断する必要がある。design.mdは書き換えていない。

公式の式だけのpytestはMIT License・著作権表記と関数ごとの公式ファイル/行を持つ`tests/refactoring/fedccfa_official_excerpt.py`を参照する。MADD・DBSCANの組・個人の行/アンカーの等重み平均・表現部の件数平均・H/γ・対照CE・均衡件数を許容差1e-6で照合し成功。

#### 元の条件での挙動の確認（成功）

MNIST4、10 client、600標本、間隔50、R=3、window200-scaled、ε0.1、seed0。A=client0〜4は概念0のまま、B=5〜9は標本200から概念1で固定。`generate_random_client_concept_traces`の入口だけをtest内で差し替える。予測は全体runの記録を読み、学習は実装した通常の経路。

時刻9〜11の最後のroundの組は次のとおり。U={0,1,2,3,4,5,6,7,8,9}、A={0,1,2,3,4}、B={5,6,7,8,9}。単独のclientは組に含めない。

| 時刻 | クラス0 | クラス1 | クラス2 | クラス3 | クラス4〜9（各クラス） |
|---|---|---|---|---|---|
| 9 | U | A、{5,7,8,9} | {0,1,2}、{3,4}、B | U | U |
| 10 | U | A、B | A、B | U | U |
| 11 | U | A、B | A、B | {0,1,2,4,5,6,7,8,9} | U |

(1) クラス1・2で群をまたぐ組が無く両群内に2人以上の組がある、(2) 非交換8クラスすべてで群をまたぐ組がある、(3) 群Bの数字1・2の精度が時刻4〜5の3.092784%→時刻9〜11の94.193548%（+91.100764ポイント、要求30以上）、(4) 時刻1〜3のクラス1・2に両群をまたぐ組がある、のすべてを満たした。閾値や条件は緩めていない。初回150.22秒、再確認164.55秒（ほかの回帰testも同時に実行していた）。

#### 実行した検証と所要時間

環境は指定の`../../venv/Scripts/python.exe`、`PYTHONPATH=src;.`、OMP/MKL/OPENBLAS各1thread、`FDE_MNIST_DATA_DIR=C:\Users\yshin\Research\FedSDA-experiment\data\mnist`。pytestのcacheは無効。以下は最終の成功結果で、失敗した途中の試行を合計へ水増ししない。

| 対象 | 結果 | 所要時間 |
|---|---|---:|
| 新FedCCFA test（挙動の確認・6つの小さいrunを含む） | 38 passed。追加した配り方の1 testも別途成功し、新testは計39件 | 196.79秒＋配り方1件3.37秒 |
| 最後の単体・境界（新test32＋依存境界2266） | 2298 passed、slow7件はこの実行から除外 | 5.79秒 |
| 最後の6つの小さい全体run | 6 passed（以下に個別の秒数） | 26.29秒 |
| 関係する通常のtest（全feddrift、構成/掃引/識別子、スキーマ、FIFO/T軸） | 1236 passed、Linux専用4 skipped | 55.12秒 |
| 既存の重い全体run等（下表） | 30 passed | 491.81秒 |
| 基準版1〜16のgoldenのファイル全体＋`tests/test_regression.py` | 381 passed、tmp fixtureを持つ19件は権限のため未実施 | 499.83秒（旧回帰の1 testは80.07秒） |
| run設定の検証とモデル構造の既存test | 1243 passed | 2.36秒 |
| 構成一覧/除外集合と構成の配り方 | 2 passed | 2.91秒 |
| 配り方の既存回帰（FedEM・構成・golden） | 3 passed | 2.90秒 |
| Ruff check / format --check（設定範囲） | 成功、402ファイル整形済み | 実行済み |
| Pyright（設定と同じ範囲・strict対象） | 226ファイル、エラー0・警告0 | 5.34秒 |
| `spec_checks.py names` | 新runtimeの公開5名、未登録無し | 実行済み |
| `spec_checks.py progress` | phase implementation、0/4の保留と整合 | 実行済み |

Ruff・Pyrightの対象スコープは変更していない。通常の`pyright --pythonpath ../../venv/Scripts/python.exe`はこの実行環境でsite-packagesを探索できず、numpy/torch等のimportが未解決になった。`pyproject.toml`の同じinclude/strict/extraPathsを読み取ってパスを検証用の設定の場所に合わせ、既存venvを`venvPath/venv`で明示した`.validation-fedccfa/pyrightconfig.json`で確認した。リポジトリの設定は変えていない。

小さいrunの最後の実測: SINE2/window/R1 1.76秒、MNIST4/window/R1 7.76秒、SINE2/window200-scaled/R1 1.40秒、MNIST4/window200-scaled/R1 8.46秒、SINE2/window/R3 1.95秒、SINE2/window200-scaled/R3 2.86秒。

重い関係testのファイル別の成功数と呼出し時間の合計:

| ファイル（test_feddrift_の後） | 成功数 | 呼出し秒数 |
|---|---:|---:|
| cfl_window_configuration.py | 2 | 31.48 |
| driftsurf_configuration.py | 3 | 73.37 |
| fedccfa_configuration.py | 6 | 26.75 |
| fedem_configuration.py | 6 | 42.79 |
| ifca_window_configuration.py | 4 | 83.29 |
| oracle_concept_assignment_configuration.py | 2 | 69.78 |
| window_and_adaptive_fedavg_configuration.py | 4 | 65.67 |
| window_multiple_training_rounds.py | 3 | 95.80 |

pytestの成功した異なるnodeは**5179件**（再実行・重複を除く）、Linux専用skipは4件。個別の全結果・時間は[related-final-results.json](../../../.validation-fedccfa/related-final-results.json)、[slow-results.json](../../../.validation-fedccfa/slow-results.json)、[golden-results.json](../../../.validation-fedccfa/golden-results.json)、[settings-results.json](../../../.validation-fedccfa/settings-results.json)と、新test/境界のJUnit（`.validation-fedccfa/new-final.xml`・`boundary-unit-final.xml`・`small-final.xml`）へ保存した。

権限について: 最初の関係testの試行では**104件がsetupでPermissionError（WinError 5）**となり、pytest終了時のtmpディレクトリの走査も拒否された（[related.xml](../../../.validation-fedccfa/related.xml)、`.validation-fedccfa/related.txt`）。既知の同じ失敗を繰り返さないため、その後はtmp_path/tmp_path_factory等に依存する関係test110件（slowの6件も含む）とgoldenファイルの19件、計**129件を未確認**として外した。権限変更・sandbox解除は行っていない。129件を成功・環境skipとして数えていない。除外したnodeは上記JSONの`excluded_permission_fixtures`に全て記録した。新しいFedCCFAの39件にこの除外は無い。要約/NPZ/CSVの保存・読込みのtestの一部もこの未確認に含まれる。版・CLI・計画・構成の固定を持つ既存testは実行可否にかかわらず読んで期待値を修正した。

#### 残る確認

1. 公式コードのアンカーの参照共有の副作用を再現するか、承認済み設計の入力を変更しない平均を維持して公式との差として扱うか。現在は後者の実装を保持し、rawの公式照合はFAILと記録する。
2. 上記の仕様に明記されていない少数参加・ゼロベクトル・初めてのクラスの境界の判断。
3. 権限により未確認の129件、主担当による独立レビューと全pytest。Task 3・4の実験には進んでいない。

#### 変更したファイル

- `.kiro/specs/feddrift-fedccfa-configuration/naming.md`
- `.kiro/specs/feddrift-fedccfa-configuration/tasks.md`
- `docs/experiments/differences-from-feddrift.md`
- `docs/experiments/experiment-sweep.md`
- `docs/reference/options.md`
- `federated_drift_experiment/experiment_spec/options.py`
- `federated_drift_experiment/experiment_spec/parameters.py`
- `federated_drift_experiment/experiment_spec/sweep.py`
- `src/federated_learning_experiments/learning/models/model_architecture_classifier.py`
- `src/federated_learning_experiments/learning/training/independent_model_parameter_update.py`
- `src/federated_learning_experiments/runtime/fedccfa_classifier_clustering.py`
- `src/federated_learning_experiments/runtime/fedccfa_training_round.py`
- `src/federated_learning_experiments/runtime/feddrift_client_stream_processing.py`
- `src/federated_learning_experiments/runtime/feddrift_run_artifact_files.py`
- `src/federated_learning_experiments/runtime/feddrift_run_client.py`
- `src/federated_learning_experiments/runtime/feddrift_run_client_settings.py`
- `src/federated_learning_experiments/runtime/feddrift_run_participant_factory.py`
- `src/federated_learning_experiments/runtime/feddrift_run_record_tables.py`
- `src/federated_learning_experiments/runtime/feddrift_run_server.py`
- `src/federated_learning_experiments/sweep/experiment_configurations.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_command.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_plan.py`
- `src/federated_learning_experiments/sweep/experiment_sweep_records.py`
- `tests/conftest.py`
- `tests/refactoring/behavior_golden_cases.py`
- `tests/refactoring/fedccfa_official_excerpt.py`
- `tests/refactoring/test_experiment_sweep.py`
- `tests/refactoring/test_feddrift_experiment_run_artifacts.py`
- `tests/refactoring/test_feddrift_fedccfa_configuration.py`
- `tests/refactoring/test_feddrift_window_multiple_training_rounds.py`
- `tests/refactoring/test_fedsda_configuration_variants.py`
- `tests/refactoring/test_fedsda_fixed_share_time_scale_axis.py`
- `tests/refactoring/test_fedsda_pending_assignment_fifo_capacity_axis.py`
- `tests/refactoring/test_single_run_dependency_boundaries.py`
- `tests/test_parameter_schema.py`

検証専用でGit管理外: `.validation-fedccfa/official_round_equivalence.py`、`run_checks.py`、`pyrightconfig.json`、各JSON/XML/log。既存のコミット保留資料・resultsは変更・stageしていない。

### 主担当の確認（2026-11-05）

- 実装担当（GPT-6.1-sol）の記録を読み、公式の`aggregate_protos`（`entities/FedCCFA.py:300-309`）・`aggregate_label_protos`（453-462行）の参照共有と、`train_with_protos`の`global_protos`（157行）が同じTensorを参照することを公式のコードで確かめた。論文の記述（組の中の等重みの平均）に従い、副作用は再現しない（design.mdの「実装で決めたこと」と違い(j)）。参照共有だけを隔離した照合を主担当が再実行し、(A)(B)×整合の開始0・20の4条件すべてで最大の差9.54e-7、`passed=true`を確かめた。隔離しない照合の不一致は、この副作用から来る。
- 実装担当が決めた境界の4点（2 client以下のMADDの距離0、ゼロのベクトルのcos 0、アンカーの無い初めてのクラスの除外、整合の対象が無いroundの扱い）を確かめ、design.mdへ記録した。
- 新しいtestファイル: 39 passed（主担当のWindows、6 workers、167.7秒。最も重い`test_original_mnist_behavior`は152秒）。
- 実装レビュー（GPT-6 Luna medium）: 1回目FAIL（Major 2: 参照共有を隔離した対照が公式と一致しない、予測の時点とround内の順序のtestが無い）。どちらも事実の読み違い（隔離した対照は一致、`test_small_full_run`の`inspect_predict`・`stage_trace`・`completed_stage_updates`が該当）なので、根拠のファイルを示して2回目を依頼し、PASS（指摘なし）。Lunaは一時ディレクトリの権限でpytestを実行していない。
- 全pytest（主担当、Windows、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`-n 8 --dist loadfile tests`、1,049秒、`.validation-fedccfa/full-pytest-claude.xml`）: 13,492 passed、8 skipped、1 failed。失敗は`test_fedsda_default_waited_registration_and_alarm_wait.py`のCLIの引数の一覧の期待値に`--fedccfa-dbscan-eps`が無かったもの（実装担当の更新漏れ）。期待値だけを直し、該当ファイルを再実行して3 passed。全体の再実行はしていない。実装担当の環境で権限のため未確認だった129件は、この全pytestで実行された（skipは従来と同じ8件）。
