# Implementation Plan

2026-11-07のユーザー指示によりTask 1〜6を実装・検査中。D1・D2は案Aで確定、仕様書群の承認はspec.jsonの記録が正。Task 1・2・3・5は対象検査まで完了、Task 4・6は未達の検査・独立CLI承認を残す。Task 7の本規模実行と実集計は主担当。commit・push・stashは行わず、差分を作業ツリーに残す。経緯と今回の証拠はImplementation Notesを参照。

- [x] 1. 学習用重みの方針を設定として宣言する
  - 案Aのユーザー確認と仕様承認が前提。変更前HEADからdesign「既存構成とgoldenの固定基準」の順序・保存mapping全体・hash・識別子を取得し、新fixture案`tests/refactoring/soft_training_single_model_prediction_existing_configurations.json`へ基点と条件付きで固定する。golden／乱数fixtureのパスとSHA256、旧3構成の数値と入力系列の固定資料も変更前に取得する。取得後に既存の設定受理・拒否を固定するtestを置き、新方針の型・宣言・依存制約と、既定値の保存省略を足す。
  - 完了条件: 新しい組だけが受理され、soft＋AdaHedge・逐次更新・不正な再較正等の契約外の組は拒否される。既存設定のシリアライズは変化しない。
  - 完了条件の証拠: 変更前の全構成（既存soft・診断構成を含む）をfixtureが覆い、実装後のbuilderから期待値を再取得していない。既存`soft_training_existing_configuration_hashes.json`の値・既存拒否条件も保持する。fixture取得条件とコマンドをNotesへ記録する。
  - _Boundary: 構成と設定_
  - _Requirements: 1.1, 2.3_

- [x] 2. 学習用重みの更新と再生を独立して提供する
  - 専用Fixed-Shareと同期写しを受け、観測時の同期損失による更新、保留標本の局所損失による再生を、既存の計算を再利用して提供する。モデル集合変更、候補追加、空の保留を扱う。
  - 完了条件: 固定入力で既存softの重みの更新と再較正に数値が一致し、実予測用owner・固定済みq・モデルの値・optimizer・RNGは変化しない。追加の順伝播が既存の費用計測へ入る。
  - 完了条件の証拠（design S1・S4）: 局所と同期が異なる複数モデルを使い、既存softの観測関数と同期損失列・更新前後の重みを複数標本で照合する。集合変更・候補追加・登録ID・統合IDを含める。既存再較正の実関数と、保留の局所損失列・再生順・件数・ID・前後のFixed-Share状態を照合する。空保留／1モデルも検査し、実予測owner・固定済みq・モデル値・grad・optimizerが不変であることを確認する。
  - 完了条件の証拠（design S5・S6）: 初期化済み状態を基準に、run所有Random.getstate・RandomState.get_state、torch CPU get_rng_state、global Python／NumPyの状態を補助評価・写し取得・再生の各前後で完全照合する。共有部／概念固有部の推論件数と推論MAC、局所／同期内訳と総計の増分を実計測器で確認し、training forward・backward・optimizer stepの増分0、経過時間scope内、写しstorage・保持byte数、追加送受信の増分0を記録する。メモリ実測未実施はN/Aとする。
  - _Boundary: 学習用重み更新_
  - _Requirements: 2.1, 2.2, 2.4, 3.2_

- [x] 3. 新構成を単体予測と柔らかい学習へ接続する
  - clientが専用ownerを生成・保持し、実予測後・FIFO解放前の重み更新、配布後の写し取得と学習用再較正、責任度providerへの注入を接続する。新構成だけを一覧末尾へ足す。
  - 実予測の単体経路と戻り値は維持し、補助評価の全モデル損失を監視や実予測の記録へ流さない。ID変更時は既存ownerとストアの対応付けを引き継ぐ。
  - 完了条件: SINE2・MNIST4の小さい全体runで単体予測・複数モデルの柔らかい更新・警報と候補が発火し、責任度と集約重みが既存softの規則に従う。未来のラベル使用・二重更新・余分なRNG消費がない。
  - 完了条件の証拠（design S1・S2）: 同一モデル・current ID・統計・検出器状態・入力列で4構成の監視用局所損失列、baseline、log-e・警報を比較する。実予測はsingle対照と一致させる。ラベル利用前の予測確定→ラベル開示→同期損失による学習用更新→局所損失監視→解放→局所確率によるq固定→共同更新、の順をspyで確認する。入力recordがラベルを既に持つ現APIでも予測確定に未来のラベルを利用しない。
  - 完了条件の証拠（design S2・S3・S4）: 同一状態と同一解放入力から既存soft providerの局所正解確率・事前重み・全q列と一致する。FIFO・警報区間・採用後追加、分母0／微小分母を含める。正の標本数がB未満／B／超過の参加ID、q/n抽出確率、同じRandom状態の抽出ID・順・重複batch、抽出後Random状態を照合する。同じbatchでSGD・Adam standard・AMSGradのloss・全値・grad・optimizer stateが既存共同更新と一致し、実集約入力のn_m・参加分sum(n_m)と集約後全パラメータが既存集約経路と一致する。
  - 完了条件の証拠: current変更と、追加時の過去q列0・登録時の列改名・統合時の列加算・保有外標本の扱い、配布後写し取得と再生後も固定q不変を確認する。小runで発火した警報・候補・更新・集約・統合の回数と検査回数をNotesに残し、未発火の統合等は固定入力検査で補ったと明記する。全体run同士の学習軌跡の一致を合格条件と混同しない。
  - 完了条件の証拠（design S5）: 4構成のモデル初期化順・個数・値、入力生成呼出し順・全clientの特徴／ラベル／概念／標本indexを照合する。入力生成所有RandomStateとbatch所有Randomを区別し、補助評価のRNG消費0を確認する。最初の学習前の予測以外の共通状態・判断の一致も確認する。
  - _Boundary: client接続（明示的な統合task）_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.4, 3.1, 3.2, 3.3_

- [ ] 4. 既存構成の不変性と新構成の保護を検査する
  - 実装前HEADの全構成の順・保存設定・識別子を固定したfixtureと比較し、新構成をgoldenなしの保護集合へ登録する。既存fixtureやgoldenの値は更新しない。
  - 共用fresh process、依存境界、設定スキーマへ接続し、新しい全体runが重ければslowと配り方を登録する。構成の説明を追加し、設定参照は生成コマンドで再生成する。
  - 完了条件: 対象test・基準版1〜16golden・旧回帰2本が成功し、新構成の識別子だけが新しい差を表す。未実施・skipは理由とともに記録する。
  - 明示的な合格条件: Task 1の全構成fixtureの全行で保存mapping・hash・識別子が一致し、構成一覧はfixtureの旧一覧＋新構成1件の末尾追加だけとなる。新構成の保存設定差はdesignの許可表だけで、その識別子だけが差を反映する。旧3構成のSINE2・MNIST4固定runの全既存表（列・dtype・値）・非秒run_metrics・最終値・イベント列・入力系列が変更前資料と一致する。
  - 明示的な合格条件: `test_important_behavior_regression_goldens.py`で`behavior_regression_goldens/v1/`〜`v16/`の環境別JSONの全保存ケースを比較する。関連乱数fixtureの既存test、`test_fedsda_soft_training_configuration.py`、構成・設定・掃引・識別子・依存境界、共用fresh process、`tests/test_regression.py`・`tests/test_proposed_regression.py`を実行する。goldenと乱数fixtureのパス・SHA256が変更前と一致し、更新・再生成していない。Windows／WSL別に実行した検査名・成功件数・skip・未実施理由をNotesへ記録する。
  - _Boundary: 記録・評価と回帰保護_
  - _Requirements: 1.1, 2.4, 3.1, 3.4, 4.4_

- [x] 5. 既存3構成との対の集計を実装する
  - ルートを引数で受け、JSON・summary・NPZ・保存設定の差の許可表から4構成各200キーを検査する。版15の別名対応と診断構成の補助値は分析側にだけ明示する。
  - 完了条件: 欠損・重複・未成功・許可外設定差を入れた固定入力は拒否され、対の差、差の差、20セルをseed内で平均する区間が固定入力で一致する。既存結果を上書きせず、分析CSVとMarkdownへ標準指標・欠損・照合結果を出力できる。
  - 完了条件の証拠: 同seedでもラベル・概念・標本順・特徴digestが異なる固定入力、環境／来歴不明、summaryとJSONの不一致、NPZ不読を含めて拒否・一覧化する。画像digest未保存は照合済み扱いせず未検証欄へ出す。Task 3のRNG所有・初期化順・生成順と全入力照合testの証拠を参照する。監視定義は4構成共通で学習後の値は変わりうると記載する。
  - 完了条件の証拠: 固定入力のMAC・共有／概念固有内訳・上下byteと合計・単位換算が既存計測の期待値と一致し、補助費用を除外しない。時間の数値は記述値、メモリ・未保存診断はN/Aとして扱い、0へ補わないことも固定入力で検査する。
  - _Boundary: 比較の集計_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 6. 全実装の独立レビューと検証を完了する
  - 全実装後に、別セッションの独立CLIで仕様と差分を1回レビューする。可変状態・数値と接続が対象なのでHaiku 5.5 mediumを優先し、利用不能時はLuna medium。指摘の採否・必要な再確認を記録する。
  - 指摘反映後に共通手順の全pytest（slowを除外せず、`--dist loadfile`）、Ruff、Pyright、pip checkを行い、対象HEAD・差分・件数・未検証を記録する。
  - 完了条件: 独立レビューの承認と全検証の証拠がそろい、既存3構成と入力生成の不変性が確認できる。古いsoft specの検証結果を流用しない。
  - 完了条件の証拠: Task 1・4のfixture照合、design S1〜S6に対応するtest名・結果・実際に発火した経路を一覧化する。基準版1〜16・乱数fixtureの前後hash、旧3構成の数値・イベント・入力の照合、全pytestのJUnit・件数、共通手順の品質検査を今回のrevisionへ結び付け、未実施検査名と理由を記す。
  - _Boundary: 全体検証_
  - _Requirements: 3.4, 4.2_

- [ ] 7. 新構成200 runを既存結果と照合して集計する
  - 主担当が実装検証後に、同じ入力条件・環境の準備、読み取り専用の計画と重複先確認を行う。将来の実験用コード共有はその時点のユーザー指示に従う。この下書き作成依頼のcommit・push禁止を過去の許可で上書きしない。
  - 新構成だけを実行するコマンドをユーザーへ渡す。サーバ実行・転送はユーザー担当。実験開始後に台帳を更新し、結果を受領してから全200対の照合と集計を行う。
  - 完了条件: 新構成200件が成功し、既存3構成各200件との設定・来歴・入力の照合結果、標準指標と診断の表、判断と制約が分析ファイルと結果READMEに残る。既存構成の自動再実験はしない。
  - _Boundary: 実験と結果の判断_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

## Implementation Notes

### 独立レビュー後の4点の修正（2026-11-07）

- ユーザーから提示されたMajor 2件・Minor 1件・全pytest失敗1件を対象とした修正。commit・push・stashは行っていない。過去のレビュー記録・承認は変更しない。
- 費用: 現在の実装では`fedsda_measured_run_execution`のprocess全体hookが補助評価を`local_processing`、再生を`synchronization`として既に集計することを確認した。二重加算は足さず、`training_responsibility_weight_updates.py`と`fedsda_run_client.py`へ計測境界をコメントで明記した。`test_auxiliary_cost_is_in_local_and_total_measurement_without_extra_communication`を1500標本へ広げ、非ゼロの複数モデル再生、補助処理ごとの全fieldと全体meter増分の完全一致、全内訳とrun総計の一致、補助処理の時間がrun呼出し全体の経過時間内にあることを検査した。学習・optimizer・追加通信の増分0も維持。run成果物への経過時間の新規保存やメモリ実測は追加せず、未保存値はN/Aのまま。
- 集計: 構成ディレクトリ名、summary全行の構成名、JSONの設定軸を照合し、間隔100・200の除外判定より前に対象外構成を拒否する。特徴digest・client ID・標本indexが不足する対は`unverified`を照合CSVへ記録し、数値集計へ進まず終了コード2で停止する。ファイル作成を伴わないCLI検査・ルート検査・間隔50/100/200の構成名/設定軸拒否検査を追加。成果物fixtureにも対象外構成と誤ルートを追加した。
- 設定参照: 固定venvで`python -m tools.generate_option_docs`を実行。HEADとの差分は新方針のノード・依存辺・参照表の3行追加であることを確認した。
- 予測の既定: `test_fedsda_default_synchronized_local_mixture_prediction.py`の例外表へ新構成の`current_local_parameters`だけを追加。同期写しは学習用で実予測は現在モデルの局所値を使うため、という理由をコメントへ記載。他構成の期待は維持した。
- 検証: 固定venv、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`pytest <指定3ファイル> tests/refactoring/test_single_run_dependency_boundaries.py -k 'not artifact_gate' -q -p no:cacheprovider`で**2342 passed, 11 deselected（86.80秒）**。その後に間隔別拒否6件を追加し、集計ファイルのみ再実行して**19 passed, 11 deselected（1.31秒）**。sourceの計算処理は変更していない。費用検査の単独実行も1 passed（13.84秒）。
- 未実施範囲: 成果物作成fixtureの11ケースは、一時ディレクトリへの`PermissionError`のため実行できない。初回の集計ファイル実行は一時ディレクトリ準備でエラーになり、権限・認証設定は変更していない。全pytest・WSL検証は今回未実施。Pyrightは相対/絶対pythonpath指定ともnumpy/torch等のimport解決ができず213 errorsで不成功。対象source/testのRuff check・format、集計toolの除外解除付きRuff check・format、`git diff --check`は成功。
- 独立再確認: Luna mediumの独立CLIはapp-server初期化のアクセス拒否（os error 5）で起動できなかった。規約に従いHaiku 5.5 mediumへ代替したが、session `8ef92210-ae6c-49bf-8fae-6f94ae8c91b0`は`is_error=true`、`ECONNREFUSED`、`modelUsage={}`で利用不能だった。独立PASS未確認のため、Task 6のレビュー待ち・spec未完了は維持する。

### 下書き作成時の調査（2026-11-07）
- 基点`939489ec24df05dbe6129eedbbac454c590b735e`、ブランチ`refactor/architecture`。開始時のgit管理外`.tmp/`・`.validation-*`は触らない。
- AGENTS、resume、agent-handoff、product・tech・structure、命名規約、既存softとsingleのspec、構成生成・設定検査・予測・責任度・再較正・ID対応のコードを確認した。fable-methodとローカルkiro-spec-init/requirements/design/tasksの手順・雛形を参照した。ユーザー指定に従い、段階の自動承認や実装への進行は行わない。調査メモはこの5ファイル内にまとめた。
- 設定合成の読み取り専用確認は`RunSettingsValidationError`。最初の確認は実行条件のimport先を誤り`ModuleNotFoundError`となったため、既存importを確認して訂正し、組合せ拒否を再確認した。sourceの変更はない。
- 既存結果のJSON・summaryの件数とNPZの存在、200キー、保存設定、環境を確認した。詳細と未検証範囲はdesign.mdの既存結果の節。新しい数値実験・pytest・品質検査・独立CLIレビューは未実施。
- D1・D2は判断待ち。仕様書群の独立レビュー・task graphの独立確認・命名承認は未実施であり、自己確認でPASSとは記録しない。
- 下書きの機械確認: JSON解析、全承認false・実装開始不可、指定5ファイルのみ、要求16項目とtask参照の集合一致、未完了0/7、雛形の残り・行末空白なしを確認した。確認scriptの表示ラベルだけ一度「20項目」と誤記したため、実数16を再計数した。`spec_checks.py progress`と`git diff --check`も成功。後者は未追跡の新規ファイルを対象にしないため、5ファイルを直接検査した。独立レビューや動作testの代替ではない。
- commit・push・stash・実装は行わない。下書きの進捗は0/7で、作成完了と実装完了を区別する。

### Lunaレビュー指摘の反映（2026-11-07）

元レビュー: `results/_sol_prompts/luna-review-soft-training-single-model-spec.out`末尾のFAIL全文。実モデル`gpt-6-luna`・effort `medium`、session `01a1b2b7-6d81-7532-b841-de774c639fef`をログ冒頭で確認した。対象は基点`939489ec`上の未コミット5ファイル。元判定はFAILのまま保持し、今回の反映を独立再確認や承認とは扱わない。直前節の「未実施」「判断待ち」は初回下書き時の履歴である。

| 指摘 | 採否と反映先・理由 |
| --- | --- |
| Major 1 監視損失 | 採用。design「監視損失のコード上の事実」で既存3構成の共通呼出し・局所classifier・有界損失・検出器への引渡しを関数と行で特定した。新構成は未実装の維持契約として区別。requirements 1.4・3.2の具体化、design S1、Task 3で4構成の監視損失列の照合を要求。別定義の選択判断は不要 |
| Major 2 softとの全体照合 | 採用。design S1〜S4とTask 2・3で同期損失列と解放時の局所確率を分け、重み・q・参加・抽出確率と重複batch・共同更新・実集約入力／結果・再生・固定q・集合／ID対応を既存実経路と照合。要求1.3・2.2・3.2へ対応 |
| Major 3 費用・乱数・入力 | 採用。requirements 2.4・3.3・4.2の具体化、design S5・S6、集計前の照合契約、Task 2・3・5・6。run所有Python／NumPy・torchとglobal状態、初期化と入力生成順、全入力照合、推論件数／MAC・内訳／総計・時間scope・保持byte・追加通信0を指定。未保存メモリはN/A |
| Major 4 既存構成・golden | 採用。design「既存構成とgoldenの固定基準」、Task 1・4・6で変更前HEADからの全構成fixture、旧3構成数値・イベント・入力保護と既存goldenの不変hashを定めた。Task 4に新構成だけ末尾追加・既存保存設定と識別子不変を明示 |
| Major 5 D1・D2 | 採用。主担当の指定を受け案Aに全5ファイルを統一。designとspec.jsonへ「主担当の判断（2026-11-07）、ユーザーが2026-11-07に案Aを確定」を記録。Task 1冒頭とspec.jsonの次工程で、案A確定まで実装不可を明記 |
| Minor 1 時系列 | 採用。requirements 2.1の具体化、designのフロー・S1・S2、Task 3で予測確定→ラベル開示→観測時の同期損失による更新→局所監視→解放時の局所確率からq固定→学習を明記。配布後の写し取得→局所保留損失の再生も指定 |
| Minor 2 golden 1〜16 | 採用。design固定基準とTask 4・6にv1〜v16の環境別ファイル、CASE_NAMES_BY_BASELINE_VERSION、比較test・乱数fixture・旧回帰2本・共通手順のコマンドを特定。検査名・件数・skip・未実施理由を記録する |

監視の調査はsource読取りで行い、実runでの4構成比較は未実施（新構成も未実装）。pytest・golden照合・品質検査・独立CLI再確認は今回実施していない。既存の承認履歴は変更せず、Task進捗は0/7、全承認false、`ready_for_implementation=false`を維持する。主担当の新たな方針判断を要する点はなく、D1・D2のユーザー最終確認と修正後のレビュー確認・承認を残す。

修正後の機械確認: 5ファイルのJSON・文字コード・行末空白・Markdown内の相対リンクとコード囲み、全承認false・実装開始不可・案A確認待ち、未完了0/7、要求16項目とtask参照の集合一致、7指摘とS1〜S6の対応を確認した。`spec_checks.py progress fedsda-soft-training-single-model-prediction`は成功。`git diff --check`は成功、tracked差分は無し（このspecは未追跡なので5ファイルを直接検査した）。実装・commit・push・stashは行っていない。

### Task 1〜6の実装・検査（2026-11-07、未コミット）

- ユーザーの今回の指示でTask 1〜6を実装・検査。commit・push・stash・本規模実験は実施しない。kiro-implのmanual / taskごとのreview offでtestを先に置き、全体の独立CLIレビューを別に試みた。過去の承認・レビュー記録は変更しない。現在の実装開始根拠はユーザーの承認済みという指示とspec.jsonのapprovals・ready_for_implementation。
- 今回の基点は"'`ccc5a68caab1b77831778cbdbb49036db7f0a2ce`。下書き時の`939489ec`から進んだHEADを変更前に確認。既存37構成の順・保存mapping全体・SHA256・run識別子と、golden／乱数JSON 37ファイルのパス・SHA256を、新fixture `tests/refactoring/soft_training_single_model_prediction_existing_configurations.json`へ取得した。条件は既存build_settingsと同じSINE2、seed0、2client、300標本、間隔50。取得コマンドは`../../venv/Scripts/python.exe .validation-soft-single/capture_settings.py`（source変更前、再生成していない）。元のsoft_training_existing_configuration_hashes.jsonも未変更。
- 変更前に旧3構成×SINE2(1500標本)/MNIST4(800標本)の全表とrun_metricsを`.validation-soft-single/before-*.npz/.json`へ保存。変更後の再実行6件で全列・dtype・値とrun_metricsが完全一致（existing-runs.log）。最終全パラメータと入力生成の補足証拠は、固定HEADのsrcだけをgit archiveで検証用フォルダへ取り出し、独立processで得た期待値と比較した。全入力のclient・位置・概念・ラベル・特徴dtype/shape/digest、生成呼出し順、初期値、client/serverの最終全パラメータが6件とも一致（head-compare-sine2.log / head-compare-mnist4.log）。これは変更前取得とは区別した固定HEADからの補足照合。固定HEAD側のMNIST初回は配置先の相対データパスで停止し、同じ実データの絶対パスをFDE_MNIST_DATA_DIRへ指定してMNISTだけ再実行した。データやgoldenは変更しない。

| Task | 状態と証拠 |
| --- | --- |
| 1 | 達成。既定値は保存省略。独立方針は責任度学習＋共同更新＋局所単体予測＋予測再較正なしの組だけを受理。37構成の旧mapping/hash/IDは固定fixtureと一致、新構成1件のみ末尾追加 |
| 2 | 実装と対象検査を達成。同期写し選択・有界損失・正規化・Fixed-Share更新と局所保留損失の再生は既存計算を再利用。既存softの実関数との重み・分散・再生件数/ID照合、追加/登録/統合の集合変更、空保留/1モデル、固定q・モデル値/grad/optimizer・RNG不変を確認。写しの独立storageは9288 parameter bytes（固定SINE2の2モデル例）、peak RSSはN/A。推論件数/MAC・学習/optimizer増分0、実runの局所/同期/総計への計上と補助処理中の追加通信0を確認 |
| 3 | 実装と対象検査を達成。専用ownerをproviderへ注入。予測→補助更新→局所監視→q固定→共同更新をspyで確認。同じ状態の4構成の監視結果、ラベルを変えても単体予測確率が不変、4構成の初期値・生成順・全client入力をSINE2/MNIST4で比較。既存softのqと復元batch・RNGを直接照合。既存のSGD/Adam standard/AMSGradの共同更新oracleと、実集約のn_m/共有部分母・全parameterの検査を再利用 |
| 4 | 未達の検査を残す。新構成のgoldenなし保護・共用fresh・依存境界・構成/設定/掃引/識別子検査と文書を接続。4構成入力照合の重いtestはslow、配り方はloadfileの既定。数値goldenは既存testの実行対象206ケース成功。ただし既存testが候補検証削除により112ケースを除外するため、仕様の318保存ケース全件の数値一致とは言わない。旧proposed回帰とfreshはtempの権限エラー。WSLは未実施 |
| 5 | 実装と固定入力検査を達成。4構成各200キー、成功記録、summary/JSON/NPZ、許可表の設定差、環境/来歴、ラベル/概念/位置/特徴digest、欠損/重複を検査。解析の別名対応は分析側だけ。seed内の対の差、20セルのseed内平均によるt(9)区間、相互作用、標準計測・純度/欠損/診断をCSV/Markdownへ出力。800固定runを使ったCLI出力検査も成功。本規模結果の照合自体はTask 7 |
| 6 | 未達。Haiku/Lunaの独立CLIがともに利用不能。自己承認しない。全pytest・fresh・Pyrightも下記の環境制約で完了していない |
| 7 | 主担当。新構成200 runの例はdocs/experiments/experiment-sweep.md。開始時の台帳・結果受領後の実集計/README/判断も主担当 |

#### 仕様と現コードとの差、取り扱い

1. 掃引CLIには`--duplicate-policy error`がない。既存runの再開方式を勝手に変更せず、既存`--print-plan`で今回の新構成200件すべてmissingを確認した。結果ディレクトリは未作成、manifestの重複先は表示されない。実行前の重複・来歴の点検を主担当へ残す（fullscale-plan.log）。本規模コマンドの例に未対応のオプションを混ぜない。
2. 基準版1〜16の既存golden registryは318件、そのうち候補検証削除による112件が既存testでは数値照合対象外。今回、除外・保存値・goldenの意味を変更しない。実行対象206件成功と、両環境の保存構造/hash保護を区別する。
3. sweepの時刻は既存の掃引全体のscopeであり、run別の経過時間・メモリ実測は成果物にない。集計ではN/A。特徴digest未保存の既存結果を直接画像照合済みとはしない。実際の再利用判断と未保存診断は主担当へ残す。

#### 検査の実測（重複するsuiteは合算しない）

| 検査 | 結果 | 証拠（.validation-soft-single/） |
| --- | --- | --- |
| 新構成/独立重みの対象test | 20 passed（最終の監視ラベル不変の加筆も1件再実行成功） | verified-new.xml/log、追加の単独pytest出力 |
| 集計の軽量test | 8 passed、8 deselected（ファイルtestは別記） | final-analysis.xml |
| 集計のファイルtest | pytestでは8 setup error、7 passed後にtemp cleanupもWinError 5。同じ8関数を通常の作業ディレクトリ引数で直接実行し8件PASS | aggregate-tempでの実行出力、manual-final/ |
| 集計CLI・固定800 run | PASS、20セルと全体のseed内平均、差10pp・相互作用−25pp、CSV/Markdown確認 | check_aggregate_cli.py、aggregate-cli-final/（loaderの異常系は上の8件で別途検査） |
| 既存softの拡張suite | 34 passed（新構成のSINE2/MNIST4も含む）、後の境界対象1件追加は軽量suiteで確認 | soft.xml/log |
| 新構成・集計・既存softの軽量確認 | 54 passed、14 deselected | final-focused.xml |
| 設定と依存境界 | 3453 passed | schema-boundary.xml |
| 掃引の設定/識別子 | 27 passed、154 deselected | config-sweep.xml |
| 構成一覧・goldenの両環境構造・候補削除後のfixture保護 | 56 passed、14 deselected | golden-structure.xml |
| 基準版1〜16数値golden | 206 passed（MNIST4 6＋その他200）、0 failed、0 skipped。既存除外112件は上記 | golden-mnist4.xml、golden-other.xml |
| 旧回帰2本＋option/parameter/metric schema | 29 passed、1 failed、0 skipped。failedはtests/test_proposed_regression.pyの一時ディレクトリWinError 5 | legacy-schema.xml/log |
| Ruff check/format | 成功。src/tests/refactoringと、通常の除外対象の新規集計toolはisolatedで別検査 | 最終コマンド出力 |
| 命名機械照合 | 成功、公開関数2＋CLI main 1、未登録なし | spec_checks.py names |
| pip check | 成功（No broken requirements found） | コマンド出力 |

全pytestは`-n 12 --dist loadfile`で試したが、xdistのworker起動前に`Temp/pytest-of-yshin`のWinError 5でINTERNALERROR（テスト数0、全体JUnitなし）。共用fresh_process_smoke.pyも前半の既存経路成功後、TemporaryDirectoryへのWinError 5で停止し、新規末尾の経路には到達しない。Pyrightは固定Pythonの相対パスと絶対パスで試したが、NumPy/Torch等のimportを解決できず213 errors・0 warnings、成功扱いにしない。未解消のsandbox制限を外したり、設定・認証を変更したりしていない。WSLの全検査は未実施。基準版goldenの除外112件、旧proposed回帰、共用fresh、全pytest、Pyright、独立レビューを主担当へ引き継ぐ。

#### 独立CLIレビュー

- 可変状態・数値/RNG・接続のためHaiku 5.5 mediumを優先。`review-haiku.json`はsession `d3375f45-8d72-4177-96a3-55c3ec038961`、`is_error=true`、`modelUsage={}`、ECONNREFUSED。指定モデルの実行・判定は確認できず、PASSとは記録しない。SessionEnd hookにもspawnのEPERM。
- 代替のLuna mediumは初回のPowerShell引数分割を訂正してstdinから再実行。`.codex/tmp/arg0`へのアクセス拒否とin-process app-server初期化のos error 5で起動不能（review-luna.log）。実モデル・effortのログ確認とレビュー判定はいずれも未取得。
- 独立承認は未達。review-prompt.txtへ対象HEAD・未コミット差分・対象ファイル・実測・既知制約・許可した検査コマンドを残した。内蔵子エージェントや自己レビューで代替していない。
'"

#### 変更ファイルの一覧（検証用のGit管理外フォルダを除く20ファイル）

| ファイル | 要点 |
| --- | --- |
| src/federated_learning_experiments/runtime/fedsda_run_client_settings.py | 方針の宣言、既定値の保存省略、組合せの拒否 |
| src/federated_learning_experiments/sweep/experiment_configurations.py | 新構成を末尾追加 |
| src/federated_learning_experiments/runtime/training_responsibility_weight_updates.py | 同期損失での補助更新と局所保留損失の再生 |
| src/federated_learning_experiments/runtime/observed_sample_processing.py | 実予測後・監視前への補助更新の接続 |
| src/federated_learning_experiments/runtime/fedsda_run_client.py | 独立owner・provider、配布後の写しと再生 |
| tools/aggregate_soft_training_single_model_prediction.py | 保存物照合と対の集計CLI |
| federated_drift_experiment/experiment_spec/options.py | 新方針のオプション宣言と依存の説明 |
| docs/reference/options.md | 宣言から再生成 |
| docs/experiments/experiment-sweep.md | 構成の説明、費用と制約、本規模コマンド |
| tests/refactoring/soft_training_single_model_prediction_existing_configurations.json | 変更前37構成とgolden/乱数hashの固定資料 |
| tests/refactoring/test_soft_training_single_model_prediction.py | 設定・数値・RNG・順序・入力・計測・通信の20検査 |
| tests/refactoring/test_aggregate_soft_training_single_model_prediction.py | 集計の数値8検査と保存物の8異常系/正常系 |
| tests/refactoring/test_fedsda_soft_training_configuration.py | 既存の小run・集約oracleを新構成へ適用、候補採用数も記録 |
| tests/refactoring/test_fedsda_configuration_variants.py | 末尾追加の期待値、実旧対応のない集合への分類 |
| tests/refactoring/behavior_golden_cases.py | 新構成の専用testによる保護 |
| tests/refactoring/fresh_process_smoke.py | 新構成の共用fresh経路（今回sandboxで未到達） |
| tests/refactoring/test_single_run_dependency_boundaries.py | 新moduleと呼出元の許可依存・対象一覧 |
| .kiro/specs/fedsda-soft-training-single-model-prediction/spec.json | 進捗4/7と独立CLI利用不能の記録 |
| .kiro/specs/fedsda-soft-training-single-model-prediction/tasks.md | 進捗、照合結果、未検証、仕様との差 |
| .kiro/steering/resume.md | 未完了specと次の確認への入口 |
