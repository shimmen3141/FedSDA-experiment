# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4 → task 5。task 2はtask 1の新しい項目と研究ツールとの照合の一致が前提。task 3（版13）はtask 1・2の既定が確定した後。
task 4（文書）はtask 3の版の定数とgoldenの後。task 5は全部の後。

- [x] 1. 損失の低下で監視の基準を下げる扱いを選べるようにする（既定は、まだ変えない）
  - 部品のtest（Validation Designの(i)〜(viii)）と、設定の検査のtest（不正な値・型、ADWINとの組合せ）を先に書き、REDを確かめる。
  - 値の名前と組、検出の設定の新しい項目と検査、監視器の下がる向きの検出器と基準の引下げ、監視の計算回数への算入（合計と下がる向きの検出器の分）、観測の新しい項目、状態の写しの新しい項目、
    基準を下げた事象の記録の表（`append_baseline_lowering_record`）とrunの指標。全部の`LossChangeDetectionSettings`の組立て（最終構成、構成、旧と比べる入口、実旧との照合の組、test・fresh script）で今の扱いを明示し、
    基準版1〜12の「後に足された項目」の表へ足す。
  - 研究ツールの診断の各事象にclient IDを足し（要求6.4）、研究ツールの専用の検証が通ることを確かめる。
  - 研究ツールとの照合（基準版12の既定の上で、新しい2つの値だけを置き換えた製品の全体runと、研究用の入口の組み合わせた変種。4データ）を行い、結果を記録する。
    この照合で使う上限＝clientの数の設定は、照合の一時scriptで置き換える（既定はtask 2で変える）。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜12）が値を変えずに通り、研究ツールとの照合が一致する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.2, 5.4, 6.4_

- [x] 2. 最終構成の既定を替え、研究ツールの基点を固定する
  - 既定と全FedSDA構成の2つの値（候補検証ありの構成は「基準を警報まで保つ扱い」と上限3、ADWINの2構成は「基準を警報まで保つ扱い」）のtestを先に書き、REDを確かめてから、既定を替える。
    版1〜12の上限3の明示（参加者の設定の「値が変わった項目」）を足す。既定に依るtest（runの識別子ほか）を、失敗したものから合わせる。
  - 研究用CLIと研究用の入口の基点の固定・IMPROVE-031の変種の基準版12への戻しを足し、既定を替える前に保存した小さい条件の結果と、替えた後の結果が同じことを確かめる
    （研究用CLIの`default`・`product-default-loss-monitor-decrease-rebaseline`、研究用の入口の組み合わせた変種）。
  - 完了条件: 構成のtestが通り、基準版1〜12と実旧との照合が値を変えずに通り、研究ツールの既存の変種の値が変わらない。
  - _Requirements: 2.1, 3.1, 4.2, 6.1, 6.2, 6.3_

- [ ] 3. 基準版13を足す
  - [x] 今回の担当範囲のコード（版13の登録・経路の記録と必須の経路・goldenを使わない版12→13の差のtest）。golden生成・配置・定数の切替とWindows/WSLの全golden照合は主担当が行うため、親taskは未完了。
  - 経路の記録の新しい項目（版13から）、版13の定義と必須の経路、版12→13の差の説明のtestを足し、commitしてから、Windowsの基準環境とWSLでgoldenを生成して足し、現在の版を13へ上げる
    （goldenの生成・設置・定数の切替は主担当が行う）。
  - 完了条件: 版1〜13の照合と、版12→13の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 4. 文書を合わせる
  - 手法の説明、基準（版13）、実験の入口、LEGACY-023・LEGACY-021・実装の不具合の一覧、IMPROVE-031、v12・v13比較、再開案内。
  - 完了条件: 文書が新しい扱いを説明する。
  - _Requirements: 7.1_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合の結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 最後の対象検証（task 5の全pytestではない）: `../../venv/Scripts/python.exe`、`PYTHONPATH=src;.`、OMP/MKL/OPENBLAS各1。新spec、損失監視・記録store・設定検査・依存境界・既定/構成・候補検証・クロス評価・掃引/識別子・研究専用検証の実行可能部分は4778 passed / 88 skipped / 88 deselected（34.94秒、exit 0、`results/_sol_tmp/rebaseline_targeted_final.log`）。追加の補助部品のtestは293 passed / 1 deselected（61.17秒）。別実行の件数は重複するため合算しない。tmp_path/tmp_path_factoryを使うtestは既知のPermissionErrorのため、Git管理外の実行補助でskipし、未実行として扱った。先に通常のpytestで確認したsetup errorは隠さず途中の記録へ残す。
- 品質検査: 指定範囲`src tests/refactoring tools`のRuff check成功（exit 0）、Ruff format --check成功（397 files already formatted、exit 0）、git diff --check成功（exit 0）。初回のformat失敗は編集したファイルの整形で解消した。GitのLF/CRLF警告がPowerShellのリダイレクトでNativeCommandErrorに表示されたため、gitの実exit codeも明示確認した。無関係なファイルの整形は行っていない。
- 未実施と懸念: 全pytest、版1〜12の全ケースのgolden、版13のgolden生成・設置と数値照合、Windows/WSLの全体照合、Pyright、fresh process smoke、独立実装レビュー（task 5）は未実施。権限で使えないtestを成功扱いにはしていない。版13のgoldenを前提とする既存testは、主担当の設置まで完了しない。旧実装・旧goldenと版1〜12のgolden・AGENTS.md・CLAUDE.md・agent-handoff.md・requirements.md/design.md/naming.mdは差分なし。commit/push/stash・再委譲・独立レビューの起動は行っていない。
- 変更したファイル（責務別。新規test以外は既存のファイル）:
  - `src/federated_learning_experiments/`: `methods/fedsda/loss_change_detection/{loss_change_detection_settings,overall_and_true_class_loss_monitoring}.py`、`evaluation/{loss_change_alarm_record_store,loss_monitoring_computation_count_store}.py`、`runtime/{fedsda_final_configuration_run_settings,fedsda_run_client,fedsda_run_metric_derivation,fedsda_run_record_tables,observed_sample_processing}.py`、`sweep/experiment_configurations.py`。
  - `tools/`: `research_loss_monitor_rebaseline.py`、`research_provisional_candidate_expert/{run,loss_monitor_rebaseline,checks_loss_monitor_rebaseline}.py`。
  - `tests/refactoring/`: 新規`test_fedsda_loss_monitor_rebaseline_and_all_evaluating_clients.py`、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`fresh_process_smoke.py`、`test_{alarm_wait_for_minimum_change_interval,bounded_loss_moments,loss_baseline_selection,model_and_class_loss_statistics,pending_training_assignment_buffer,run_settings_validation,single_run_dependency_boundaries,fedsda_run_client,fedsda_final_configuration_run_settings,fedsda_configuration_variants,fedsda_with_candidate_validation,fedsda_experiment_run_artifacts,fedavg_experiment_sweep,feddrift_experiment_sweep}.py`。
  - 文書: `docs/overview/proposed-method.md`、`docs/components/drift-detection.md`、`docs/experiments/{experiment-log,experiment-sweep,refactoring-baseline}.md`、`docs/research/implementation-findings/{README,legacy-021-classwise-clustering-merges-with-little-evidence,legacy-023-loss-monitor-baseline-frozen-at-alarm-reset}.md`、`docs/research/improvement-candidates/improve-031-loss-monitor-rebaseline-on-loss-decrease.md`、`docs/research/fedsda-v12-v13-comparison-and-candidate-freezing-20261026.md`、`.kiro/steering/resume.md`、本`tasks.md`。Git管理外のscriptと結果は`results/_sol_tmp/`のみ。
- Task 3のコード: 版13未登録によるRED 1 failedを確認後、同じ22ケースを登録し、版13からFedSDAのcoverageへ引下げ回数を足した。候補検証あり4ケースは0、指定されたSINE2とMNIST4の2ケースは1以上を必須とした。goldenを使わず両版の全22ケースを実行する差のtestと、版1〜13の全ケースの上限・監視方針の直接検査は23 passed / 499 deselected（221.57秒）。旧来の必須の経路も通る。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`は12、v13のディレクトリは作っていない。
- 差のtestの判断: 版13の新しい指標（回数と下がる向きの計算回数）と表は、候補検証ありでは0と空であることを直接検査し、その追加分だけを除いて版12の既存の全項目と比較する。版1〜12のcaptureも追加分が0であることを検査して除くため、既存goldenの値とschemaは変更しない。設計の「指標・表が同じ」は、この明示したschema追加分の検査を伴う同一性として実装した。
- 既定切替後の照合: 版1〜12の小さいFedSDAケース132 passed / 390 deselected（67.65秒）。runtimeのclient・server・全体run・指標・保存の実行可能部分87 passed / 80 skipped（167.44秒）。旧と照合する経路はkeepと上限3（元から異なる上限を使う組はその値）を明示する。80件はtmp_pathの権限で未実行と、それに依存する網羅test。既定の構成・研究・候補検証・識別子は、最後の対象検証107 passed / 26 skipped / 11 deselected（29.49秒）。識別子の期待値2件は、実測の`fc28fa415515`へ合わせた。
- Task 4: 手法・ドリフト検出の基準の説明・基準版の手順・実験入口と台帳・LEGACY-023/021と一覧・IMPROVE-031・版12/13比較・resumeを更新した。既定のコードが変わったことと、配置済みgolden/定数が12のままであることを分け、履歴の研究結果は保った。文書だけの変更に動作testのREDは追加しない。
- CLIの読取り専用の確認: 新既定のSINE2・seed 0・10 client・600標本・間隔100を`--print-plan`で確認し、予定1 run、状態missing、識別子`79b7f09c618b`を確認した。出力先`results/_sol_tmp/rebaseline_readonly_plan`が前後とも存在せず、計画確認がディレクトリを作らないことも確認した（4データの照合に先立つ確認ではない）。
- Task 1の研究照合（既定の切替前、全条件10 client・2500標本）: 一時scriptは`results/_sol_tmp/rebaseline_compare.py`。製品は基準版12の既定の2値だけを置き換えて全体runを実行し、製品の保存関数でrawを作成。研究は`python -m tools.research_loss_monitor_rebaseline`の組合せ変種で保存した。以下の4条件で(a)研究の既存145配列をarray_equal（浮動小数はequal_nan=True）、(b)列挙した除外以外の全要約、(c)製品の監視合計−下がる向きの分、(d)client ID別の位置・前後の基準の列と総回数が許容差なしで一致した。

  | データ | seed | 間隔 | 引下げ回数 | 上限適用前の候補の最大数 |
  |---|---|---|---|---|
  | SINE2 | 0 | 50 | 66 | 10 |
  | CIRCLE2 | 0 | 50 | 113 | 10 |
  | SEA4 | 1 | 50 | 23 | 10 |
  | MNIST2 | 0 | 100 | 125 | 8 |

  結果は`results/_sol_tmp/rebaseline/<dataset>_s<seed>/comparison.json`、rawと研究診断も同じ配下（Git管理外）。追加した製品の引下げの表は研究側では空になるため、既存145配列の比較から分け、(d)で研究の診断と直接比較した。今回の照合では実行前の`--print-plan`確認を実施していない（研究入口には同オプションが無い）。出力先は専用の未使用パスで、通常の重複拒否を変えず、既存結果の上書きはしていない。この手順上の不足は、結果の一致とは分けて記録する。
- Task 2の実装と固定の確認: 既定をlower＋client数へ切替。候補検証ありはkeep＋3、ADWINの2構成はkeep＋client数を明示。製品の上限3は候補検証ありの構成でだけ宣言し、旧比較の入口と版1〜12は歴史上の値3を明示する。版1〜12はclientの「後に足された項目」の表へkeepを足し、参加者の「値が変わった項目」の表で3を組み立て、保存済み設定から除かずに比べる。研究CLIの非product系は版10、IMPROVE-031と上限撤廃だけの比較は版12へ戻してから介入する。研究入口は定数の差替えを外し、構成が呼ぶbuilderの結果を置き換える形にした。product-defaultは製品の既定を使う。
- 研究の固定の実測（SINE2・seed 0・10 client・600標本・間隔100）: `before.json`と`after.json`で、研究CLIのdefaultとproduct-default-loss-monitor-decrease-rebaselineの秒数以外の全出力が一致。研究入口の組合せ変種も`entry_before/`と`entry_after/`の145配列・全非秒要約が一致した。案Aの小さい条件でも1回の引下げがあり、そのclient IDは8。研究専用の観察だけの照合は、製品の対照を版12の設定で作る形へ合わせる（製品の新既定との直接比較では意味が変わるため）。
- Task 2の検証: 新specは22 passed。既定に依る6件とADWINの構成期待値1件の失敗を確認して修正し、既定・構成・掃引の実行可能部分は46 passed / 62 skipped / 77 deselected（2.07秒）。候補検証ありの差のtestの失敗1件を、追加された2値も直接検査する形へ修正し、再実行1 passed。skipは権限を要するtmp_pathと、それに依存する網羅testで、未実行として扱う。
- 実装担当の範囲（基点`f992a3b5`）: task 1〜4のtest・source・文書を直接実装する。task 3はコードだけ。golden生成・設置、版の定数の切替、task 5、独立レビュー、再委譲、commit・push・stashは行わない。版13のgoldenが前提となるtask 4の記述は、コード準備とgolden未配置を区別して書く（今回の指示で文書も担当するため）。
- Task 1のRED: 部品・設定のtestを先に足し、必須設定項目の未実装で18 failedを確認した。次に記録と計算回数のtestを足し、未実装で2 failed / 18 passedを確認した。境界値では下がる向きの検出器の合図が1標本で出て再開されるため、増分のtestは再開前の検出器を照合する形にした。
- Task 1の処理: 今の監視の観測→下がる向きの観測→上がる合図が無いときだけ基準を下げる、の順序を研究案Aと合わせた。全体だけの経路とクラス別混合の両方を接続。返す観測は引下げ前の値に計算回数・引下げの項目を足し、引下げ時のlast_observationはreset後のNoneのまま。観測・計算回数store・run指標に下がる向きの分を持ち、既存合計へも算入する。事象は警報と別の表`loss_monitoring_baseline_lowering`へclient・位置・前後の基準を保存する。帰属・保留・適応・重みの処理は変更しない。研究診断はContextVarでprocess_observed_sampleのclient IDを各事象へ足す。
- Task 1の途中の検証: 新規20件成功。既存の設定testの辞書の組立てとmetadata期待値の不足134件を確認して明示を足した。部品・設定・依存境界は4624 passed / 1 failed（追加したmath.isfiniteの許可登録漏れ、その後修正）。版1〜12の小さいケース群のgolden照合180 passed / 296 deselected（89.99秒）。既存goldenは変更していない。研究専用検証は56 passed / 6 deselected / 12 setup errors（一時ディレクトリのPermissionError。12件は未実行）。ログは`results/_sol_tmp/rebaseline_*.log`。
- Task 2のRED: 既定と全FedSDA構成の2つの値のtestを先に足し、既定がkeepのため1 failedを確認した。既定の切替はtask 1の研究照合の完了後に行う。
- 進め方（基準版12のspecと同じ。2026-10-23のユーザー指示を引き継ぐ）: 実装（task 1〜4のtest・source・文書。task 3のgoldenの生成・設置・定数の切替を除く）はGPT-6.1-sol（独立CLI、`codex exec`）が行い、
  主担当（Claude）は、仕様書群の作成、実装前の独立CLIによる仕様レビューの起動、実装の確認（差分と仕様書群の照合、対象testの実行）、goldenの生成、全pytest、実装レビューの起動を行う。
  実装レビューは、実装した担当とも主担当とも別の独立CLIで行う。
- 仕様レビュー1回目（GPT-6 Luna、effort medium——ログで確認——、session `01a17932-aca8-7e40-8c09-1c8ee0d5b4b4`、対象`5d3bc98b`）: FAIL。Major 4件・Minor 2件を全部反映した（`54fef164`）。
  - Major 1（研究ツールとの照合の条件が不足。標本単位の比べ方、許容差、計算回数の扱い）: 要求5.4を、許容差なしの(a)全記録の表の配列、(b)要約の全項目（除く項目を列挙）、(c)監視の計算回数（下がる向きの検出器の分を別の項目にして、合計から引いた値を比べる）、
    (d)基準を下げた事象の、clientごとの（標本の位置、前と後の基準）の列、に分けて定めた。製品に基準を下げた事象の記録の表と、計算回数の下がる向きの検出器の分の項目を足した（要求1.4・1.5、設計）。
  - Major 2（同じ標本の後続処理との順序が未定義）: 返す観測・`_last_observation`・log Eの記録・警報の判定・計算回数の記録・事象の記録・次の標本の連続性の検査の関係を、要求1.3と設計に端から端まで書いた。
  - Major 3（版1〜12の上限3の確かめ方）: 既存の照合（保存してある`run_settings`と組み立てた設定の比較）で確かめられることと、新しいtestで版1〜12・版13のケースごとの上限を直接確かめることを、設計に書いた。
  - Major 4（公開名の不足）: 新しいメソッド`append_baseline_lowering_record`と記録の表の名前を命名の表に足し、役割が変わる既存の公開名（監視器のメソッド、観測、計算回数の型と記録）を表に足した。fieldは規約どおり表に入れず、設計に書く理由を明記した。
  - Minor 5（taskの依存）: tasks.mdの冒頭に依存を書いた。
  - Minor 6（表とケースの対応）: 版1〜12の新しい項目はclientの「後に足された項目」の表、上限は参加者の「値が変わった項目」の扱い、候補検証ありの4ケースは構成の明示、と設計に書いた。
- 仕様レビュー2回目（GPT-6 Luna、effort medium——ログで確認——、session `01a17936-365d-77d0-ab79-8e51fb66a2af`、対象`0f6e1d4c`）: FAIL。1回目の6件は解消・概ね解消。Major 1件・Minor 1件を反映した。
  - Major 1（研究の診断の`monitor_index`と製品のclientの対応を、記録から直接確かめられない）: 研究ツールの診断の各事象にclient IDを持たせる要求6.4を足し、照合（要求5.4の(d)）をclient IDで直接対応させる形にした。
  - Minor 2（要求1.3の「後続処理は返す観測を使う」の範囲が広い）: 返す観測を使うのはlog Eの記録・警報の判定・計算回数の記録だけで、保留・帰属・位置の管理は今と同じ入力で行うこと、
    次の標本の検査のうち監視器の連続性の検査だけが警報の応答の後と同じで、保留の位置の検査は今と同じであることを、要求1.3と設計に書き分けた。
- 仕様レビュー3回目（GPT-6 Luna、effort medium——ログで確認——、session `01a17938-2fe1-78b2-84c7-48eaf5118a8f`、対象`38ffcc03`）: PASS。2回目のMajor 1件・Minor 1件の解消を確認、残る指摘なし。ready_for_implementation=true。
- Task 3（主担当）: 生成元`539363a7`（Solの実装のcommit）。Windows（Python 3.13.15）とWSL（Ubuntu、Python 3.14.4）で、helperの`--baseline-version 13`により各2回の独立したprocessの一致と必須の経路を確かめて22ケースを生成し
  （`../../venv/refactoring-tests/v13-golden-windows.json`、`v13-golden-wsl.json`）、`v13/`へ新規に置いて、`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を13へ上げた（`30385447`）。
  基準を下げた回数: `fedsda-sine2-adoption-without-validation`はWindows 41・WSL 35、`fedsda-mnist4-warm-started-learner`は5・6、候補検証ありの4ケースは0、ほかの既定を引き継ぐケースは0（runが短く引下げの前に終わる）。
- 主担当の確認（Solの実装の後）: 製品の差分を仕様書群と照合した（監視器の更新順序・返す観測・`_last_observation`・計算回数の2系統・`reset`での作り直し・状態の写し、
  後続処理の記録、既定と候補検証あり・ADWINの構成、設定の検査）。対象の製品のtest 4703 passed / 12 skipped（Windows、8並列）。研究ツールの専用の検証54 passed。
- 最終検証の1回目の全pytest（Windows、`tests`、8並列、`v13-full.log`）: 15056 passed / 8 skipped / 6 failed。WSLの版1〜13のgoldenとこのspecのtest: 561 passed / 5 failed。
  - 失敗5件（両環境）: 合成のgoldenのtest（`test_unknown_environment_has_no_match_but_reads_structure`、`test_unicode_dtype_is_a_valid_digest`、`test_generation_requires_two_matching_process_results`の3件）が、
    版13で足した経路の記録の項目を持たない。版12と同じく、合成のgoldenへ`loss_monitoring_baseline_lowering_count`を足した（必須の経路の2ケースは1、ほかは0）。
  - 失敗1件（Windows）: `test_current_model_selection_without_baseline_adopts_a_new_model`。新しい既定では、小さい条件で「現在のモデルのまま」の選択が起きなくなった。
    このファイルの小さい設定は基準版10の既定を基点にすると明示しているので、基点に`keep_baseline_until_alarm`を足した。
- 最終検証（`30385447`）: 全pytest（Windows、`tests`、8並列）15062 passed / 8 skipped / 失敗0（1311秒、exit 0、JUnit `../../venv/refactoring-tests/v13-full-2.xml`。版1〜13のgolden、実旧との照合、
  `tests/test_regression.py`・`tests/test_proposed_regression.py`を含む）。WSL（Ubuntu、Python 3.14.4）の版1〜13のgolden・来歴・このspecのtest 566 passed（1346秒、exit 0、`../../venv/refactoring-tests/v13-golden-wsl.xml`）。
  Pyright（venvのPythonを`--pythonpath`で指定、`src`）0 errors。Ruff check（`src tests/refactoring tools`、プロジェクトの設定）成功。
- 実装レビュー1回目（Claude Haiku 5.5、effort medium——指定値——、`is_error=false`、`modelUsage`は`claude-haiku-5-5`、session `70c92316-c3ed-4dbe-80e3-297c7ec329bb`、対象`f992a3b5..30385447`）: FAIL。
  コード・testは要求と研究の案Aに一致と判定（監視器の順序・返す観測・計算回数・reset・状態の写し・後続処理・既定と構成・版1〜12の明示・版13の必須の経路・研究ツールの基点・主担当のtestの修正2箇所、命名の検査は未登録なし）。
  レビュー担当はtestの実行が権限の承認待ちで実行できず、読みによる確認。指摘と採否:
  - Major 1（文書がHEADと矛盾。`resume.md`、`refactoring-baseline.md`、`experiment-sweep.md`が版13のgolden未作成・定数12と書く）: 採用。挙げられた3か所に加え、同じ古い記述を
    `drift-detection.md`・`proposed-method.md`・v12・v13比較・LEGACY-021・LEGACY-023・`resume.md`の別の行でも直した。あわせて、Solの書き込みで文字が「?」に壊れていた
    `docs/research/implementation-findings/README.md`のLEGACY-021・LEGACY-023の行を、元の件名と新しい状態で書き直した（レビューでは見つかっていなかった。主担当が文書の確認で見つけた）。
  - Minor 1（基準を下げた回数が環境で違う）: 採用。goldenは環境ごとに持つ設計で違反ではない。回数を`refactoring-baseline.md`と上のTask 3に記録した。研究ツールとの照合（task 1）はWindowsで行った。
  - Minor 2（引下げの後の位置の検査と処理の順序を直接確かめるtestがない）: 採らない。研究ツールとの照合で、標本ごとのlog Eを含む記録の表の全配列・計算回数・事象が許容差なしで一致し、
    版13のgoldenの必須の経路（引下げが起きるケース）も2環境で通るので、この順序は全体runで確かめられている。testは重要なもの・後で効くもの・複雑なものに絞る方針（設計）にも沿う。
  - Minor 3（Ruff format --checkの記述の範囲。`tools/`はプロジェクトの設定で対象外）: 採用。`tools/`はRuffの対象外で、上の品質検査の記述の「397 files」は`src`と`tests/refactoring`の範囲（`tools`は除かれる）。
- 実装レビュー2回目（Claude Haiku 5.5、effort medium——指定値——、`is_error=false`、`modelUsage`は`claude-haiku-5-5`、session `8573b404-8810-446e-955f-9ca3a220347b`、対象`30385447..7a85a68f`。修正の再確認）: FAIL。
  1回目の修正箇所・READMEの2行・文字化けの解消・Minorの採否は確認された。残る指摘: Major 4件（同じ古い記述が`proposed-method.md`の22行目、IMPROVE-031の冒頭、`resume.md`の「次の一手」4、
  LEGACY-021の本文の状態欄に残る）、Minor 1件（改善候補の一覧にIMPROVE-031の行がなく、「次は」の番号が古い）。全部反映した（次のcommit）。
- 実装レビュー3回目（Claude Haiku 5.5、effort medium——指定値——、`is_error=false`、`modelUsage`は`claude-haiku-5-5`、session `fe1d4d17-167c-4e81-ac40-099a0051ab05`、対象`7a85a68f..39a624b1`）: FAIL。
  2回目の5件は解消。新たにMajor 3件（`refactoring-baseline.md:240`の「現在は9」、`ablation-plan.md:11`の「現在の基準は…版6」、`resume.md`の「次の一手」2）・Minor 3件（IMPROVE-029の10行目、IMPROVE-031の節名、`resume.md`の「直近に完了したspec」）。
  ユーザーの判断（2026-10-27）: 実装に問題がないのに文書のずれでレビューを繰り返すのは無駄。日付つきの過去の記録は、その時点の記録として書き換えない。Major以上は実装・結果に影響する実質的なものに限る。
  これを規約（`.kiro/steering/agent-handoff.md`の「レビューに出す前と依頼文」の重大度と文書の扱い）に反映した。この基準での採否:
  - 直した（今の状態の正本）: `refactoring-baseline.md`の版の表（番号を書き写さず、定数と基準版の節を参照する形にした）、`resume.md`の「次の一手」2・「直近に完了したspec」・動作基準の行（定数と基準の文書を正本として参照）。
  - 直さない（日付つきの過去の記録、または正しい記述）: `ablation-plan.md:11`（「現行基準の比較（2026-10-17）」の節の記録）、IMPROVE-029の10行目（2026-10-26の書き足し）、
    IMPROVE-031の節名（研究用の変種の基点は今も基準版12で、記述は正しい）。
  - 今の状態の正本の文書（上の規約の一覧）を機械的に走査し、今の状態として古い版番号を書いた記述が残っていないことを確かめた。

