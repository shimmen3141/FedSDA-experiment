# Implementation Plan

- [x] 1. 逆向きの1本の流れの計算を足す（接続はまだしない）
- [x] 1.1 予測的符号長の途中までの和とseed
  - 途中までの和（最後の要素が今の関数の値と完全に同じ、各途中までの和が同じ順序の先頭i+1件を今の関数で測った値と完全に同じ——初期化し直した学習者は同じseedで——、runの乱数と`template_classifier`を変えない、入力の検査）と、`derive_reinitialized_learner_single_pass_initialization_seed`（`client_id:alarm_sample_index`の文字列のSHA-256の先頭8 byteから直接計算した値と同じ、検査）のtestを先に書き、REDを確かめる。`_accumulate_prequential_code_length`を、今の2つの関数の値と足す順を変えずに直す。
  - 完了条件: 新しいtestと、予測的符号長・最小記述長の選択の既存のtestが通る。依存境界の許可を登録する。
  - _Requirements: 2.2, 2.4_
  - _Boundary:_ `learning/training/prequential_code_length.py`、`runtime/alarm_interval_minimum_description_length_selection.py`（seedの関数だけ）
- [x] 1.2 純粋な計算のkの刻み
  - `select_assignment_by_minimum_description_length`の必須の引数`change_interval_step_sample_count`（刻み1で候補が最小変化区間〜nの全部、刻み5で今と同じ候補、辞書の鍵が候補と違えば拒否、正の整数でなければ拒否、今の扱いの値が変わらない、同じ長さのときの優先順が刻み1でも同じ）のtestを先に書き、REDを確かめる。呼出し側（`select_alarm_interval_assignment`、今の扱い）は刻み＝`change_interval_grid_sample_count`を渡す。
  - 完了条件: 新しいtestと、最小記述長の選択の既存のtestが通る。
  - _Requirements: 2.1, 2.5, 1.3_
  - _Boundary:_ `methods/fedsda/training_data_assignment/minimum_description_length_assignment_selection.py`と、その呼出し

- [x] 2. 設定を足し、接続し、構成を足す（既定は変えない）
  - 先に書き、REDを確かめるtest:
    - 設定: `validate_change_interval_search`（2つの値を受け、builtin str以外と他の値を拒否）、`FedsdaRunClientSettings`で`detector_change_span_and_mean_loss_increase_tolerance`と新しい値の組合せを拒否。
    - 接続（今の扱い）: 途中までの和の2つの関数と1本の流れ用のseedの関数が1回も呼ばれない（呼ばれたら失敗する差し替え）、値が変わらない。
    - 接続（新しい扱い）: 列Wを新しい順へ並べ替えた入力で、初期化し直した学習者→warm startの学習者の順に1回ずつ途中までの和を計算する、kの候補が最小変化区間〜nの1件刻み、列が最小変化区間より短いと学習者の関数を呼ばず「現在のモデルのまま」で選択の記録の候補が空、選んだkが選択の記録・件数不足の判定・区間の取込みに渡る、測定用の学習者のoptimizerの更新回数が学習者ごとにn回。
    - 構成: 新しい構成は既定から`change_interval_search`の1項目だけ違う、既定と既存の全構成の値が変わらない（既存の構成は今の扱いを明示または引き継ぐ）、全構成の組立てが例外を出さない。
  - 実装: 設定の項目と検査、clientからの受け渡し、新しい構成、最終構成の既定で今の扱いを明示。`tests/refactoring/behavior_golden_cases.py`の`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`へ、版1〜15の全部にパス`("training_data_assignment_settings", "change_interval_search")`＝今の扱いの値を足す（goldenのJSONは変えない）。FedSDAのrunの識別子を直書きしたtest（`tests/refactoring/test_experiment_sweep.py`ほか）を新しい値に合わせ、FedDrift・FedAvgの識別子が変わらないことを確かめる。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜15のgolden）が値を変えずに通る。識別子の変更前後の値をImplementation Notesに書く。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.3, 2.6, 2.7, 3.1, 3.2, 4.1, 4.2, 5.2_
  - _Depends:_ 1

- [x] 3. 研究ツールの変種R1と同じ値になることを確かめる
  - 研究ツールに、新しい構成を使う変種`every-sample-change-interval-search-configuration`を足す。照合のscript（Git管理外）で、手元の小さい条件（MNIST4・SINE2・SEA4・CIRCLE2、間隔50、seed 0、各client 1000標本程度。列Wが最小変化区間より短い警報が1回以上起きる条件を含める。起きなければ条件を足す）で、研究ツールの変種R1とこの変種の、精度・下り・上り・予測の記録のdigest・最終のパラメータのdigest・計算回数の指標・選択の記録（警報ごとの変化区間の長さ、仮説、行き先、学習者の名前、符号長）が完全に一致する（許容誤差なし）ことを確かめる。
  - 完了条件: 一致の結果（条件ごと）と、照合のscriptの場所がImplementation Notesにある。一致しない場合は、原因を特定して直すまで次へ進まない。
  - _Requirements: 5.1_
  - _Depends:_ 2

- [x] 4. 文書を合わせる
  - モデル再利用の文書の最小記述長の節、実験の入口の構成の一覧、IMPROVE-032の状態と一覧、再開案内。既定が変わらないことを明記する。
  - 完了条件: 文書が新しい選択肢と構成を説明する。
  - _Requirements: 6.1_
  - _Depends:_ 2

- [x] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー（GPT-6 Luna、effort medium）、指摘の反映、全pytest・Ruff・Pyright・WSL（goldenの版1〜15とこのspecのtest）・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 1〜6_
  - _Depends:_ 3, 4

- [x] 6. 本規模の実験と集計
  - commit・pushの後、主担当が内部確認（`--print-plan`）をして（6データのコマンドの計画に新しい構成と`fedsda`の両方が各240 runずつ含まれること、新しい4データのコマンドの計画に`fedsda`が含まれず、既定は実行済みの版15の結果とデータ・間隔・seedで対応付けられること、重複先が無いことを確かめる）、uraniumで流す2つのコマンド（設計の「本規模の実験」。6データは新しい構成と既定の480 run、新しい4データは新しい構成の160 run、64 workers）をユーザーに渡す。結果が届いたら、既定のFedSDA（6データは同じsweepの版15、新しい4データは実行済みの版15）とデータ・間隔・seedで対応させ、6データの版15の既定が版13の結果と一致するかを確かめ、`tools/research_feddrift_delta_comparison.py compare`で対の差・SD・勝ち/負け・通信・MACを、選択の記録から警報の回数と選んだkの分布を集計する。
  - 完了条件: 640 runの完了（失敗0）を確かめ、版13との一致の確認と集計をIMPROVE-032と実験の台帳に記録する。採否はユーザーが判断する（既定は変えない）。
  - _Requirements: 7.1, 7.2_
  - _Depends:_ 5

## Implementation Notes
- 背景（2026-10-29）: 研究ツールの変種R1（`default-reverse-prequential-grid1`、commit `75fcc893`）の研究室サーバでの128 run（`results/fedsda-reverse-prequential/20261029_083601_uranium-n5000`）。結果は[IMPROVE-032](../../../docs/research/improvement-candidates/improve-032-reverse-order-prequential-code-for-all-change-intervals.md)。ユーザーが、specとして実装して本規模で実験すると決めた（既定にはしない）。
- 仕様レビュー1回目（GPT-6 Luna、effort medium、対象`4acdbc31`）: FAIL。Major 3件（研究ツールとの数値の一致の比べ方、本規模の実験のtaskが無い、版15のgoldenとrunの識別子の扱いが具体的でない）、Minor 3件（純粋な計算の刻みの引数の契約、短い列の扱いと設定のtestの具体化、命名の「reverse」の意味）。全部反映: 製品と研究ツールの足す順が同じ（標本1件ずつの評価の値をfloatで足す）ことをコードで確かめ、完全一致（許容誤差なし）を要求した（要求5.1、設計）。要求7とTask 6（本規模の実験と集計）を足した。`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`への追加と識別子のtestの扱いを設計とTask 2に書いた。刻みの引数の契約、Task 2のtest項目、命名の説明を足した。
- 仕様レビュー2回目（GPT-6 Luna、effort medium、対象`78069465`）: FAIL。1回目の指摘は全部反映済みと確認（足す順が同じことも両方のコードで確認）。Major 1件（6データの比較の相手が版13の結果で、版15と同じ根拠がspecに無い）、Minor 1件（`--purpose`が仮の文字）。反映: 6データは同じsweepで版15の`fedsda`も流して比べる相手にし（240 run増えて計640 run）、版13の結果との一致は整合の確認として記録する。コマンドの`--purpose`を具体的にした。
- 仕様レビュー3回目（GPT-6 Luna、effort medium、対象`32d7fa4b`）: PASS。Minor 1件（実行前の計画の確認で見る点を明記する）。文面の補足だけなので反映し、再レビューは省いた（2026-10-09のユーザー決定）。
- Task 1（実装: GPT-6.1-sol）: REDは、1.1が新しい関数が無いことによる42件の失敗、1.2が刻みの引数が無いことによる19件の失敗。`_accumulate_prequential_code_length`に、各標本を足した直後の和を残す内部の引数を足し（今の2つの関数の値と足す順は同じ）、2つの学習者の内部の関数を共通にして、総和の関数（今の公開名のまま）と途中までの和の関数（`compute_*_prequential_code_length_prefixes`）を足した。`derive_reinitialized_learner_single_pass_initialization_seed`を足した。純粋な計算に必須の`change_interval_step_sample_count`を足し、`select_alarm_interval_assignment`は刻み＝`change_interval_grid_sample_count`を渡す（今の扱い）。既存のtestの呼出し3ファイル（`test_alarm_interval_minimum_description_length_selection.py`、`test_minimum_description_length_assignment_selection.py`、`test_research_s3_contribution.py`）に刻みの引数を足した。新しいtest 67件（`tests/refactoring/test_fedsda_reverse_prequential_change_interval_search.py`）。Solの実行で、新しいtest・関係するtest・依存境界の2,398 passed、版15のgolden`fedsda-mnist4-warm-started-learner`の照合が一致。主担当の再実行で、新しいtest・最小記述長の関係のtestと版15のgolden（MNIST4・SINE2のケース）225 passed。Ruff成功、Pyright（設定の範囲）0 errors。変更した既存のtest 3ファイルを個別にPyrightにかけると46件の指摘が出るが、どれも今回足した行ではない既存の行で、設定の範囲外（testは対象外）。依存境界の許可は、新しいmodule間のimportが無いので変えていない。
- Task 2（実装: GPT-6.1-sol）: REDは、新しい設定・接続・構成の24件が未実装で失敗。設定の項目`change_interval_search`と値の定数・検査、`FedsdaRunClientSettings`での組合せの拒否、最終構成の既定で今の扱いを明示、`process_observed_sample`→`select_alarm_interval_assignment`への受け渡し（新しい扱いは列Wを新しい順へ並べ替え、初期化し直した学習者→warm startの学習者の順に途中までの和を1回ずつ計算し、刻み1を純粋な計算へ渡す。列が最小変化区間より短いと学習者を計算しない。今の扱いは今の経路のまま）、構成`fedsda-every-sample-change-interval-search`、基準版1〜15の「後に足された項目」の表への追加（goldenのJSONは不変）。runの識別子: FedSDA `sine2_interval3_seed0` `f9bff53a0cdc`→`8f83dd4d55c0`、`circle2_interval10_seed0` `96d95864063a`→`a52261f74bb2`、FedDrift（`c5772bf27573`）・FedAvg（`fb8821cf5897`）は不変。既存の全構成は、新しい項目を除いた設定のhashが変更前と同じ。Solの環境は一時ディレクトリの権限エラーで全体の確認が完了しなかった（基準版1〜15のgoldenの数値照合188件は全部一致）。主担当の全pytest（Windows、12並列）: 13,172 passed / 8 skipped / 失敗0（1,222秒、JUnit `../../venv/refactoring-tests/reverse-prequential-task2-full.xml`。基準版1〜15のgolden、実旧との照合、`tests/test_regression.py`、`tests/test_proposed_regression.py`を含む）。ほかに、`results/implementation-audits/fedavg-20261021/test-results.txt`をpytestが収集して文字コードで失敗したエラー12件（testではない成果物のファイル。コードと無関係）。直前のspecの最終の全pytest（13,030 passed / 9 skipped）から、このspecで足したtestの分だけ増えた。Ruff成功、Pyright（設定の範囲）0 errors。
- Task 3（実装: GPT-6.1-sol）: 研究ツールに、新しい構成をそのまま使う変種`every-sample-change-interval-search-configuration`を足した（研究ツールの更新回数の測定`reverse_prequential_policy`は、この変種では使わず、`prequential_measurement`は空。R1との照合は製品の計算回数の指標と選択の記録で行う）。照合のscript`../../venv/refactoring-tests/reverse-prequential-task3-compare.py`、結果`reverse-prequential-task3-compare.json`。研究ツールの`_execute_job`を順に呼び、間隔50・各client 1000標本で、MNIST4・SINE2・SEA4・CIRCLE2（seed 0）とSINE2（seed 1、列Wが最小変化区間より短い警報を起こすために足した条件）の5条件とも、精度・下り・上り・予測の記録のdigest・最終のパラメータのdigest・計算回数の指標・選択の記録の全8列が、R1と完全に一致した（許容誤差なし）。短い列の警報は、SINE2 seed 1のclient 4・標本934〜936で長さ2・3・4件の3回、両方とも「現在のモデルのまま」。研究ツールの既存の検査`checks_reverse_prequential.py`を、Task 1・2で足した必須の引数に合わせた。主担当の環境で研究ツールの検査86 passed（Solの環境では一時ディレクトリの権限エラーで17件が未確認だった）、このspecの製品のtest 108件成功。
- Task 4（実装: GPT-6.1-sol）: `docs/components/model-reuse.md`（設定の2つの値、新しい扱いの測り方とseed、短い列、既定は不変）、`docs/experiments/experiment-sweep.md`（構成の一覧）、IMPROVE-032と一覧（製品の選択肢と構成として実装、既定にはしない、本規模の実験は未実行）、`.kiro/steering/resume.md`（現在地と次の作業）。
- Task 5: 実装レビュー（GPT-6 Luna、effort medium、対象`6bbac118..519b7170`）: PASS、指摘なし。既定の経路が新しい関数を呼ばないこと、新しい経路の並べ替え・途中までの和・kの候補・短い列と境界の長さ、設定の検査と組合せの拒否、基準版1〜15と識別子の扱い、testが個別に確かめていること、研究ツールの測定の扱いを、差分と仕様で照合した。レビュー担当の環境ではpytestが一時ファイルを作れず、独立の実行はしていない。主担当の検証: 全pytest（Task 2の後、Windows、srcはその後変更なし）13,172 passed / 8 skipped / 失敗0、研究ツールの検査86 passed（Task 3の後）、WSL（Ubuntu、Python 3.14.4、元のcheckoutの`.venv`）で基準版1〜15のgoldenとこのspecのtest 464 passed / 1 skipped（402秒、JUnit `../../venv/refactoring-tests/reverse-prequential-task5-wsl.xml`）、fresh smoke（Windows）全フローPASS（50秒、`../../venv/refactoring-tests/reverse-prequential-task5-fresh.log`）、Ruff成功、Pyright（設定の範囲）0 errors。
- Task 6: 実行前の内部確認（`--print-plan`）で、6データのコマンドは`fedsda`と新しい構成が240 runずつ、新しい4データのコマンドは新しい構成の160 runだけで、重複先が無いことを確かめた（出力のディレクトリは作られない）。uranium（64 workers、commit `541989d1`）で`results/fedsda-every-sample-change-interval-search/20261029_151540_tuned-datasets-seeds0-9`（480 run）と`20261029_172159_new-datasets-seeds0-9`（160 run）が完了・失敗0（sweepの記録の`run_status_counts`）。6データの版15の`fedsda`は、版13の結果と全体の累積精度・上りの通信・警報の回数が240 runとも一致した。`tools/research_feddrift_delta_comparison.py compare`の集計（`results/fedsda-every-sample-change-interval-search/comparison-{tuned,new}-datasets.csv`）と、別のscriptの集計が全セルで一致した。精度差は6データ平均+0.014pp・新しい4データ+0.023ppで、通信・MACの比は0.95〜1.05。kの分布・選択の内訳・変化点の推定誤差とあわせて[IMPROVE-032](../../../docs/research/improvement-candidates/improve-032-reverse-order-prequential-code-for-all-change-intervals.md#製品構成の本規模の結果2026-10-29uraniumspecのtask-6)と実験の台帳に記録した。採否はユーザーが判断する（既定は変えていない）。
