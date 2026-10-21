# Implementation Plan

- [x] 1. 最小記述長の選択の計算を足す（接続はまだしない）
  - 純粋な計算（仮説の符号長と選択、切り替わりの符号長、選び方の値の検査）と、モデルを使う計算（対数損失、予測的符号長、seedの導出、保留の列での選択）のtestを先に書き、REDを確かめる。
  - 完了条件: 新しい部品のtestが通る（3種類の選択、同じ長さの順、刻みより短い列、runの乱数を変えないこと、同じseedで同じ値、共有部を更新しないこと）。依存境界の許可を登録する。
  - _Requirements: 1.2, 1.5, 1.6_

- [x] 2. 設定を足し、標本の処理と警報の処理へ接続する（既定は、まだ変えない）
  - 設定の検査（候補検証との組合せの拒否を含む）と、接続のtest（今の扱いでは今と同じ、新しい扱いでは選択の長さと行き先が使われる）を先に書き、REDを確かめる。
  - 設定の項目と検査、clientからの受け渡し、`process_observed_sample`での選択と変化区間の長さの置き換え、警報の処理への受け渡しと、`resolve_alarm_change_interval`での行き先の使い方。全部の設定の組立てで今の扱いを明示し、旧と比べる入口と、基準版1〜9の「後に足された項目」の表へ足す。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜9）が値を変えずに通る。
  - 選択の記録の置き場と記録の表を足し、版1〜9の比べる形から、この表を取り除く（今の扱いでは空の表）。今の扱いで選択の計算が呼ばれないことのtestを足す。
  - _Requirements: 1.1, 1.3, 1.4, 1.7, 1.8, 3.1, 3.2, 4.4_

- [x] 3. 研究ツールと同じ値になることを確かめる
  - 研究ツールの変種のseedを、製品と同じ式に揃える。照合のscriptで、手元の条件（MNIST2・SINE2・CIRCLE2・SEA4、間隔100、seed 0、2500標本）で、研究ツールの変種と、新しい扱いを選んだ製品の全体runの、精度・通信量・予測の記録のdigest・最終のパラメータのdigest・計算回数が一致することを確かめる。
  - 完了条件: 一致の結果（条件ごと）と、照合のscriptの場所がImplementation Notesにある。一致しない場合は、原因を特定して直すまで次へ進まない。
  - _Requirements: 5.1, 5.2_

- [x] 4. 最終構成の既定を替える
  - 既定と全FedSDA構成の値のtest（候補検証ありの構成は今の扱い）を先に書き、REDを確かめてから、既定を替える。既定に依るtest（runの識別子ほか）を、失敗したものから合わせる。
  - 完了条件: 構成のtestが通り、基準版1〜9と実旧との照合が値を変えずに通る。
  - _Requirements: 2.1_

- [x] 5. 基準版10を足す
  - 実装の前の確認: 新しい既定の大きい条件の2ケースで、警報の区間の解決の3種類が、それぞれ1回以上起きるかを確かめる（起きなければ、ケースの選び直しを決める）。
  - 版10の定義、版9→10の差の説明のtest、必須の経路を足し、commitしてから、Windowsの基準環境とWSLでgoldenを生成して足し、現在の版を10へ上げる。
  - 完了条件: 版1〜10の照合と、版9→10の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 6. 文書を合わせる
  - 手法の説明、基準（版10）、実験の入口、研究記録、再開案内。
  - 完了条件: 文書が新しい扱いと、サーバのクラスタリングの閾値が残ることを説明する。
  - _Requirements: 6.1_

- [ ] 7. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 背景（2026-10-21）: 研究ツールの変種`default-detector-gated-segment-assignment`（commit `080f6bc`）の本規模の10 seed（uranium、240 run）で、24条件のうち17で既定との差が0以上、崩れるrunなし、下りは0.84〜1.24倍（[設計と結果の8〜10](../../../docs/research/fedsda-unified-concept-assignment-inference.md)）。ユーザーが既定にすると決めた。
- 仕様レビュー1回目（Luna high、session `01a15c25-47e8-7592-9e2e-52346797d0fe`）: CHANGES REQUIRED。Major 2件（必須の経路を、下流の指標——現在のモデルの維持ほか——で数えると、選んだ仮説を確かめられない。今の扱いで計算回数が変わらないことの確かめ方が無い）、Minor 1件（seedの導出の文字列の形式）。選択の記録を足して選んだ仮説で数える、今の扱いで選択の計算が呼ばれないことのtestと版1〜9の照合（計算回数を含む）で確かめる、`:`で区切る形式に決めてtestで固定する、と直した。
- 仕様レビュー2回目（Luna high、session `01a15c28-2d99-7313-9e1f-b490f1d1d390`）: 1回目の3件は解消。Major 1件（版9→10の差の説明で、新しい記録の表の比べ方が未定義）。両方の版を実行した結果を比べ、版10の結果から表を取り除いて比べ、取り除いた表が空であることを確かめる、`coverage`も版10で足す項目を取り除いて比べる、と直した。
- 仕様レビュー3回目（Luna high、session `01a15c29-7760-79a3-a225-6c66be918cfa`）: 設計は解消。Major 1件（要求4.2が、設計の除外を書いていない）。要求4.2に、両方の版を実行して比べること、表と経路の記録の除外、取り除いた表が空であることを書いた。
- 仕様レビュー4回目（Luna high、session `01a15c2a-6e59-71a3-b854-6c0e8e423d63`）: PASS。
- Task 1の途中の設計の変更（2026-10-21）: 依存境界の規則（runtimeの層はtorchを直接importしない。learningの層もmoduleごとに許可した名前だけ）に合わせ、`evaluate_classifier_per_sample_log_losses`を`learning/prediction/classifier_log_loss_evaluation.py`へ、`compute_new_model_prequential_code_length`を`learning/training/prequential_code_length.py`へ置いた（設計では`runtime/alarm_interval_minimum_description_length_selection.py`だった）。初期化の乱数は、既存の`isolated_cpu_torch_random_state`を使う。公開名の置き場所の変更なので、変更部分の仕様の再確認を受ける。
- 仕様の変更部分の再確認（Luna high、session `01a15c3b-0331-7ea3-8185-d77acb81cbd9`）: PASS（置き場所と依存の方向、命名表との一致、CPUの初期化の乱数が研究ツールと同じになること）。
- Task 1: REDは、新しいmoduleが無いことによる収集の失敗（新しいmoduleだけのtask）。純粋な計算のtest 17件、モデルを使う計算のtest 12件、依存境界のtestと合わせて3156 passed。Ruff・Pyright（`--pythonpath`付き）0 errors。純粋な計算のtestの1件は、最初、testの例の作り方の誤り（保有モデルの前半の損失を0にしたため、区間10が最短になった）で失敗し、例を直した。`reset_parameters`の呼び出しは、Pyrightの型の推定に合わせて`getattr`で取り出して呼べるときだけ呼ぶ形にした（torchの部品では同じ意味）。
- Task 2: REDは4件（新しい項目が無いことによる失敗）。設計からの変更2点（層の規則に合わせた）: (1) 選び方の値の定数と検査の関数`validate_alarm_interval_assignment_policy`は、純粋な計算のmoduleではなく、設定のmodule`training_data_assignment_settings.py`に置いた（`*_settings.py`は`core`と標準libraryしかimportできない）。(2) 選択の記録の置き場`AlarmIntervalAssignmentSelectionRecordStore`は、`evaluation/`ではなく`methods/fedsda/training_data_assignment/alarm_interval_assignment_selection_record_store.py`に置いた（evaluationの層から手法の層の型を参照しない）。`handle_alarm_occurrence`の新しい引数は、この関数の引数を既定値なしにする規則（testで確かめている）に合わせて必須にし、中で呼ぶ`respond_to_alarm_with_buffered_samples`・`resolve_alarm_change_interval`は既存の呼出しを変えないよう既定値`None`にした。新しい扱いで選んだときの警報の記録の`detector_candidate_start_sample_index`は、研究ツールと同じく、選んだ変化区間の先頭にした（検出器の候補の開始は、選んだ変化点より後のことがあり、記録は候補の開始≦変化点を求める）。選択は、検出器の合図か、ADWINの補助の判定で警報になるときに呼ぶ（研究ツールは検出器の合図だけ。既定のe-SRでは補助の判定は起きないので、既定の値は同じ）。runの識別子`4403b4d6816c`→`73df845459ba`（設定の項目が増えたため。値は今の扱い）。全pytest（`../../venv/refactoring-tests/mdl-task2-full.xml`）は14681件のうち231件が失敗し、全部が`test_observed_sample_processing.py`と`test_alarm_wait_for_minimum_change_interval.py`で、標本の処理の引数を、警報の処理のtestの引数の組から作るときに、新しい引数`alarm_interval_assignment_selection`が混じったため。除く引数に足し、2つのファイルで244 passed / 2 skipped。ほかのファイルは、その全pytestで通った（全pytestの取り直しはTask 7で行う）。Ruff・Pyright 0 errors。
- Task 3: 研究ツールの変種`default-detector-gated-segment-assignment`の学習者のseedを、`derive_new_model_initialization_seed`と同じ式（clientの番号は、製品が標本の処理へ渡す`client_id`）に揃え、製品の扱いを選ぶだけの変種`default-minimum-description-length`を足した。`results/fedsda-provisional-candidate-expert/20261021_mdl-product-equivalence-n2500/`（MNIST2・SINE2・CIRCLE2・SEA4、間隔100、seed 0、各2500標本、8 run）で、2つの変種の精度・下り・上り・予測の記録のdigest・最終のパラメータのdigest・計算回数・最終のモデル数・登録・統合・検出の回数・変化点の推定の指標が、4条件とも完全に一致した（照合は、Git管理外の一時scriptで行った）。4条件とも3種類の選択が起きた（SEA4は保有モデルへの切り替わりが0回）。研究ツールのseedを揃えたので、uraniumの240 run（通し番号のseed）とは、新しいモデルの学習者の初期化が違い、値は完全には一致しない（Requirement 5.2）。
- Task 4: REDは1件（既定が今の扱い）。既定を`minimum_description_length`にし、構成`fedsda-with-candidate-validation`は今の扱いを明示した。組合せの検査があるので、`_replace_candidate_acceptance_and_shared_feature_update`に、選び方も一緒に置き換える引数を足した（候補検証を外した構成は、既定の値を渡して引き継ぐ）。基準版のhelperの版1〜6の設定（候補検証ありへ替える）も、選び方の置き換えを同じ1回の置き換えにまとめた（途中の設定が「候補検証あり＋最小記述長」で拒否されたため）。既定に依るtest（最終構成の既定の期待値、候補検証ありの構成との差——2つから3つの値へ、共有部の固定と候補検証の組合せのtestの組立て）を合わせた。runの識別子`73df845459ba`→`f6cae723e023`。全pytest 14674 passed / 8 skipped / 失敗0（510.9秒、`../../venv/refactoring-tests/mdl-task4-full.xml`。基準版1〜9と実旧との照合を含む）。
- Task 5: 実装の前の確認（Git管理外の一時script、Windows）で、`fedsda-sine2-adoption-without-validation`は3種類の選択がそれぞれ1回以上起きたが、`fedsda-mnist2-adoption-without-validation`は「現在のモデルのまま」4回・「保有モデルへ」0回・「新しいモデルへ」1回だった。設計の「2ケースで3種類それぞれ1回以上」を、「2ケースで選択が1回以上、SINE2のケースで3種類それぞれ1回以上」に緩めた（MNIST2で保有モデルへの切り替わりを起こすケースを新しく足すより、既存の2ケースを保つ方を選んだ。保有モデルへの経路はSINE2のケースが守る）。版10の定義と版9→10の差の説明のtestをcommit `bcbe8ed`に置き、そのcommitで、Windowsの基準環境（Python 3.13.15）とWSL（Ubuntu、Python 3.14.4）で、各2回の独立したprocessの一致を確かめてgoldenを生成した。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を10へ上げた。合成のgoldenで現在の版の構造を確かめるtest（5件）が、版10の経路の項目を持たないために失敗したので、合成のgoldenのFedSDAのケースに3つの回数を足した。照合（goldenの版1〜10、このspecのtest、来歴のtest）: Windows 453 passed（`../../venv/refactoring-tests/mdl-golden-windows.xml`）、WSL 453 passed（`mdl-golden-wsl.xml`）。
- Task 6: 手法の説明（`docs/overview/proposed-method.md`）、再利用の説明（`docs/components/model-reuse.md`に新しい節）、基準（`docs/experiments/refactoring-baseline.md`の版10）、実験の入口（`docs/experiments/experiment-sweep.md`）、研究記録（設計の文書の12、IMPROVE-021の19と状態、改善候補の一覧）、再開案内を合わせた。サーバのクラスタリングの閾値γ=0.1が残ることを、手法の説明・再利用の説明・研究記録・再開案内に書いた。`python -m tools.generate_option_docs`で`docs/reference/options.md`に差が出ないことを確かめた（旧実装の設定の参照なので、新しい項目は載らない）。
