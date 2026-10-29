# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4 → task 5。task 1の「替える前」の保存は、既定を替える前に行う。task 3のgolden生成・設置・版の定数の切替は主担当が行う。

- [x] 1. 替える前の結果を保存する（コードは変えない）
  - このspecの基点のcommitで、要求4の5条件の構成`fedsda-every-sample-change-interval-search`の全体runのrawと、要求6.4の研究ツールの小さい条件の結果（研究用CLIの`default`、組Cの3変種、`product-default`、`product-default-loss-monitor-decrease-rebaseline`、`every-sample-change-interval-search-configuration`、研究用の入口の組み合わせた変種）を、Git管理外（`../../venv/refactoring-tests/every-sample-default-before/`）へ保存する。保存のscriptもGit管理外に置く。
  - 研究ツールの全変種（研究用CLIと研究用の入口の両方の`_VARIANTS`の全部）の設定と差し替えを、設計の取り出し方（実際の実行の経路を通し、製品のrunの直前で止める）で取り出して保存する（要求6.3）。task 2で同じscript・同じ変種の集合で取り出して比べる。要求6.4の、組Eの代表と研究用の入口の`product-default`に新しい扱いだけを当てた結果も保存する。
  - 版15の18ケースのうちFedSDAのケースを、`change_interval_search`だけ新しい扱いにして実行し、最小変化区間の倍数でないkの選択の回数をケースごとに記録して、版16の必須の経路の2ケースを決める（要求5.3）。
  - 完了条件: 保存した場所・条件・commitと、ケースごとの回数・決めた2ケースをImplementation Notesに書く。取り出した設定と差し替えに、2つの入口の`_VARIANTS`の各名前がちょうど1回ずつ現れること（名前と件数の照合）を確かめる。研究ツールの小さい条件で警報が1回以上起きることを確かめる。
  - _Requirements: 4.1, 5.3, 6.3, 6.4_
- [x] 2. 既定を替え、構成と設定の組と研究ツールを合わせる
  - 既定・構成（要求1）と最小記述長でない選び方の組（要求2.1）のtestを、新しいtestのファイル`tests/refactoring/test_fedsda_every_sample_change_interval_search_default.py`に先に書き、REDを確かめてから、既定を替え、新しい構成を足し、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ足す。
  - 既定に替えた後に失敗する、最小記述長でない選び方の組（旧と比べる入口、実旧との照合の組、testとfresh script）で今の扱いを明示する。既定に依るtest（runの識別子、構成の一覧ほか）を、失敗したものから合わせる。
  - `fedsda-every-sample-change-interval-search`を既定をそのまま返す形にし、識別子が実行済みの結果と同じで`fedsda`と違うことのtestを足す（要求1.2）。
  - 研究ツールの組A〜Eの値と測定の差し込みの条件を、検査のtestを先に書いてから足す（研究用CLIと研究用の入口。要求6.1〜6.3）。
  - 替えた後の結果を`../../venv/refactoring-tests/every-sample-default-after/`へ保存し、task 1の結果と、要求4.1の(a)〜(c)・要求6.3（全変種の設定と差し替え）・要求6.4で比べる（一時scriptはGit管理外）。
  - 研究ツールのsourceで`change_interval_search`を読む箇所を検索し、要求6.3の記述（`reverse_prequential_policy`の1か所と検査のtestだけ）と合うことを記録する。
  - 完了条件: 新しいtestと研究ツールの検査が通り、基準版1〜15と実旧との照合が値を変えずに通り、要求4.1と6.4の比較が期待どおり（同じであるべきものは同じ、替わるべきものは替わる）。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1, 3.2, 4.1, 6.1, 6.2, 6.3, 6.4_
  - _Depends:_ 1
- [ ] 3. 基準版16を足す
  - 版16の登録、経路の記録の新しい項目（版16から）と必須の経路、goldenを使わない版15→16の差のtestを足す（実装担当）。commitの後、主担当が、必須の経路の2ケースの値を確かめ、Windowsの基準環境とWSLでgoldenを生成して`v16/`へ置き、`BASELINE_VERSION`と`BEHAVIOR_BASELINE_VERSION`を16へ上げる。
  - 完了条件: 版1〜16の照合と、版15→16の差のtestが、WindowsとWSLで通る。
  - _Requirements: 5.1, 5.2, 5.3_
  - _Depends:_ 2
- [ ] 4. 文書を合わせる
  - 提案手法の概要、モデル再利用の文書、基準の文書（版16）、実験の入口、IMPROVE-032と一覧、再開案内。
  - 完了条件: 文書が新しい既定を説明し、過去の日付の記録を書き換えていない。
  - _Requirements: 7.1_
  - _Depends:_ 3
- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_
  - _Depends:_ 4

## Implementation Notes
- Task 1・2（実装: GPT-6.1-sol、1回の依頼でTask 1を全部終えてからTask 2）: 保存先は`results/_sol_tmp/every-sample-default/{before,after}/`（Git管理外。specの`../../venv/refactoring-tests/`の代わり）、比較の結果は同じ場所の`comparison.json`、変更の一覧と検証の証拠は`final-evidence.json`。基点は`a4efa6e3`。
  - 要求4.1: 5条件（MNIST4・SINE2・SEA4・CIRCLE2は間隔50・seed 0、SINE2はseed 1。10 client・各client 1000標本、`run_experiment_sweep.py --workers 1`）とも、記録の表の129配列・要約・最終のパラメータ（全モデル）が完全一致。
  - 要求6.3: 研究用CLIの72変種と研究用の入口の6変種の全部（名前と件数が`_VARIANTS`と一致）で、組A〜Cは設定と差し替えが前後で同じ、組D・Eは`change_interval_search`の1か所と`reverse_prequential_policy`の有無だけが違う（`every-sample-change-interval-search-configuration`は差なし）。
  - 要求6.4（SINE2・MNIST4、600標本・間隔100）: 20組のうち18組が全出力で一致。組Eの`product-default-class-conditioned-prediction`の2組は`prequential_measurement`だけが違う（前は測定を差し込み、後は要求6.2のとおり差し込まず空）。これは決めた差なので一致とみなす（主担当の判断）。後の`product-default`は、前の`every-sample-change-interval-search-configuration`と同じで、最小変化区間の倍数でないkの選択がMNIST4で2回。研究用の入口の`product-default`も、前に新しい扱いだけを当てた結果と同じ。
  - 研究ツールで`change_interval_search`を読む箇所（検査のtestを除く）: 設定の組立て（`run.py`の3か所、`research_loss_monitor_rebaseline.py`の1か所）と、`run.py`の測定を差し込むかの判定の1か所。`reverse_prequential_policy`は製品の選択へ引数をそのまま渡すだけで、名前では読まない。どれも要求6.3の比較の範囲（設定と差し替え）。
  - 要求5.3（版15のFedSDAのケースに新しい扱いだけを当てた、Windows）: `fedsda-sine2-adoption-without-validation` 7回、`fedsda-mnist4-warm-started-learner` 8回、ほかのFedSDAの12ケースは0回。第一候補の2ケースに決めた。
  - 変更: 既定（`fedsda_final_configuration_run_settings.py`）、構成（`experiment_configurations.py`。`fedsda-every-sample-change-interval-search`は分岐を消して既定をそのまま返し、新しい構成`fedsda-minimum-interval-multiples-change-interval-search`の分岐を足した）、研究ツール（`run.py`、`research_loss_monitor_rebaseline.py`、新しい検査`checks_change_interval_search_default.py`、`checks_loss_monitor_rebaseline.py`）、test（新しい`test_fedsda_every_sample_change_interval_search_default.py`、`behavior_golden_cases.py`、`test_fedsda_configuration_variants.py`、`test_fedsda_final_configuration_run_settings.py`、`test_fedsda_reverse_prequential_change_interval_search.py`、`test_single_run_dependency_boundaries.py`）。`src/`で最小記述長でない選び方の組に今の扱いを明示する箇所は無かった（既定を引き継いで最小記述長でない選び方にする組立てが`src/`に無い）。
  - 主担当の確認: 関係するtest 9ファイルの3,815件のうち2件が失敗（`test_feddrift_experiment_sweep.py`と`test_fedavg_experiment_sweep.py`の、FedSDAの識別子の直書き`sine2_interval3_seed0_8f83dd4d55c0`）。既定の値が変わったための想定どおりの変化なので、新しい値`82d64db8888d`に直し、2ファイル50 passed。研究ツールの検査103 passed。Ruff check・format成功。全pytestとgoldenの全版の照合はTask 5で行う。
- Task 3（コード: GPT-6.1-sol、`8835cad4`）: 版16の登録（版15と同じ18ケース）、経路の記録`non_minimum_interval_multiple_change_interval_selection_count`（版16から、FedSDAだけ）と手で作った表での定義のtest、必須の経路（2ケースで1以上）、goldenを使わない版15→16の差のtest（18件）。Solの実行で新しい23件成功。版16のgoldenが無い間は、構造の検査1件が失敗し数値の照合18件がskip（版13・15と同じ扱い）。
- 版16のgoldenの生成（主担当）: Windowsは`8835cad4`で生成に成功（149秒、必須の経路の新しい回数はSINE2のケース7・MNIST4のケース8）。WSLは、版11からの必須の経路「`fedsda-mnist4-warm-started-learner`でwarm startの学習者が1回以上短い」が0回で失敗した。WSLの版16では、FedSDAの14ケースすべてでwarm startの学習者の回数が0（Windowsの版16はMNIST4のケースだけ2）。WSLの版15のこのケースは1回、版16の新しい回数は6回。ユーザーの判断（2026-10-29「版16は必須から外す」）で、版16ではwarm startの学習者の必須を外し、初期化し直した学習者の必須は残した（`behavior_golden_cases.py`の`assert_required_paths`。要求5.3に追記）。warm startの学習者の計算は、前のspecの部品のtest（途中までの和）で守る。
