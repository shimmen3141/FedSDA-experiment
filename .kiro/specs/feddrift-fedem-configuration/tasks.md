# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求3.1〜3.3のtestを先に書き、REDを確かめてから、設定の項目（割り当ての方式の新しい値、学習の標本の範囲の新しい値）、4構成、runtimeの分岐（E-step、重み付きのM-step、全モデルの集約、$\pi_c$ による予測の混合）、round数の掃引の値の許可、記録の表（保存形式の版を上げる）を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、IFCA・Window・CFL・FedDrift・Oracleのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`が通る。Ruff・Pyright。
  - _Requirements: 1.1〜1.9, 2.1, 3.1〜3.4_
- [x] 2. 文書と検証の記録
  - 要求4.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）とnaming.mdの追記。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、予備の実行（96 run）のコマンドを渡す。design.mdの集計をユーザーに示し、本規模に使う構成とround数をユーザーが決める。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2

## Implementation Notes

### Task 1・2の実装（2026-11-04、基点fa83ab50、未コミット）

- 4構成を構成一覧の末尾へ追加。`model_assignment_policy="fedem_responsibility_assignment"`、windowは`training_sample_window="current_timestep"`、historyは`"all_observed_history"`。初期化は既存の`all_random` / `pretrained_cluster0`を再利用。既存27構成の保存設定と識別子の固定hashはそのまま照合する。
- `runtime/fedem_mixture.py`にclientのπ・qとE-step・混合予測を置いた。全学習標本を最大256標本ずつforwardし、今の時刻のqの平均でπを更新する。historyでも記録するqの最大成分はその時刻の標本だけ。E-stepはno_gradで実モデルを呼ぶため、既存の計算量hookが推論標本を数える。
- `feddrift_client_stream_processing.py`で予測の後に標本を蓄積し、全成分の学習storeへ渡す。`feddrift_run_client.py`で毎roundのE-step→π更新をM-stepの直前へ接続。`feddrift_client_round_training.py`で、既存と同じ件数条件・非復元抽出・成分ごとの更新予算を使い、`Σ qℓ / |B|`で更新する。集約にはqを渡さない。
- `feddrift_run_server.py`でFedEMの全成分を毎roundの対象にする。既存の学習標本件数による集約・全clientへの配布・通信計数を再利用。participant factoryでM個の登録・初期化を許可し、掃引計画で4構成のRを許可。IFCA等の既存の拒否条件は維持。
- `feddrift_run_record_tables.py`に`fedem_client_mixture` / `fedem_sample_responsibility`を追加。FedDrift run artifactとsweep recordの版は9。版番号を固定する既存test（artifact、round、fixed-share、FIFO、sweep）を更新。CLI引数一覧と計画の項目を固定するtestを読み、今回は新引数・新計画項目がないため、それらの期待値は維持した。
- `experiment_spec/options.py`に選択肢・有効条件・選択肢間の制約を登録。新しい数値項目はなく、round数は既存のParameterSpecと掃引軸を再利用。`python -m tools.generate_option_docs`で設定参照を再生成。構成一覧、公式との差(a)〜(g)、名前表を追記。
- RED: 新testを先に追加し、FedEM module未実装の`ModuleNotFoundError`（collection error、exit 2）を確認した後に実装した。公式996faddの式の出典をtestに記載。式、初期化、二値・4クラスの実E-step→π→重み付きM-step、R=3の予測順・混合・転送・更新回数を確認する。
- 先行の確認: 依存境界2,260 passed。スキーマ（option / parameter / metric）28 passed。新testの非slow部分9 passed。構成順の位置を固定するWindow testの期待値を更新し、該当ファイルの非slow 26 passed。既存の全27構成の設定・識別子の固定hashの照合はPASS。
- Ruff checkはPASS。Pyrightは通常の`--pythonpath`ではsite-packagesを探索できず204 errors（依存の未解決）。元のpyproject.tomlを継承しvenvPath/venvだけを明示した`.validation-fedem/pyright.json`では224ファイル、0 errors / 0 warnings。検査対象・規則は変更していない。命名検査（`spec_checks.py names ... --base fa83ab50`）は未登録の公開名なし。
- 一時ディレクトリを使う既存testは`PermissionError: [WinError 5]`でsetupできないものがある。成功扱いにはしない。pytest終了時のtmpdir cleanupも同じ理由で失敗するため、JUnitと`.validation-fedem/`の各test結果を使って区別する。先行の非slow確認では54件がsetup権限エラー、構成順の位置指定の2 failureは上記の修正後に通過。
- Windows golden: `test_important_behavior_regression_goldens.py`は380 passed / 19 setup errors / 0 failures / 0 skips（JUnit 399件、973.361秒）。基準版1〜16の数値golden照合206件は全件PASS。19件はtmpdirのWinError 5またはその作成に続くWinError 3で、数値照合の不一致ではない。`.validation-fedem/golden.xml`へ記録。goldenは更新していない。
- run artifactの既存testは20 passed / 37 setup errors。37件はすべてpytest tmpdirのWinError 5。版番号の期待値は、実行できない保存testも含め9へ更新済み。
- FedDrift系全testファイルの非slow確認は1,140 passed / 50 setup errors / 4 skipped / 0 failures。setup errorsはrun artifact 37件とexperiment sweep 13件（すべてtmpdirのWinError 5）。skip 4件はLinux専用の数値指標のtest。`.validation-fedem/feddrift-fast-reports.json`に各testの結果を保存（上記artifact単独確認と重複するため、合計へ二重加算しない）。
- 最終版の全体runはR=1をSINE2/MNIST4×window/historyの4条件、R=3をSINE2×window/historyの2条件で確認する。R=3の(b)(d)を両scopeで守り、R=1の4条件では(a)〜(h)を確認する。混合の照合はforward出力をhookで捕捉し、oracleのための追加forwardを計算量へ混ぜない。
- 独立レビューCLI・全pytest・commit・push・stash・Task 3の予備実験は実行していない。Task 1の完了条件のうち権限制約と主担当の独立レビュー・全pytestは残るため、checkboxは保留する。

変更ファイル（検証用の未追跡`.validation-fedem/`は別）:

- runtime: 新規`fedem_mixture.py`、`feddrift_client_round_training.py`、`feddrift_client_stream_processing.py`、`feddrift_run_client.py`、`feddrift_run_client_settings.py`、`feddrift_run_participant_factory.py`、`feddrift_run_server.py`、`feddrift_run_record_tables.py`、`feddrift_run_artifact_files.py`。
- sweep: `experiment_configurations.py`、`experiment_sweep_plan.py`、`experiment_sweep_records.py`。
- 宣言: `federated_drift_experiment/experiment_spec/options.py`。
- test: 新規`test_feddrift_fedem_configuration.py`、`behavior_golden_cases.py`、`test_single_run_dependency_boundaries.py`、`test_experiment_sweep.py`、`test_feddrift_experiment_run_artifacts.py`、`test_feddrift_ifca_window_configuration.py`、`test_feddrift_window_and_adaptive_fedavg_configuration.py`、`test_feddrift_window_multiple_training_rounds.py`、`test_fedsda_fixed_share_time_scale_axis.py`、`test_fedsda_pending_assignment_fifo_capacity_axis.py`（いずれも`tests/refactoring/`）。
- 文書: `docs/experiments/experiment-sweep.md`、`docs/experiments/differences-from-feddrift.md`、生成した`docs/reference/options.md`、本specの`naming.md`と`tasks.md`。

### 主担当の確認（2026-11-04）

- 実装担当（GPT-6.1-sol、session `01a1a2b3-5f0d-7562-952d-930232816389`）はCodexの使用上限で、最後の確認（`-history`の小さい全体run）の途中で止まった。残りの確認は主担当が行った。
- 主担当が読んで確かめたこと: E-stepの式、2クラスのBCE（モデル出力が確率で、既存の学習と公式の2クラスと同じ）、更新の回数が`updates_per_sample × detection_batch_sample_count`で累積の標本数によらないこと、予測のπが直前のE-stepの値であること。
- 新しいtestファイル全体（slowの小さい全体run 6件を含む）: 15 passed（1,690秒、主担当のWindows）。
- 実装レビュー: Lunaが使用上限のため、ユーザーの指示（2026-11-04）で一時的にClaude Haiku 5.5（`--effort medium`の指定、JSONの`is_error=false`、`modelUsage`は`claude-haiku-5-5`、session `b5a707d8-857f-429d-b2fe-146c8823a7af`）が担当した。判定PASS、Major以上なし（`results/_sol_prompts/fedem-impl-review-haiku.json`）。Haikuが独立に実行したもの: 新しいtestの非slow 9件、`sine2-history-1`の全体run、依存境界2,260件、Ruff。
- Minorの採否: (1) qと標本の対応を多重集合でしか見ていない——採用。学習のstoreの標本列がqを計算した標本列と同じ順であることを確かめる検査を足した。(2) `-pretrained-component0`のID 1〜$M-1$が乱数の初期化の構成と同じ値か未検証——採用。両構成を組み立てて比べる検査を足した。(3) 1次元tensorの`.T`の警告——採用。(4) E-stepがπを更新した後にtrainerのqの検査が走る——不採用。qは内部で作る値で、検査で例外が出ればrunは止まり、結果に影響しない。(5) 他moduleの内部属性の参照——不採用。既存の`_settings`の参照と同程度。修正後の非slow 9件はpassed（UserWarningをエラー扱いにしても通る）、Ruff check・format check成功。
- 全pytest（主担当、Windows、`-n 8 --dist loadfile tests`、5,537秒、`.validation-fedem/full-pytest-claude.xml`）: 13,442 passed、8 skipped、1 failed。失敗は`test_fedsda_configuration_variants.py`の構成の一覧と比較対象から除く集合の期待値にFedEMの4構成が無かったもの（実装担当の環境では一時ディレクトリの権限で見えていなかった）。期待値だけを直し、該当testを再実行して1 passed（`.validation-fedem/rerun-variants-1.xml`）。全体の再実行はしていない。全pytestが前回（約40分）より長いのは、slowの全体run 6件（単独で約28分）を持つ新しいtestファイルが`--dist loadfile`で1 workerに載るためと推定する（ファイルごとの所要時間は測っていない）。
