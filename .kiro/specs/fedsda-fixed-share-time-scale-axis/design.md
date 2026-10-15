# Design Document

## Overview
2か所の同値の検査（`FedsdaRunClientSettings.__post_init__`と`validate_experiment_run_settings`）を外す。掃引の計画（`ExperimentSweepPlan`）へ`fedsda_fixed_share_time_scales: tuple[int, ...] | None = None`を足し、FedSDAのrun設定の変種を、容量の軸とTの軸の組で作る。容量の軸はTを替えなくなる。コマンドの引数、一覧の表の列、掃引の記録の形式の版を足す。帰属の保留・Fixed-Share・混ぜる予測の処理は変えない（混ぜる重みの分け合いの割合は、今どおり`fedsda_run_client.py`で1/T）。

## Boundary Commitments
### This Spec Owns
- `runtime/fedsda_run_client_settings.py`: 同値の検査を消す（`pending_capacity`は、すぐ後の警報を待つ扱いの検査が使うので、変数は残す）。
- `configuration/run_settings_validation.py`: 同値の検査を消す（この検査だけが使う`cast`・型の読込みが不要になれば、それも消す）。
- `runtime/fedsda_final_configuration_run_settings.py`: Tに自分の定数`_FIXED_SHARE_WEIGHT_REDISTRIBUTION_TIME_SCALE_SAMPLES = 30`を持たせ、容量の定数を使い回さない。見出しのコメント（112行「Fixed-Shareの時間尺度は、FIFOの容量と同じ値」）を直す。値は同じなので、run設定と識別子は変わらない。
- `sweep/experiment_sweep_plan.py`の容量の項目のコメント（53〜54行）、`tests/refactoring/test_fedsda_pending_assignment_fifo_capacity_axis.py`の冒頭の説明（1〜3行）とコメント（111行）。
- `sweep/experiment_sweep_plan.py`: 計画の項目と検査（容量の項目のコメントを含む）、変種の作り方（2つの軸の組）、容量の軸がTを替えないこと、FedSDAの構成が無いときの拒否を2つの軸へ。
- `sweep/experiment_sweep_command.py`: 引数`--fedsda-fixed-share-time-scales`（`nargs="+"`、`type=int`、`metavar="TIME_SCALE"`）と、容量の引数の説明文（Tを替えない）。
- `sweep/experiment_sweep_records.py`: 列`fixed_share_time_scale`、`SWEEP_RECORD_FORMAT_VERSION`を7へ。
- testの期待の変更（Requirement 4.1）と、新しいtest `tests/refactoring/test_fedsda_fixed_share_time_scale_axis.py`。
- 文書: `docs/experiments/experiment-sweep.md`、研究バックログの8、IMPROVE-015、再開案内。
### Out of Boundary
帰属の保留・Fixed-Share・混ぜる予測の処理、既定の値、FedDrift・FedAvg、run設定の保存の形。
### Allowed Dependencies
新しい依存は無い。依存境界のtestは変える必要がない見込み（`run_settings_validation.py`から読込みが減るだけ）。
### Revalidation Triggers
設定の検査（受理する組が増える）、掃引の計画のキー（掃引の記録）、一覧の表の列、容量の軸の意味。

## Architecture
### 同値の検査を外す
- `fedsda_run_client_settings.py`の`fixed_share_time_scale != pending_capacity`の`RunSettingsValidationError`を消す。`fixed_share_time_scale`の変数は、ほかで使わなければ消す。
- `run_settings_validation.py`の、`prediction_combination_settings`と`training_data_assignment_settings`の値を比べる検査（`validate_experiment_run_settings`の最後の検査）を消す。
- 2つの値の個別の検査（型、Tの下限2は`PredictionCombinationSettings`の項目の検査、容量の下限1は`TrainingDataAssignmentSettings`の項目の検査、警報を待つ扱いでの容量の下限はclientの組の検査）は残る。容量1は、警報を待つ扱いの構成では、変化区間の最小件数（5）の検査で拒否され、警報を即時にする構成では受理される（今まではTも1になってTの下限で拒否された）。
- 既存の構成の値は2つとも30のままなので、run設定・識別子・動作は変わらない。

### 計画
- 項目`fedsda_fixed_share_time_scales: tuple[int, ...] | None = None`（容量の軸の次）。`__post_init__`で、`_validate_distinct_values`（builtin int、重複なし。boolは型の検査で拒否）と、全部が1以上を確かめる（`ValueError`）。実効の下限2は、作ったrun設定の検査（`RunSettingsValidationError`）が確かめる。
- 容量の項目のコメントを、「Tは替えない（Tは`fedsda_fixed_share_time_scales`）」へ直す。容量の下限のコメント（101行「実効の下限は2（Fixed-Shareの時間尺度の下限）」）を、「実効の下限は、警報を待つ扱いの構成では変化区間の最小件数（作ったrun設定の検査が確かめる）」へ直す。
- `_build_fedsda_settings_variants`: 変種を`(run_settings,)`から始め、容量の軸があれば各変種の`training_data_assignment_settings.pending_assignment_buffer_capacity_samples`だけを替えた組、Tの軸があればさらに各変種の`prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples`だけを替えた組にする（容量の順の中で、Tの順）。同値の検査が無くなるので、軸ごとに`run_client_settings`を作り直してよい。途中で作る設定も、clientの組の検査（警報を待つ扱いでの容量の下限ほか）を通る値だけになる（容量だけ・Tだけを替えた設定は、それ自体が完全な設定のため）。
- `build_planned_experiment_runs`の分岐の条件を「FedSDAのrun設定で、容量かTの軸がある」へ広げる。どちらかの軸があり、FedSDAのrun設定が1つも無ければ、指定した軸の項目名を含む`ValueError`（両方指定なら両方の名前）。
- 軸が無ければ、今と同じ1つ。構成の値と同じ値を当てた変種は、元の設定と等しいので、識別子も同じ。

### コマンド
- `--fedsda-fixed-share-time-scales`（`nargs="+"`、`type=int`、`metavar="TIME_SCALE"`）。説明: 「FedSDAの構成の、Fixed-Shareの時間尺度T（標本数。予測の重みを分け直す速さ。混ぜる予測の重みの分け合いの割合も1/T）。FedSDAの構成を選んだときだけ」。計画へ`tuple`で渡す（容量と同じ）。
- 容量の引数の説明から「Fixed-Shareの時間尺度も同じ値にする」を外す。
- 計画の`ValueError`・設定の`RunSettingsValidationError`は、今の仕組みで終了コード2。

Tの軸は、FedSDAの全部の構成に当てる（要求2.1）。重みを使わない構成（`fedsda-single-model-prediction`）や、Fixed-Shareの重みを予測に使わない構成（`fedsda-adahedge-prediction-weights`）では、Tを替えても予測は変わらない（識別子は別なので取り違えはない）。これは文書に書き、計画では除外しない。

### 一覧の表
FedSDAの行に、列`fixed_share_time_scale`を足す。値は、要約の`run_client_settings`→`prediction_combination_settings`→`fixed_share_weight_redistribution_time_scale_samples`を、容量の列と同じく途中も`.get`で読む（無い要約は空欄、行を外さない）。

### 掃引の記録
計画（`asdict(sweep_plan)`）に新しい項目が増えるので、`SWEEP_RECORD_FORMAT_VERSION`を7へ上げる。文書の、記録の形式の版の説明へ「版7から、Fixed-Shareの時間尺度の軸を持ち、容量の軸がTを替えない（版5・6の容量の軸は、Tも同じ値に替えた）」を足す。

## Validation Design
`tests/refactoring/test_fedsda_fixed_share_time_scale_axis.py`:
1. 設定の検査: 既定の`fedsda`の設定で、Tだけを60にした設定、容量だけを60にした設定が、clientの組の検査と全体の設定の検査（`validate_experiment_run_settings`と`ValidatedExperimentRunSettingsSubset`の直接の構築）の両方で受理される。T＝1は、どちらでも`RunSettingsValidationError`（Tの下限）。
2. 計画: Tの軸で、FedSDAのrunがTごとに作られ、Tだけが替わり（容量は構成の値）、それ以外の設定は元と同じ（保存の形で比べる）。FedDrift・FedAvgのrunは増えない。容量の軸では、容量だけが替わり、Tは構成の値のまま。
3. 組: 容量の軸（2値）とTの軸（2値）で、FedSDAの1つの条件から4つのrunが、容量の順の中でTの順に作られる。識別子は全部違う。
4. 既定と同じ: 軸を指定しない計画と、`fedsda_fixed_share_time_scales=(30,)`の計画で、識別子が同じ。
5. 拒否: 重複、空、list、0、負、bool、FedSDAの構成を含まない計画は`ValueError`または`TypeError`（FedSDAの構成が無いときのメッセージに項目名）。T＝1は`RunSettingsValidationError`。コマンドでは終了コード2。
6. 混ぜる予測: 混ぜる予測の構成で、Tの軸60のrunの、clientの混ぜる重みの分け合いの割合が1/60になる（`assemble`の結果で確かめる）。
7. コマンドと表: 小さい掃引で、`--fedsda-pending-assignment-fifo-capacities 30 60 --fedsda-fixed-share-time-scales 60`の2 runが保存され、`run_summary.csv`の`pending_assignment_fifo_capacity`が30・60、`fixed_share_time_scale`が60・60。掃引の記録の形式の版が7で、計画の項目を持つ。

既存のtestの期待の変更（Requirement 4.1）:
- `test_fedsda_run_client_settings.py::test_settings_bundle_requires_fixed_share_time_scale_equal_to_pending_capacity`: 違う値を受理するtestへ変える（名前も変える）。
- `test_run_settings_validation.py::test_fixed_share_time_scale_must_match_assignment_buffer_capacity`（容量1・3・100、T＝2）: 受理へ変える（名前も変える）。この時の容量1は、全体の設定の検査では受理される（警報を待つ扱いの下限はclientの組の検査だけが持つ）ことを、実装の前に確かめ、Implementation Notesへ書く。
- `test_fedsda_pending_assignment_fifo_capacity_axis.py`: 容量の軸でTも替わる期待（`(30, 30), (60, 60)`、`(4, 4)`）を、Tは構成の値（30）のままへ。容量1の拒否の理由（今はTの下限）を、警報を待つ扱いの下限へ直す。
- `test_fedsda_pending_assignment_fifo_capacity_axis.py`の記録の形式の版の期待（211行、6→7）。
- `test_fedsda_synchronized_local_mixture_prediction.py::test_configuration_combines_with_the_pending_fifo_capacity_axis`: Tの期待を30へ。
- `test_run_settings_validation.py::test_direct_subset_construction_uses_the_same_validation`: `training_data_assignment_settings`の容量1・3・100の3つのcase（同値の検査で拒否され、期待名をTへ置き換える分岐を持つ）を外し、その分岐（205〜207行）も消す。違う値を、全体の設定の検査と、直接の構築（`ValidatedExperimentRunSettingsSubset`）の両方が受理することは、新しいtestの1で確かめる。
- `test_experiment_sweep.py`: 記録の形式の版（759行・1464行、6→7）、記録の計画の完全一致（新しい項目`None`）、コマンドの引数の一覧。
- `test_fedsda_default_waited_registration_and_alarm_wait.py`（59〜85行）: コマンドの選択肢の完全な一覧へ、新しい引数を足す（docstringの経緯も）。
- 実装の前に、ほかに同値を前提にするtestが無いかを、`fixed_share_weight_redistribution_time_scale_samples`と`pending_assignment_buffer_capacity_samples`の全参照で確かめる。

全pytest（goldenの版1〜5を含む）・Ruff・Pyright。WSLで、新しいtestと、goldenの版1〜5。

## 旧と違う点
- 設定の検査が、容量とTの違う組を受理する。
- 容量の軸がTを替えない（同じコマンドでも、記録の形式の版6以前とrun設定が違う。識別子は設定の内容から作るので、版6以前の結果を同じrunと取り違えることはない）。
- 既存の構成の動作・値・識別子は変わらない。
