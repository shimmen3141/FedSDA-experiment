# Implementation Plan

依存: 1.1・1.2・1.3（互いに独立） → 2.1 → 2.2と3.1（互いに独立） → 3.2。

- [ ] 1. 生成器
- [ ] 1.1 STAGGERの生成器 (P)
  - 要求1・4の定義のtest（値の範囲、ラベルの規則、尺度、再現性、拒否、構築で乱数を使わない）を先に書き、REDを確かめてから`data/stagger/`を実装する。
  - 完了条件: 新しいtestが通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.3_
  - _Boundary: StaggerSampleGenerator_
- [ ] 1.2 Agrawalの生成器 (P)
  - 同じ形のtestを先に書く。摂動の後の値が設計書の範囲に収まること、hvalueが`(9 - zipcode) * 100000`か0になること、commissionがsalary 75000以上で0になることを含める。
  - 完了条件: 新しいtestが通る。
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3_
  - _Boundary: AgrawalSampleGenerator_
- [ ] 1.3 Random RBFの生成器 (P)
  - 同じ形のtestを先に書く。中心がrunのseedに依らないこと、ラベルが選んだ中心のラベルであることを含める。
  - 完了条件: 新しいtestが通る。
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3_
  - _Boundary: RandomRbfSampleGenerator_

- [ ] 2. 製品への接続
- [ ] 2.1 定義・実行条件・生成器の表
  - データ名の一覧を完全一致で比べる既存のtest（`test_dataset_definitions.py`、`test_run_settings_validation.py`の`allowed_parameter_values`、`test_synthetic_sample_generation.py`の生成器の表）と、`is_observed_sample_generator_of_dataset`のtestを先に直し、3データを`dataset_definitions.py`・`experiment_run_conditions.py`・`observed_sample_generation.py`へ足す。
  - 依存境界（`test_single_run_dependency_boundaries.py`）: 3packageの`__init__.py`（説明だけ）を`package_boundary_paths`とpackageごとの一覧へ足し、3moduleの許可集合を`sea_sample_generation`と同じ形で登録する。
  - 完了条件: 定義・生成器の表・依存境界のtestが通り、既存のデータのtestが値を変えずに通る。
  - _Requirements: 1.1, 2.1, 3.1, 4.4_
  - _Depends: 1.1, 1.2, 1.3_
- [ ] 2.2 全手法の設定・掃引・全体run
  - 要求7.2の(iii)〜(v)のtestを先に書き、FedSDA・FedDriftの表に3データを足す。データ名を固定の一覧で持つ既存のtest（`test_fedavg_experiment_run.py`、`test_feddrift_experiment_sweep.py`、`fresh_process_smoke.py`）は、設計書のTesting Strategyの規則で扱う。
  - 共用script`tests/refactoring/fresh_process_smoke.py`へ、3データのFedSDAの全体runの流れを足す。
  - 完了条件: 4手法（FedSDA・FedDrift・FedAvg・DWM）の設定のtest、全体run（各データ1 runずつ）、掃引の計画のtest、fresh smokeが通り、全版のgoldenが通る。
  - _Requirements: 5.1, 5.2, 5.3, 7.1, 7.2_
  - _Depends: 2.1_

- [ ] 3. 単一ストリームの版と文書
- [ ] 3.1 単一ストリームの名前の検査とRiverとの照合
  - 名前の一致の検査のtestを先に書き（既存の`test_river_synthetic_recurring_streams.py`の「製品の定義が無い」testを替える）、`river_synthetic_recurring_streams.py`の検査を替える。`test_single_stream_metrics_and_input.py`の`stagger3`を製品の外の名前として使うtestを、製品に無い名前`unknown_recurring_stream3`へ替える。
  - 要求6.2の照合のtest（(i)〜(iii)の全て。`importorskip("river")`、torchをimportしない）を足し、WSLの`~/river-env`で実行する。
  - 完了条件: Windowsでは照合のtestがskipの理由を出し、WSLで照合のtestと既存の単一ストリームのtestが通る。
  - _Requirements: 6.1, 6.2_
  - _Depends: 2.1_
- [ ] 3.2 文書と確認
  - `docs/experiments/dataset-characteristics.md`へ3データを足す（Random RBFの実測の範囲を含む）。`python -m tools.generate_option_docs`を実行し、`docs/reference/options.md`に差が出れば再生成の結果を含める。全pytest、ruff、pyright。独立CLIの実装レビュー1回。
  - 完了条件: 文書に要求8.1の項目があり、全pytestが通り、実装レビューがPASSである。
  - _Requirements: 8.1, 8.2, 7.1_
  - _Depends: 2.2, 3.1_
