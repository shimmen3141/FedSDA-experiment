# Implementation Plan

依存: 1.1・1.2・1.3（互いに独立） → 2.1 → 2.2と3.1（互いに独立） → 3.2。

- [x] 1. 生成器
- [x] 1.1 STAGGERの生成器 (P)
  - 要求1・4の定義のtest（値の範囲、ラベルの規則、尺度、再現性、拒否、構築で乱数を使わない）を先に書き、REDを確かめてから`data/stagger/`を実装する。
  - 完了条件: 新しいtestが通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.3_
  - _Boundary: StaggerSampleGenerator_
- [x] 1.2 Agrawalの生成器 (P)
  - 同じ形のtestを先に書く。摂動の後の値が設計書の範囲に収まること、hvalueが`(9 - zipcode) * 100000`か0になること、commissionがsalary 75000以上で0になることを含める。
  - 完了条件: 新しいtestが通る。
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3_
  - _Boundary: AgrawalSampleGenerator_
- [x] 1.3 Random RBFの生成器 (P)
  - 同じ形のtestを先に書く。中心がrunのseedに依らないこと、ラベルが選んだ中心のラベルであることを含める。
  - 完了条件: 新しいtestが通る。
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3_
  - _Boundary: RandomRbfSampleGenerator_

- [x] 2. 製品への接続
- [x] 2.1 定義・実行条件・生成器の表
  - データ名の一覧を完全一致で比べる既存のtest（`test_dataset_definitions.py`、`test_run_settings_validation.py`の`allowed_parameter_values`、`test_synthetic_sample_generation.py`の生成器の表）と、`is_observed_sample_generator_of_dataset`のtestを先に直し、3データを`dataset_definitions.py`・`experiment_run_conditions.py`・`observed_sample_generation.py`へ足す。
  - 依存境界（`test_single_run_dependency_boundaries.py`）: 3packageの`__init__.py`（説明だけ）を`package_boundary_paths`とpackageごとの一覧へ足し、3moduleの許可集合を`sea_sample_generation`と同じ形で登録する。
  - 完了条件: 定義・生成器の表・依存境界のtestが通り、既存のデータのtestが値を変えずに通る。
  - _Requirements: 1.1, 2.1, 3.1, 4.4_
  - _Depends: 1.1, 1.2, 1.3_
- [x] 2.2 全手法の設定・掃引・全体run
  - 要求7.2の(iii)〜(v)のtestを先に書き、FedSDA・FedDriftの表に3データを足す。`test_fedavg_experiment_run.py`・`test_feddrift_experiment_sweep.py`の実旧との照合の一覧は変えない（設計書のTesting Strategy）。
  - 共用script`tests/refactoring/fresh_process_smoke.py`へ、3データのFedSDAの全体runの流れを足す。
  - 完了条件: 4手法（FedSDA・FedDrift・FedAvg・DWM）の設定のtest、全体run（各データ1 runずつ）、掃引の計画のtest、fresh smokeが通り、全版のgoldenが通る。
  - _Requirements: 5.1, 5.2, 5.3, 7.1, 7.2_
  - _Depends: 2.1_

- [x] 3. 単一ストリームの版と文書
- [x] 3.1 単一ストリームの名前の検査とRiverとの照合
  - 名前の一致の検査のtestを先に書き（既存の`test_river_synthetic_recurring_streams.py`の「製品の定義が無い」testを替える）、`river_synthetic_recurring_streams.py`の検査を替える。`test_single_stream_metrics_and_input.py`の`stagger3`を製品の外の名前として使うtestを、製品に無い名前`unknown_recurring_stream3`へ替える（`stream.dataset_name`の確認も合わせて替える）。
  - 要求6.2の照合のtest（(i)〜(iii)の全て。`importorskip("river")`、torchをimportしない）を足し、WSLの`~/river-env`で実行する。
  - 完了条件: Windowsでは照合のtestがskipの理由を出し、WSLで照合のtestと既存の単一ストリームのtestが通る。
  - _Requirements: 6.1, 6.2_
  - _Depends: 2.1_
- [x] 3.2 文書と確認
  - `docs/experiments/dataset-characteristics.md`へ3データを足す（Random RBFの実測の範囲を含む）。`python -m tools.generate_option_docs`を実行し、`docs/reference/options.md`に差が出れば再生成の結果を含める。全pytest、ruff、pyright。独立CLIの実装レビュー1回。
  - 完了条件: 文書に要求8.1の項目があり、全pytestが通り、実装レビューがPASSである。
  - _Requirements: 8.1, 8.2, 7.1_
  - _Depends: 2.2, 3.1_

## Implementation Notes

### 仕様書群のレビュー（2026-11-20）
- 3回。全てClaude Haiku 5.5（effort medium、読むだけ）。GPT-6 Lunaは使用量の上限で使えなかった。session・判定はspec.json。
- 1回目（Major 6）: Agrawalの照合を摂動なしで行う、既存の単一ストリームのtest2件・製品のデータ名の一覧のtest・依存境界・fresh smokeを範囲に入れる、照合の許容を差の標準誤差の4倍にする。全部採用。
- 2回目（Major 1）: DWMを対象に含める。Minorとして尺度の理由、δの根拠（1274行・455行のFedDriftの表、274行はDriftSurf）、重みの選択の逐次の累積和、多重比較の扱いなど。全部採用。
- 3回目（PASS、Minor 6）: STAGGERの照合は27通りを直接、重みの選択の端、中心の重みの対応、既存の一覧のtestの扱いを先に決める、単一ストリームのtestの名前の確認。5件を文書で直した。
  不採用1件: 「直書きのtupleと`DEFINED_DATASET_NAMES`の一致を確かめるtestが無い」は事実と違う（`tests/refactoring/test_dataset_definitions.py`の30行が`allowed_parameter_values == DEFINED_DATASET_NAMES`を確かめる）。

### 実装（2026-11-20、主担当）
- commit: 生成器`3f3e71d1`、接続`a4d78e53`、設定・全体run`069cdc70`、単一ストリーム`18eb778e`、文書`bf05abec`、レビューの修正`c06df943`。
- testの新しいmoduleは収集の失敗をREDとしない規約のため、task 1.1〜1.3のREDは記録していない。task 2.1・2.2・3.1は既存のtestを先に直して失敗を確かめた。
- Riverとの照合（WSL `~/river-env`、River 0.27.0）: 22件が通過（STAGGERの27通り×3概念、Agrawalの摂動なし1万標本×4概念、Random RBFの全中心、11概念の分布）。単一ストリームのtest全体は124 passed・3 skipped。保存済みの系列6本（3データ×seed 0・7）を再生成し、hashが一致した。
- fresh smoke: `stagger3`・`agrawal4`を足した。`randomrbf4`は、smokeの極小条件（事前学習40標本）では偶然の正解率を超える判定が成り立たない（seed 7で0.5以下）ので入れていない。10 client×各200標本のFedAvgで精度0.846、各1000標本ではFedSDA 0.821・FedAvg 0.797（seed 0の1 runずつの動作確認）。全体runの完了は`test_recurring_synthetic_dataset_runs.py`（4手法×3データ）が確かめる。
- 依存境界: SEAの生成器と同じく、新しい3moduleは個別の許可集合を持たずdata層の共通の規則で通る。package一覧と`observed_sample_generation.py`の許可に足した。
- `docs/reference/options.md`は再生成して差が出なかった。
- 全pytest（`-n 12 --dist loadfile`、`bf05abec`）: 13,793 passed・23 skipped・2 errors。2 errorsは`test_feddrift_cfl_window_configuration.py`・`test_feddrift_fedccfa_configuration.py`の収集時の`ImportError`（sklearnの`_tree`のDLLの読み込みがアプリケーション制御のポリシーで拒否）。環境の制約で、この変更と関係しない（`dwm-neural-network-baseline`でも同じ種類の1件を記録）。`c06df943`は文書だけなので取り直していない。

### 実装レビュー（2026-11-20）
- Claude Haiku 5.5（effort medium）。対象test 3,728 passed、ruff、`spec_checks.py names`を独立に実行。WSLのtest・全pytest・fresh smokeは主担当の実測を照合。
- 指摘1件（Major、文書だけ）: Random RBFの実測の範囲が再現しない。主担当は1つの生成器で概念0〜3を続けて生成しており、概念ごとに作り直す手順と値が違った。手順を明記して測り直した値に直した（`c06df943`）。文書だけの修正なので、規約により再確認は要さない。
