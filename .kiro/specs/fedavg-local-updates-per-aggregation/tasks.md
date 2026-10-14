# Implementation Plan

- [x] 1. 構成とtest
  - [x] 1.1 goldenの事例の一覧を、今の13構成の名前で固定し、golden以外のtestで守る構成の一覧を、空の辞書で置く。事例の一覧の検査を合わせ、構成の一覧の全部を回して`golden.build_case_settings`を呼ぶ2つのtest（待つ登録と警報、早い標本の解放）を、golden側の13構成で回すように変える（構成を足す前に、今の値で通ることを確かめる）。
    - _Requirements: 2.1, 2.2_
  - [x] 1.2 新しいtestを先に書き（設定、振る舞い、検出力、コマンド）、構成名と組み立てを足す。同じ変更で、golden以外のtestで守る構成の一覧へ新しい名前と専用のtestのファイルを入れ、構成の一覧の完全なtupleと`CONFIGURATION_VARIANTS`の除外（`test_fedsda_configuration_variants.py`）へ新しい名前を足す。完了の条件: 新しいtest、構成の一覧を使う3つのtest、goldenの版1〜5、`test_experiment_sweep.py`が通り、`fedavg`のrunの識別子が変わらない。
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.3_

- [x] 2. 文書
  - 実験の入口の文書（構成の表と、FedAvgの節）、研究バックログの8、再開案内。
  - _Requirements: 3.1_

- [x] 3. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright。
  - _Requirements: 全部_

## 仕様レビュー3回目（PASS）のMinorの扱い（実装で行う）

- 1.2の完了条件へ、依存境界のtestと、並列の組分けのtest（`test_fedsda_configuration_variants.py`の`test_parallel_scope_…`）を含める。
- 一覧の表は構成のディレクトリごとに`run_summary.csv`が分かれる（同じ表に`fedavg`と新しい構成は入らない）。列は足さない。設計の記述を直す。
- 振る舞いのtestへ、集約間隔をbatchの件数より小さくした条件（例: 20。最初のroundは累積の保持標本が足りず、更新が捨てられる）を足す。
- `experiment_configurations.py`のdocstringの「提案構成以外の構成は…1つの要素だけを置き換える」へ、FedDrift・FedAvgの構成を除く旨を足す。
- 未使用になるimportを消す。別のmoduleから使う名前は`_`を付けない（`BASELINE_CONFIGURATION_CASE_NAMES`）。naming.mdを合わせる。
- 検出力のtestは、判定の条件（seed・標本数）を固定して書く。所要時間が長ければ、関数に`slow`の印を付ける。

## Implementation Notes

- 手順: 新しいtestを先に書き、構成が無い状態で収集の段で失敗することを確かめてから、構成を足した。検出力は、同じ検査を`fedavg`（1標本ごとの局所更新）に当てると食い違いが出るtestで確かめた。
- goldenの事例の一覧は、基準版1〜5の13構成（`BASELINE_CONFIGURATION_CASE_NAMES`）で固定した。事例の集合と順は変えていない。後から足す構成は`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ、専用のtestのファイルとともに登録する（登録漏れは`test_case_registry_preserves_all_registered_configurations`が検出する）。
- 仕様レビュー3回目のMinor: 依存境界と並列の組分けのtestは、全pytestと実装レビューで通った。batch未満のroundは、集約間隔20の条件で確かめた。docstring・未使用のimport・名前（`_`を付けない）・設計の表の記述は直した。新しいtestは6件で約6秒のため、`slow`の印は付けていない。
- 実装レビュー（独立Haiku 5.5 medium、1回）: APPROVED（Minor 3）。研究バックログのFedSDA側の文を「ユーザーの案（実装は未着手）」に直し、実験の入口の文書のFedDriftの時機の記述へ根拠を足した。`fedavg`の識別子を固定値と比べるtestの追加（任意）は見送った（`fedavg`の設定と組み立ては変えておらず、`test_fedavg_experiment_sweep.py`の既存の識別子の期待も通っている）。
- 検証: 全pytest 14261 passed / 8 skipped（失敗0。JUnit `../../venv/refactoring-tests/fedavg-per-aggregation-full.xml`）、Ruff成功、Pyright 0件。レビューの後の修正は文書だけ。
