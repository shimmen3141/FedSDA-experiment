# Implementation Plan

- [ ] 1. 構成と学習の経路とtestを足す
  - 要求3.1〜3.3のtestを先に書き、REDを確かめてから、構成`fedsda-soft-training`と`fedsda-training-mass-diagnostics`、学習の量のコンセプト別の記録の設定の項目（既定は記録しない）、設定の項目（`fedsda`のclientの設定の学習の割り当ての方式。既定`single_model_assignment`は保存から省き、新しい値`responsibility_weighted_training`。`experiment_spec/options.py`のOptionSpec・ActivationRule。構成の定数は`sweep/experiment_configurations.py`の既存の変種と同じ形。design.mdの「設定の登録」）、共有の学習ストアと責任度、重み付きの共同更新、集約の重み、統合での責任度の列の扱い、柔らかい学習の量のコンセプト別の記録、純度の導出を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界・スキーマのtest、`fedsda`の変種のtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`、`tests/test_proposed_regression.py`が通る。Ruff・Pyright。新しい全体runのtestの所要時間を記録し、重いものは`slow`と`tests/conftest.py`の配り方へ登録する。
  - _Requirements: 1.1〜1.6, 2.1, 3.1〜3.4_
- [ ] 2. 文書と検証の記録
  - 構成の一覧、設定参照の再生成（`python -m tools.generate_option_docs`）、naming.mdの追記。関係するtestの結果をImplementation Notesに記録する。
  - 主担当: 独立CLIの実装レビュー（GPT-6 Luna）を起動し、判定・指摘・採否をImplementation Notesとspec.jsonに記録する。全pytest（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）を流して結果を記録する。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 本規模の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、本規模の実行（2構成×200 run＝400 run）のコマンドを渡す。design.mdの集計をユーザーに示し、実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2

## Implementation Notes
