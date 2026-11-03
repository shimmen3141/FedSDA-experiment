# Implementation Plan

- [ ] 1. 構成と挙動とtestを足す
  - 要求3.1〜3.3のtestを先に書き、REDを確かめてから、設定の項目（割り当ての方式の新しい値、学習の標本の範囲の新しい値）、4構成、runtimeの分岐（E-step、重み付きのM-step、全モデルの集約、$\pi_c$ による予測の混合）、round数の掃引の値の許可、記録の表（保存形式の版を上げる）を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、IFCA・Window・CFL・FedDrift・Oracleのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`が通る。Ruff・Pyright。
  - _Requirements: 1.1〜1.9, 2.1, 3.1〜3.4_
- [ ] 2. 文書と検証の記録
  - 要求4.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）とnaming.mdの追記。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、予備の実行（96 run）のコマンドを渡す。design.mdの集計をユーザーに示し、本規模に使う構成とround数をユーザーが決める。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2

## Implementation Notes
