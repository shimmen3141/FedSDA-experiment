# Implementation Plan

- [ ] 1. 構成と挙動とtestを足す
  - 要求2.1〜2.3のtest（分割の判定の部品、分割が決定的に起きる入力でのサーバの集約の経路、小さい全体run）を先に書き、REDを確かめてから、設定の項目（割り当ての方式の新しい値、$\gamma$）、構成、サーバの分割の判定と集約の省略、割り当ての保持、記録を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・DriftSurfのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4_
- [ ] 2. 文書と検証の記録
  - 要求3.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と本規模の実行と集計
  - commit・pushの後、主担当が内部で計画（run数と重複先）を確かめ、予備の実行（24 run）のコマンドを渡す。分割の回数・時刻・モデルの数・精度を集計してユーザーに示し、本規模へ進むかをユーザーが決める。進むなら本規模（400 run）のコマンドを渡す。結果を論文の表1の手法と対で集計し、開始と結果を実験の台帳に記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes
