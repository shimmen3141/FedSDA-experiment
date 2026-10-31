# Implementation Plan

- [ ] 1. 構成と挙動とtestを足す
  - 要求2.1・2.2のtestを先に書き、REDを確かめてから、設定の項目、構成、runtimeの分岐（$K$ 個のモデルの登録、標本ごとの振り分け、予測のモデル、統合を行わない）、記録を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3_
- [ ] 2. 文書と検証の記録
  - 要求3.1の文書。実装レビュー（独立CLI）と、関係するtestの結果をImplementation Notesに記録する。runtimeを変えるので、全pytestも主担当が流す。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 本規模の実行と集計
  - commit・pushの後、主担当が内部確認をしてコマンドを渡す。結果をFedHASP・FedDrift・FedAvgと対で集計し、実験の台帳に記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes
