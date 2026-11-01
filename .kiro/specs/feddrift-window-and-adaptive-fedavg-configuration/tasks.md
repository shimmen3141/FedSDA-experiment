# Implementation Plan

- [ ] 1. 構成と挙動とtestを足す
  - 要求4.1〜4.3のtestを先に書き、REDを確かめてから、学習率の更新の部品、設定の項目（単一のモデルの割り当ての方式、学習率の調整の設定とSGD）、2構成、掃引のオプション、runtimeの分岐（1個のモデルの登録、全標本のID 0への帰属、学習の窓、Adaptive-FedAvgのSGDと時刻ごとの学習率、集約の後の学習率の更新、検出・作成・統合を行わない）、学習率の記録を足す。Oracle・IFCAで足した項目と分岐を再利用する。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCAのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4_
  - _Depends:_ IFCAのspecの実装（`.kiro/specs/feddrift-ifca-configuration/`のTask 1、`7874c449`）
- [ ] 2. 文書と検証の記録
  - 要求5.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 5.1_
  - _Depends:_ 1
- [ ] 3. 調整・本規模の実行と集計
  - commit・pushの後、主担当が内部確認をして、Windowの本規模とAdaptive-FedAvgの調整のコマンドを渡し、$\eta_0$ を選んでからAdaptive-FedAvgの本規模のコマンドを渡す。結果をFedHASP・FedDrift・FedAvg・Oracle・IFCAと対で集計し、実験の台帳に記録する。
  - _Requirements: 5.2_
  - _Depends:_ 2

## Implementation Notes
