# Implementation Plan

- [ ] 1. 構成と挙動とtestを足す
  - 要求2.1〜2.3のtestを先に書き、REDを確かめてから、状態遷移の部品、設定の項目（割り当ての方式、学習の窓、DriftSurfの設定）、構成、掃引のオプション、runtimeの分岐（役割のモデルの登録と作り直し、clientの時刻の標本の保管と、学習の直前に窓の標本だけを学習の標本の集合へ入れる接続（design.mdの「窓と既存のstoreの接続」）、窓の標本数による集約の重み、clientの正解数の報告、サーバの状態遷移、昇格、予測の役割の配布、検出・作成・統合を行わない）、DriftSurfの記録を足す。Oracle・IFCA・FedAvg+Window／Adaptive-FedAvgで足した項目と分岐を再利用する。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・FedAvg+Window・Adaptive-FedAvgのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4_
  - _Depends:_ FedAvg+Window／Adaptive-FedAvgのspecの実装（`f89a113b`）
- [ ] 2. 文書と検証の記録
  - 要求3.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 調整・本規模の実行と集計
  - commit・pushの後、主担当が内部確認をしてδの調整のコマンドを渡し、δを選んでから本規模のコマンドを渡す。結果を表1の手法と対で集計し、実験の台帳と論文のnotesに記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes
