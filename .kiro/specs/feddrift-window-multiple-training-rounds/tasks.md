# Implementation Plan

- [ ] 1. 1時刻のround数の設定と、CFLの時刻の途中の分割を足す
  - 要求3のtest（既定の不変、掃引と制約、FedAvg+Window $R=3$ の小さい全体run、CFLの $R=3$ のサーバの経路、CFLの $R=3$ の小さい全体run）を先に書き、REDを確かめてから、掃引の値・設定の登録・組み立ての検査・CFLの分割の後の標本の付け替え・roundごとの対象の集合・記録の列`round_index`（保存形式の版8）を足す。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・Window・Adaptive-FedAvg・DriftSurf・CFLのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_
- [ ] 2. 文書と検証の記録
  - 要求4.1の文書（設定の参照の再生成、`differences-from-feddrift.md`のD11とCFL+Windowの節、構成の一覧）とnaming.mdの追記。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、予備の実行（96 run）のコマンドを渡す。design.mdの「予備の実行」の集計をユーザーに示し、本規模へ進むかと $R$ の決め方をユーザーが決める。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2

## Implementation Notes
