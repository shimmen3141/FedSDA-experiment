# Implementation Plan

- [x] 1. 即時送信可能な新規モデルを既存の保留枠へ登録する
  - 即時状態・置換・snapshot検査のtestを先に書く。
  - 遅延登録の既存契約を保ち、不正入力では前の保留が変わらず、即時保留のtestが成功することを完了条件とする。
  - _Boundary: 送信保留owner_
  - _Requirements: 2.3, 3.2_
- [x] 2. 一バッチの判定から割当・モデル生成・標本保持を接続する
  - 実旧のバッチ処理を代役なしのoracleとしてtestを先に書く。
  - 非検出・新規モデルと統計、複数バッチ、抽出乱数・位置・前回値、部分バッチを照合する。
  - 異常系でownerと乱数が変わらず、正常系の全ownerと生成RNGが実旧と一致するtest成功を完了条件とする。
  - _Boundary: バッチ処理のruntime接続_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2_
- [ ] 3. 新実装だけの実行と依存境界・独立レビュー・全回帰を確認する
  - 依存を登録し、fresh smokeから実際の生成・割当を呼ぶ。
  - 全task後の独立レビューの有用な指摘を反映し、全pytestと品質検査の成功を完了条件とする。
  - _Boundary: 共用実行への接続と検証_
  - _Depends: 2_
  - _Requirements: 3.3_

## Implementation Notes
- requirements/design/tasksのreview gateとtask-graph sanityは主担当inlineでPASS。既存部品のAPIと旧分岐を確認し、9要求を3taskへ割り当てた。共通手順のユーザー委任による段階承認。子エージェントは使わない。task-localレビューはoff、最後の独立CLIを維持する。
- 数値判定の前specは完了済み。本specは全体runやFedDriftの全機能の完了を主張しない。旧early_stoppingの0epochでも生じるrandpermは後続で扱う。
- task 1: 即時操作の欠落で4 failedを確認し、追加後は対象と既存の遅延保留testが71 passed。task 2はtestを先に追加し、test内の遅延importで新moduleの欠落を確認した（収集エラーではないが、数値・判断の失敗を示すREDではない）。実装後に26 passed、連続生成testの旧モデルのbiasの参照名を修正して66 passed。追加の一時ID選択testも含め67件。新規モデル生成・割当・統計・標本・ID・Torch/Python乱数を、実旧__init__と実_detect_and_assign（代役なし）で照合した。
- 接続: runtimeの許可依存をexact検査対象へ登録し、外側依存の拒否3件を追加。共用fresh smokeのFedDrift経路（新規生成・即時送信保留・既存モデルへ切替）は新実装だけで実行成功。共用smoke全体は実行中。Ruff check/format、Pyright 0 errors、pip check成功。全回帰と独立レビューはまだ実施していない。
