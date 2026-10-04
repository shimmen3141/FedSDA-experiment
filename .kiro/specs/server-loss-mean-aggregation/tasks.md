# 実装タスク
既存venv/pytestを使い1→2→3の順で実行。保存前の独立task graphレビュー後に承認する。

- [x] 1. サーバ向けの件数加重平均を実装する
  - TDDで空/zero/単一/複数/順序・丸め/非零M2から、旧Baseと共有バックボーンサーバ両方へ全fieldを照合する。
  - 上位でモデル所有・正training件数・stats有無を明示選別し、training件数と統計nが異なるケースも直接照合する。
  - 完了時、正の合計nで旧平均/M2zero、空・全zeroでNoneとなるpure関数があり、参加判定/モデル操作/サーバ状態を持たない。
  - _Boundary: server loss mean aggregation_
  - _Requirements: 1.1, 1.2, 1.3, 3.1_

- [x] 2. 拒否・独立性とサーバ補完への接続を検証する
  - exact型/後段forgedfield/算出overflow・範囲外を検査し、入力非変更/frozen/別結果独立/RNG/grad/default/keywordを確認する。
  - LEGACY008の2極大入力を旧/新で直接対照し、旧未修正/通常影響未確認を記録する。
  - 完了時、Noneでは既存whole serverrecord保持、正nではclassを空にしたwhole置換、ID補完→store次更新→baselineへ明示接続できる。
  - _Depends: 1_
  - _Boundary: test integration_
  - _Requirements: 2.1, 2.2, 2.3, 3.2_

- [x] 3. 依存境界と全回帰・独立起動を統合検証する
  - exact moments module/型だけをASTで許可し、同module別関数/private/子module/別module/上位/旧/torch/NumPyを禁止注入する。
  - 完了時、全tests旧11/最終3golden不変、stdlib -S freshsmoke、全9条件/roadmap/LEGACY008状態、Luna最終GOの証拠が揃う。
  - _Depends: 1, 2_
  - _Boundary: final integration_
  - _Requirements: 3.3_

## Implementation Notes
参加・保存・classclearは上位testで明示する。trainingcountは重みではない。極大入力の拒否を通常数値のアルゴリズム変更や旧server修正と扱わない。
