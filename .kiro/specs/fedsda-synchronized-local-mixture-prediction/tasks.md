# Implementation Plan

- [x] 1. 実装（1つのまとまりで行い、終わった時点でtestが通る）
  - [x] 1.1 混ぜる重みのownerと、そのtestを先に書く（設計のValidation Designの1）。依存境界の許可。
    - _Requirements: 1.3_
  - [x] 1.2 前提: 同期の時点の予測の構成と、混ぜる構成とで、`combined_prediction_is_correct`が違う標本が出る小さい条件を、下調べで選ぶ（検出力のtestの条件）。設定の値と組合せの検査、予測の関数と標本の処理の引数、clientの接続、構成名と登録。testを先に書く（Validation Designの2〜5）。設定の検査のtestの許す値、構成の一覧を使うtest、依存境界の許可を同じ変更で合わせる。
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 2.1, 2.2, 3.1_

- [x] 2. 文書
  - 実験の入口の文書（構成の表）、研究バックログの8、再開案内、完了specの一覧。
  - _Requirements: 4.1_

- [x] 3. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright・WSLの対象test。
  - _Requirements: 全部_

## Implementation Notes

- 手順の逸脱: 混ぜる重みのownerのtestは先に書いたが、予測の関数・client・構成の結合testは、実装の後に書いた。代わりに、主な処理を1つずつ壊して、新しいtestが落ちることを確かめた（損失の式の簡略化、混ぜる重みを更新しない、分け合いを外す、記録の正否を混ぜた確率にしない、clientが混ぜる重みを渡さない。5つとも1件以上失敗。元に戻した）。
- 検出力の条件: 同期の時点の予測のtestと同じ小さい条件（sine2、client 3、標本400、間隔50、seed 1）で、混ぜた予測の正否が同期の時点の予測と違う標本が出た（Fixed-Share・AdaHedge、容量30・60の3通りとも）。
- 予測の関数の型の検査は、写しのownerの検査と同じく、ほかの入力の検査より前に置いた。
- 実装レビュー（独立Haiku 5.5 medium、1回）: APPROVED（Minor 4）。1: 現在のモデルだけの経路での拒否のtestを足した。2: 研究バックログの、照合した組合せの書き方を直した。3: 依存境界の許可の`math`・`torch`（module名）は残した（このtestは`from math import exp`をmodule名としても数えるので、写しのownerのmoduleの許可と同じ扱い）。4: WSLの結果を下に書いた。
- 検証: 全pytest 14311 passed / 8 skipped、失敗0（JUnit `../../venv/refactoring-tests/mixture-full.xml`）、Ruff・Pyright 0件。WSLで、新しいtest・同期の時点の予測のtest・goldenの版1〜5の244件が通った。レビューの後の修正は、test 1件の追加と文書だけ。
