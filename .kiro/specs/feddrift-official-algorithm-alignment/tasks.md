# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4 → task 5 → task 6。実装はGPT-6.1-sol（task 1〜3、task 5の文書の下書き）。task 4のgoldenと、task 6の確認は主担当。

- [ ] 1. 設定の項目を足し、版1〜13の今の値を明示する（既定は、まだ変えない）
  - 設定の検査のtest（不正な値・型、Validation Designの(vi)）を先に書き、REDを確かめる。
  - naming.mdの項目と値、検査（拒否する組合せ）。全部の`FeddriftRunClientSettings`・`FeddriftServerSettings`の組立て（既定、構成、旧と比べる入口、golden case、test・fresh script）で今の値を明示する。
  - δのデータ別の既定の対応表を置く（まだ既定にはしない）。
  - 完了条件: 新しいtestが通り、既存のtest（旧との照合、版1〜13のgolden）が値を変えずに通る。
  - _Requirements: 1.4, 2.3, 4.2, 5.1, 7.3_

- [ ] 2. clientの検出の塊の判定（D01・D02・D03）
  - Validation Designの(i)(ii)のtestを先に書き、REDを確かめてから実装する。
  - 誤り率の評価、最小のモデルの選択と検知、複写元、基準値。誤り率の予測の規則は今の予測の関数に合わせ、根拠を記録する。
  - 完了条件: (i)(ii)が通り、今の値の設定の既存のtestが値を変えずに通る。
  - _Requirements: 1.1, 1.2, 2.1, 2.2_

- [ ] 3. サーバ側（D01の距離・D04・D08・D07の上限）
  - Validation Designの(iii)(iv)(v)(vii)(x)のtestを先に書き、REDを確かめてから実装する。
  - クロス評価の誤りの数の集計と距離（0での下限）、今の時刻に割り当てたモデルの集合と学習・集約の対象、統合の重み。集合の伝達と通信の計数の判断を記録する。
  - 完了条件: (iii)(iv)(v)(vii)(x)が通り、今の値の設定の既存のtestが値を変えずに通る。
  - _Requirements: 1.3, 3.1, 3.2, 3.3, 3.5, 4.1, 5.2, 6.1, 6.2_

- [ ] 4. 既定を替え、基準版14を足す
  - 既定のFedDriftの設定（新しい値の全部、上限＝clientの数、δのデータ別の既定）と、掃引で閾値を指定しないときの既定のtest（(ix)）を先に書き、REDを確かめてから既定を替える。
  - `BEHAVIOR_BASELINE_VERSION = 14`、runの設定の記録の新しい項目。既定に依るtest（runの識別子ほか）を直す。
  - goldenの版14（Windows・Linux）は主担当が作り、版14のFedSDA・FedAvgのcaseが版13と同じ値であることを確かめる（(viii)）。
  - 完了条件: 全部の対象のtestと版1〜14のgoldenが通る。
  - _Requirements: 1.5, 1.6, 6.1, 7.1, 7.2, 7.4_

- [ ] 5. 文書
  - `docs/experiments/differences-from-feddrift.md`を書き直す（そろえた差・開示する差・根拠・監査と論文の参照）。`docs/experiments/refactoring-baseline.md`の版の表に版14。
    `docs/reference/options.md`は直接編集しない（必要なら生成の道具で再生成する）。
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 6. 確認（主担当）
  - 全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回（文書だけの修正は再確認を要しない）。
