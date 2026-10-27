# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4。実装はGPT-6.1-sol（task 1〜4）。確認・全pytest・レビューの起動は主担当。

- [ ] 1. 読み込みと生成器
  - 要求4.2の(i)〜(iv)のtestを先に書き、REDを確かめてから実装する。設計書の「回転の対応の確認」に、公式のcommit `18e7fbe`の該当の行を書き足す。
  - 完了条件: 新しいtestが通り、既存の`mnist2`・`mnist4`のtestと版1〜14のgoldenが値を変えずに通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

- [ ] 2. 全手法の設定と掃引
  - 要求4.2の(v)(vi)(vii)のtestを先に書き、REDを確かめてから実装する。
  - 完了条件: 新しいtestが通り、既存の設定のtestと版1〜14のgoldenが通る。
  - _Requirements: 3.1, 3.2, 3.3, 4.1_

- [ ] 3. 文書
  - 完了条件: データの説明に4データの定義・根拠の実装（公式FedDriftのファイルとcommit、IFCA・FedCCFAの論文）・開示する差があり、Fashion-MNISTの取得の手順（URLとMD5の照合）があり、
    `docs/reference/options.md`が直接編集されていない（必要なら再生成されている）。
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 4. 確認（主担当）
  - 全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回。
