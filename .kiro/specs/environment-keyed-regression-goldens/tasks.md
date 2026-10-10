# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。

- [ ] 1. 環境ごとのgolden
- [ ] 1.1 選択・名前・作成の規則と、そのtest
  - 一時ディレクトリでの、選択（一致、なし、重なり）、名前、作成の拒否（固定のgoldenの環境、上書き）のtestを先に書く。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 3.2, 3.3_

- [ ] 1.2 既存のLinux用のgolden（WSL）を移し、全goldenの検査と、旧実装の照合をつなぐ
  - 完了: WSLで、旧実装の照合が成功する。Windowsの基準環境で、全goldenの検査が成功し、旧実装の照合が、既存の回帰testに任せてskipされる。
  - _Depends: 1.1_
  - _Requirements: 1.4, 2.2, 2.3, 2.4_

- [ ] 2. 新実装の照合
- [ ] 2.1 goldenの照合のtestが、環境の記録でgoldenを選ぶ
  - 完了: WSLとWindowsの基準環境で、3ケースの照合が成功する（skipなし）。
  - _Depends: 1.2_
  - _Requirements: 2.2, 2.3_

- [ ] 3. 文書と検証
- [ ] 3.1 基準環境の文書を、環境ごとのgoldenの説明へ書き直す（研究室サーバでgoldenを足す手順を含む）
  - _Requirements: 4.1_

- [ ] 3.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: Windowsの全pytest、Ruff、Pyright、`pip check`、照合script（`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とWindows用のgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.1, 3.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
