# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。Windowsの全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. Linux用のgolden
- [ ] 1.1 照合のtestと、作成の道具
  - 形の検査、Linuxでの照合、Linux以外での作成の拒否のtestを先に書く。
  - 完了: Windowsで、形の検査（goldenを作った後）と、作成の拒否のtestが成功し、Linuxの照合がskipされる。
  - _Requirements: 1.3, 1.4, 2.2, 2.3, 2.4_

- [ ] 1.2 WSL Ubuntuで、Linux用のgoldenを作り、再現を確かめる
  - `--update`で作る。別のprocessで、照合のtestを実行する。
  - 完了: WSL Ubuntuで、旧実装の照合が成功する。形の検査が、WindowsとWSLで成功する。
  - _Depends: 1.1_
  - _Requirements: 1.1, 1.2, 2.1_

- [ ] 2. 新実装の照合
- [ ] 2.1 goldenの照合のtestが、実行環境のgoldenを選ぶ
  - 完了: WSL Ubuntuで、指標の導出のtest（3ケースの、実旧との照合と、Linux用のgoldenとの照合）が成功する。Windowsで、同じtestが成功する。
  - _Depends: 1.2_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 3. 文書と検証
- [ ] 3.1 基準環境の文書へ、Linuxの手順を書く
  - _Requirements: 4.1_

- [ ] 3.2 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: Windowsの全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とWindows用のgoldenの差分が空である。WSLでの実行結果を、「WSLで成功」と区別して記録してある。
  - _Depends: 2.1, 3.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
