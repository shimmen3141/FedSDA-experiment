# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。

- [x] 1. 環境ごとのgolden
- [x] 1.1 選択・名前・作成の規則と、そのtest
  - 一時ディレクトリでの、選択（一致、なし、重なり）、名前、作成の拒否（固定のgoldenの環境、上書き）のtestを先に書く。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 3.2, 3.3_

- [x] 1.2 既存のLinux用のgolden（WSL）を移し、全goldenの検査と、旧実装の照合をつなぐ
  - 完了: WSLで、旧実装の照合が成功する。Windowsの基準環境で、全goldenの検査が成功し、旧実装の照合が、既存の回帰testに任せてskipされる。
  - _Depends: 1.1_
  - _Requirements: 1.4, 2.2, 2.3, 2.4_

- [x] 2. 新実装の照合
- [x] 2.1 goldenの照合のtestが、環境の記録でgoldenを選ぶ
  - 完了: WSLとWindowsの基準環境で、3ケースの照合が成功する（skipなし）。
  - _Depends: 1.2_
  - _Requirements: 2.2, 2.3_

- [ ] 3. 文書と検証
- [x] 3.1 基準環境の文書を、環境ごとのgoldenの説明へ書き直す（研究室サーバでgoldenを足す手順を含む）
  - _Requirements: 4.1_

- [ ] 3.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: Windowsの全pytest、Ruff、Pyright、`pip check`、照合script（`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とWindows用のgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.1, 3.1_

## Implementation Notes

### 実装（task 1.1〜3.1）

- きっかけ（2026-10-11、ユーザーが研究室サーバで実行）: 新実装は、同じprocessの中の旧実装と、3ケースとも一致した（Python 3.10.16、NumPy 2.2.6でも動く）。旧実装のsine2の結果が、WSLで作ったLinux用のgoldenと違った（候補の棄却が2件。WSLは3件）。sea2とmnist2は、WSLと同じだった。
- 手順からの逸脱: 選択・名前・作成の規則の関数と、そのtestを、同じfileに、同時に書いた（testを先に書いて失敗を確かめる手順からの逸脱）。
- 前のspec（linux-proposed-regression-golden）のtestのfileとgoldenのfileを、`git mv`で、新しい名前・場所へ移した（goldenの中身は、変えていない）。OSの名前で選ぶ形（`GOLDEN_PATH_BY_SYSTEM`）と、Linux以外での作成の拒否は、なくなった。
- 「環境が違うときは、警告して比較する」扱いは、環境ごとのgoldenでは、なくした（完全に一致するgoldenだけを選ぶ。なければskip）。既存の回帰test（Windows用）は、変えていない。
- sourceは変えていない。

### 照合の結果

- WSLで成功（WSL Ubuntu、Python 3.14.4、NumPy 2.4.6）: `test_environment_proposed_regression.py`と`test_fedsda_run_metric_derivation.py`が、45 passed・skipなし（旧実装と新実装の3ケースが、移したgoldenと一致）。`--update`は、「すでにgoldenがある」で拒否された。
- Windowsの基準環境で成功: 実行環境の記録が、固定のgoldenの`_env`と一致し、新実装の3ケースが、固定のgoldenと一致（skipなし）。旧実装の照合は、既存の回帰testに任せてskip（1件）。`--update`は、「固定のgoldenの環境」で拒否された。
- この作業の途中、Windowsで、スマートアプリコントロールがtorchの読込みを一時ブロックした（既知の現象。保護設定・venvは変えず、WSLで先に進め、解けた後にWindowsで実行した）。

### 未検証・残る制約

- 研究室サーバ用のgoldenは、まだない（ユーザーが、サーバで`--update`を1回実行して足す。手順は docs/experiments/refactoring-baseline.md）。それまで、サーバでは、goldenとの照合がskipされる（新実装と旧実装の、同じprocessの中の照合は、行われる）。
- 環境の版が変わると、goldenが選ばれなくなり、照合がskipになる（research.mdの「Trade-off」）。
- 同じ環境の記録で、結果が違う計算機があるかどうかは、分からない。
