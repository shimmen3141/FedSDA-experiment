# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。Windowsの全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. Linux用のgolden
- [x] 1.1 照合のtestと、作成の道具
  - 形の検査、Linuxでの照合、Linux以外での作成の拒否のtestを先に書く。
  - 完了: Windowsで、形の検査（goldenを作った後）と、作成の拒否のtestが成功し、Linuxの照合がskipされる。
  - _Requirements: 1.3, 1.4, 2.2, 2.3, 2.4_

- [x] 1.2 WSL Ubuntuで、Linux用のgoldenを作り、再現を確かめる
  - `--update`で作る。別のprocessで、照合のtestを実行する。
  - 完了: WSL Ubuntuで、旧実装の照合が成功する。形の検査が、WindowsとWSLで成功する。
  - _Depends: 1.1_
  - _Requirements: 1.1, 1.2, 2.1_

- [x] 2. 新実装の照合
- [x] 2.1 goldenの照合のtestが、実行環境のgoldenを選ぶ
  - 完了: WSL Ubuntuで、指標の導出のtest（3ケースの、実旧との照合と、Linux用のgoldenとの照合）が成功する。Windowsで、同じtestが成功する。
  - _Depends: 1.2_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 3. 文書と検証
- [x] 3.1 基準環境の文書へ、Linuxの手順を書く
  - _Requirements: 4.1_

- [ ] 3.2 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: Windowsの全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とWindows用のgoldenの差分が空である。WSLでの実行結果を、「WSLで成功」と区別して記録してある。
  - _Depends: 2.1, 3.1_

## Implementation Notes

### 実装（task 1.1〜3.1）

- 手順: 照合のtestと作成の道具（task 1.1）を先に書き、Linux用のgoldenがない状態で、形の検査が失敗することを確かめた。その後、WSL Ubuntuでgoldenを作った。
- 実装中の発見: WSLのgitは、Windows側のworktree（`.git`が、Windowsの形式のpathを指す）を読めない。作成の道具へ、commitを渡す引数（`--source-commit`）を足した（渡さなければ、`git rev-parse HEAD`）。goldenの`source_commit`は、Windows側で求めた`31e927b`（仕様のcommit。旧実装は、固定旧`748c3aa`から無変更）。
- sourceは変えていない。変更は、testのfile 2つ、goldenのfile 1つ、基準環境の文書。

### 照合の結果

- Linux用のgoldenの作成環境: WSL Ubuntu / x86_64 / Python 3.14.4 / NumPy 2.4.6 / torch 2.12.1+cpu / 1 thread。
- Windows用のgoldenとの違い: sine2は指標10・離散列15、sea2は指標1・離散列2が違い、mnist2は一致（2026-10-10の調査と同じ）。イベント件数（登録・採用・棄却・統合・切替）は、3ケースとも、Windowsと同じ。
- WSLで成功: 作成の後、別のprocessで、旧実装の3ケースが、Linux用のgoldenと一致（`test_linux_proposed_regression.py` 3 passed）。新実装の3ケースが、同じprocessの中の実旧と、Linux用のgoldenの両方に一致（`test_fedsda_run_metric_derivation.py`と合わせて 39 passed、skipなし）。goldenの条件ごとの、経路の期待と、候補の判定の理由の期待は、Linuxでも、Windowsと同じだった。
- Windowsで成功: 形の検査と、作成の拒否のtest。旧実装とLinux用goldenの照合は、skip。新実装は、これまでどおり、Windows用のgoldenと一致。

### 未検証・残る制約

- 研究室サーバでは、実行していない（主担当は入れない）。WSL Ubuntuと結果が同じかどうかは、分からない。照合の手順と、一致しなかったときの扱いは、docs/experiments/refactoring-baseline.mdの「Linux用の最終構成golden」に書いた。
- 旧の代表11ケース（tests/regression_golden.json）のLinux版は、作っていない。WSLでは、`tests/test_regression.py`と`tests/test_proposed_regression.py`は、これまでどおり失敗する（Windows用のgoldenと比べるため）。
