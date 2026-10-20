# Implementation Plan

- [ ] 1. 統合で同じ共有部を保つ扱いを選べるようにする（既定は、まだ変えない）
  - 部品のtest（加重平均の分岐）、設定の検査のtestを先に書き、REDを確かめる。
  - 参加者の設定の新しい項目と受け渡し、加重平均の分岐、比べ方の関数の共有。全部の参加者の設定の組立てで旧の扱いを明示し、旧と比べる入口と、基準版1〜8の「後に足された項目」の表へ足す。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜8）が値を変えずに通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2_

- [ ] 2. 最終構成の既定を替える
  - 既定と全FedSDA構成の値のtest、共有部の送り直しが無くなる全体runのtestを先に書き、REDを確かめてから、既定を替える。既定に依るtest（runの識別子ほか）を、失敗したものから合わせる。
  - 完了条件: 構成と送り直しのtestが通り、基準版1〜8と実旧との照合が値を変えずに通る。
  - _Requirements: 2.1, 4.3_

- [ ] 3. 基準版9を足す
  - 版9の定義と、版8→9の差の説明のtestを足し、commitしてから、Windowsの基準環境とWSLでgoldenを生成して足し、現在の版を9へ上げる。
  - 完了条件: 版1〜9の照合と、版8→9の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 4.1, 4.2_

- [ ] 4. 文書を合わせる
  - 基準（版9）、指標（通信量の注記）、実験の入口、IMPROVE-020、再開案内。
  - 完了条件: 文書が新しい扱いと、版8の結果のSEAの通信量の注意を説明する。
  - _Requirements: 5.1_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合の結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 下調べ（2026-10-20、手元、1 run）: `C:\\Users\\yshin\\AppData\\Local\\Temp\\claude\\c--Users-yshin-Research-FedSDA-experiment\\799b882d-c886-45dd-a18e-a06f3e20f4b4\\scratchpad\\shared_resend_probe.py`。間隔500・2500標本・seed 0で、SEA4・SEA2は統合1回で共有部が変わり（最大の差1.19e-7）、送り直し70.8KBのうち47.4KBが共有部。CIRCLE2は統合3回のうち1回で変わった（最大5.9e-8）。
- 仕様レビュー1回目（Luna high）: Major 2件（全体runの送り直しのtestが、統合が起きなくても通る。版8→9の比較の対象に指標が無い）、Minor 1件（共有部が一致する条件の、要求と設計の書き方の違い）。新旧の両方で統合が起きたことと、旧の扱いで送り直しが起きたことを確かめる、統合の無いケースは指標も比べる、正の件数のメンバー全員が代表と同じときだけ保つ（代表の件数0・共有部が空の場合も明記）、と直した。2回目（再確認）PASS。
