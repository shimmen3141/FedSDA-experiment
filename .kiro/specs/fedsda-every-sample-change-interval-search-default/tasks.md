# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4 → task 5。task 1の「替える前」の保存は、既定を替える前に行う。task 3のgolden生成・設置・版の定数の切替は主担当が行う。

- [ ] 1. 替える前の結果を保存する（コードは変えない）
  - このspecの基点のcommitで、要求4の5条件の構成`fedsda-every-sample-change-interval-search`の全体runのrawと、要求6.3の研究ツールの4つの小さい条件の結果を、Git管理外（`../../venv/refactoring-tests/every-sample-default-before/`）へ保存する。保存のscriptもGit管理外に置く。
  - 完了条件: 保存した場所・条件・commitをImplementation Notesに書く。研究ツールの`default-reverse-prequential-grid1`の条件で警報が1回以上起きることを確かめる。
  - _Requirements: 4.1, 6.3_
- [ ] 2. 既定を替え、構成と設定の組と研究ツールを合わせる
  - 既定・構成（要求1）と最小記述長でない選び方の組（要求2.1）のtestを、新しいtestのファイル`tests/refactoring/test_fedsda_every_sample_change_interval_search_default.py`に先に書き、REDを確かめてから、既定を替え、新しい構成を足し、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ足す。
  - 既定に替えた後に失敗する、最小記述長でない選び方の組（旧と比べる入口、実旧との照合の組、testとfresh script）で今の扱いを明示する。既定に依るtest（runの識別子、構成の一覧ほか）を、失敗したものから合わせる。
  - 研究ツールの基点の固定と、IMPROVE-031・IMPROVE-032の変種の今の扱いへの戻しを足す（研究用CLIと研究用の入口）。
  - 替えた後の結果を`../../venv/refactoring-tests/every-sample-default-after/`へ保存し、task 1の結果と、要求4.1の(a)〜(c)と要求6.3で比べる（一時scriptはGit管理外）。
  - 完了条件: 新しいtestが通り、基準版1〜15と実旧との照合が値を変えずに通り、要求4.1と6.3の比較が一致する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1, 3.2, 4.1, 6.1, 6.2, 6.3_
  - _Depends:_ 1
- [ ] 3. 基準版16を足す
  - 版16の登録、経路の記録の新しい項目（版16から）と必須の経路、goldenを使わない版15→16の差のtestを足す（実装担当）。commitの後、主担当が、必須の経路の2ケースの値を確かめ、Windowsの基準環境とWSLでgoldenを生成して`v16/`へ置き、`BASELINE_VERSION`と`BEHAVIOR_BASELINE_VERSION`を16へ上げる。
  - 完了条件: 版1〜16の照合と、版15→16の差のtestが、WindowsとWSLで通る。
  - _Requirements: 5.1, 5.2, 5.3_
  - _Depends:_ 2
- [ ] 4. 文書を合わせる
  - 提案手法の概要、モデル再利用の文書、基準の文書（版16）、実験の入口、IMPROVE-032と一覧、再開案内。
  - 完了条件: 文書が新しい既定を説明し、過去の日付の記録を書き換えていない。
  - _Requirements: 7.1_
  - _Depends:_ 3
- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_
  - _Depends:_ 4

## Implementation Notes
