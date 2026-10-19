# Implementation Plan

- [ ] 1. 候補の共有部の更新を選べるようにする（既定は、まだ変えない）
  - 候補の学習のtest（「固定する」で共有部が変わらず概念固有部が変わる——固定エポックと早期終了——、共有部を持たない構造で2つの値が同じ、組合せの検査）を先に書き、REDを確かめる。
  - `CandidateEpochTrainingSettings`の新しい項目、候補の学習の分岐、`FedsdaRunClientSettings`の組合せの検査。最終構成の既定・test・fresh scriptの全部の生成箇所で、`update_shared_features_with_candidate`を明示する。旧と比べる入口と検査に、共有部の明示を足す。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜6）が値を変えずに通る（明示した値は、これまでの動作と同じ）。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.3_

- [ ] 2. 最終構成の既定を替え、構成と基準版1〜6の明示を合わせる
  - 構成の値のtest（全FedSDA構成の採否の方針と共有部、全構成の組立てと小さい実行、コマンドの引数の一覧）と、`fedsda-with-candidate-validation`の専用のtestを先に書き、REDを確かめる。
  - 最終構成の2つの既定、`fedsda-with-candidate-validation`の追加（構成の一覧、goldenにしない構成の登録は版7で事例にするので不要か確かめる）、`fedsda-without-candidate-validation`の明示、旧のgoldenの条件の明示、基準版1〜6の明示（足された項目と、変わった項目の表）。既定に依るtestを、設計の表のとおり、失敗したものから合わせる（実旧と比べるtestの期待値は変えない）。
  - 完了条件: 構成のtestが通り、基準版1〜6と実旧との照合が値を変えずに通る。
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2_

- [ ] 3. 基準版7を足す
  - 新しい2ケースの条件を下調べで決め（必須の経路を通るseed）、版7の定義（ケース、候補検証の4ケースの構成、経路の記録の新しい項目、必須の経路）と、版6→7の差の説明のtestを足す。sourceとtestをcommitしてから（この中間のcommitでは、版7のgoldenがまだ無いので、両方の基準環境のgoldenが揃うことを確かめるtestが失敗する。前の版と同じ扱い）、`--baseline-version 7`で、Windowsの基準環境とWSLで生成して足し、照合する。現在の版（`BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION`）を7へ替えるのは、goldenを置いた後。
  - 完了条件: 版1〜7の照合と、版6→7の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 4. 研究ツールと文書を合わせる
  - 研究ツールの、候補検証ありの対照の構成名を`fedsda-with-candidate-validation`へ替え、替えた研究ツールのchecksを明示実行する。
  - 実験の入口の文書（構成の表）、基準の文書（版7）、最終提案構成の文書（固定設定）、研究記録（既定の判断）、再開案内（決定事項と完了specの一覧）。
  - 完了条件: checksが通り、文書が新しい既定と根拠を指す。
  - _Requirements: 5.1, 5.2_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL（goldenの版1〜7と、このspecのtest）。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合の結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 仕様レビュー1回目（Luna high、spec.json）: Minor 3件（命名表に設定の項目を載せた、乱数とoptimizer状態の検証条件の不足、task5の完了条件）を全部採用して反映した。Minorのみのため再レビューなし。
- 版7の新しい2ケースの条件の下調べは research.md。
