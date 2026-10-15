# Implementation Plan

- [x] 1. 同値の検査を外す
  - 2か所の検査を消し、既存の3つのtest（同値を求める2つと、直接の構築の検査の容量の3 case）を、設計のとおり変える。最終構成のTに自分の定数を持たせる。設計のValidation Designの1をtestで先に書く。実装の前に、同値を前提にするtestがほかに無いかを全参照で確かめ、全体の設定の検査での容量1の扱いを確かめる（Implementation Notesへ）。
  - _Requirements: 1.1, 1.2, 1.3, 4.1_

- [x] 2. 軸
  - [x] 2.1 計画の項目と検査、変種の作り方（2つの軸の組、容量の軸はTを替えない）、FedSDAの構成が無いときの拒否。計画の項目が掃引の記録に入るので、同じ変更で、掃引の記録の形式の版（7）と、既存のtestの期待（記録の版、記録の計画の完全一致、容量の軸のtest、混ぜる予測の構成と容量の軸のtest）を合わせる。testを先に書く（Validation Designの2〜6）。
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.2, 4.1_
  - [x] 2.2 コマンドの引数（新しい引数と、容量の引数の説明）、一覧の表の列、コマンドの引数の一覧のtestの期待。testを先に書く（Validation Designの5のコマンドの部分と7）。
    - _Requirements: 2.4, 3.1, 3.2, 4.1_

- [x] 3. 文書
  - 実験の入口の文書（Tの軸、混ぜる分け合い1/T、容量の軸の意味の変化、目安、組合せの順、記録の形式の版7）、研究バックログの8、IMPROVE-015（実装済み）、再開案内、完了specの一覧。
  - _Requirements: 5.1_

- [x] 4. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright、WSL（新しいtestとgoldenの版1〜5）。
  - _Requirements: 全部、4.1_

## Implementation Notes

- 実装の前に、同値を前提にするtestを全参照で確かめた（仕様レビューで見つかった直接の構築の検査の3 caseを含め、設計の一覧のとおり。`fresh_process_smoke.py`は2つを同じ値で作るだけで、影響なし）。全体の設定の検査は警報を待つ扱いの下限を持たないので、容量1はそこでは受理される（clientの組の検査だけが、警報を待つ扱いで拒否する）。
- testを先に書き、実装の前に失敗することを確かめた（task 1で6件、task 2.1で14件、task 2.2で3件）。
- 変種の作り方は、clientの設定の一部を替える補助`_replace_client_settings`で、容量の軸、Tの軸の順に掛け合わせる（同値の検査が無いので、軸ごとに作り直してよい）。
- 実装レビュー（独立Haiku 5.5 medium、1回）: APPROVED（Minor 4）。1: 文書の目安の言い回し（Tだけ・分け合いだけを、それぞれ60にした結果）を直した。2: 警報を即時にする構成で、容量1がT（30）と違う値のまま受理されるcaseを、新しいtestへ足した。3: 混ぜる重みの分け合いの割合を内部の属性で読むtestは、そのまま（コメントで明記）。4: このcheckboxを付けた。
- 検証: 全pytest 14325 passed / 8 skipped、失敗0（JUnit `../../venv/refactoring-tests/time-scale-full.xml`）。全pytestの開始の後に、新しいtestへ1 caseを足し、そのファイルだけを再実行して16 passed。Ruff・Pyright 0件。WSLで、新しいtestとgoldenの版1〜5の220件が通った。
