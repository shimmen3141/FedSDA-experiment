# Development finding: 公開symbolの依存guardでmodule全体のimportを許していた

- 観測日: 2026-10-06
- 観測した作業: held-model-training-state-registry Task3 Lunaレビュー
- 改善先: project（依存境界検査と移植時のレビュー）
- 関連artifact: ../.kiro/specs/held-model-training-state-registry/review.md、../tests/refactoring/test_single_run_dependency_boundaries.py

## 観測した事実

公開型だけを許可する設計に対し、AST allowlistへfrom importの解決用base moduleも追加していた。そのため新guardが`import torch`、`import torch.nn`、`import dataclasses`等を受理し、未許可symbolへアクセスできた。Lunaは具体的な注入で再現しTask3をREJECTEDとした。
7種のmodule import拒否testは修正前7 failed。単純にresolver結果の先頭を除く最初の修正では、resolverがpackageごとにbaseを含めるか異なり、runtime/configuration import拒否の2 testが失敗した。
実際にfromが束縛するsymbolを直接解決し、bare importを拒否することで、対象＋AST617 passed/exit0になった。
旧adopted-candidate-shared-feature-integration用の既存guardも`import torch`に対して違反なしを返すことを同じ関数で観測した。実productionが未許可symbolを使っているという観測ではなく、機械的な防止範囲の不足である。旧アルゴリズムの不具合とは区別する。

## 影響とworkaround

- 影響: 依存guardのGREENだけでは「指定公開型だけ」の制約を保証できない。
- その場の対応: 今回のregistryだけにsymbol検査とbare/module import拒否を追加した。他の既存guardを同時に変更せず、再検討候補としてこの記録へ残す。

## 仮説と改善案

- 仮説: from解決のbase許可と、実際の名前束縛の許可を同じallowlistで扱うと同様の抜けが生じやすい。
- 改善案: 今後のexact依存guardではbare import、moduleを選択するfrom、alias、wildcard、package経由の注入を受け入れ条件に含める。既存guardの見直しは対象とする公開symbol契約ごとに別変更で行う。

## 改善結果

新registryのguardはbare importを拒否し、fromの具体symbolだけを比較する。module/aliasを含む7再現ケースがGREEN、対象＋AST617 passed。最終全回帰と独立再レビューは対象specへ記録する。
