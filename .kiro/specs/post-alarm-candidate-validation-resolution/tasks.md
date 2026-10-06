# 実装task revision 2

- [x] 1. 確定の適用の組立と依存境界をTDDで実装する
  - testを書いたら実装より前にREDを実行して記録する。GREEN後、実装をstubと誤実装へ一時的に差し替えてtestの失敗を確認し、元へ戻してhashを照合する。実旧_finalize_forward_validationの4分岐と、全owner状態・結果種別と旧actionの対応・帰属先・変更記録と旧通知引数を対照（class2/4、保留標本0/複数）。新の評価結果は同じ損失列から移植済み評価関数で作り、旧判定と一致を確認（test-only接続）。1.5の不変。拒否時不変: 2.1 評価結果と現在ID owner、2.2 概念ID列（全結果種別）、2.3 各結果種別の委譲先の拒否代表入力と未保有の再利用先。要求2.4（検証を終えてから状態を変更する）は、上の各拒否で全状態が不変であることと、呼出順の観測（採用は採用の組立1回だけ、再利用は吸収→現在ID切替え、維持/棄却は吸収1回だけで現在IDのownerを更新しない）で確認する。AST注入RED→16symbol exact guard GREEN。品質/型/独立Luna/主担当gate。
  - Requirements: 1.1,1.2,1.3,1.4,1.5,2.1,2.2,2.3,2.4,3.1
- [x] 2. 確定後の学習を接続する
  - 12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）×結果種別（採用・再利用）の実NNで、共同更新→実旧確定処理/新の評価と確定→各自の標本storeを使う共同更新を実旧と照合。fresh新CPUで旧importなしの評価→確定（4種別）→学習。独立Luna/主担当gate。
  - Requirements: 3.2
- [x] 3. 固定環境の全回帰を確認する
  - 新moduleと依存guardの追加が既存testと固定旧実装の値へ影響しないことを同じcommitで示す。全pytest/旧11・最終3golden（主担当実測＋JUnit、基準はsteering/agent-handoff.md）、品質/型/pip/diff、固定旧差分空、承認/source hash・実測を記録。独立Luna/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終Luna GOで正本・再開案内/roadmap更新、commit/pushする。
