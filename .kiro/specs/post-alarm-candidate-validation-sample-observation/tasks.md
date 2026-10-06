# 実装task revision 1

- [ ] 1. 検証標本の観測の組立と依存境界をTDDで実装する
  - REDより前に実旧_observe_forward_validationを実client・実session・実旧参照snapshotで実行できることを確認（不可なら設計へ戻す）。testを書いたら実装より前にREDを実行して記録する。GREEN後、実装をstubと誤実装へ一時的に差し替えてtestの失敗を確認し、元へ戻してhashを照合する。各観測回後の候補損失列・参照損失列（値・順序・ID順）と到達の戻り値を実旧sessionと対照（class2/4、規定件数2/4）。分類器のparameter/grad/学習mode・乱数の不変（1.3）。拒否時に収集が不変: 2.1 収集と対応の型、2.2 複数行・分類器/特徴/ラベル不正、2.3 位置の不連続・参照IDの不一致・到達済み。呼出順（2.4）。AST注入RED→4symbol exact guard GREEN。品質/型/独立レビュー/主担当gate。
  - Requirements: 1.1,1.2,1.3,2.1,2.2,2.3,2.4,3.1
- [ ] 2. 観測から評価・確定・学習までを接続する
  - 実旧の観測処理を規定件数まで実行した後の状態と、新の観測→評価→確定の状態を照合（履歴平均なし/現行モデル/別モデル、class2/4）。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）で共同更新→観測と確定→各自の標本storeを使う共同更新を実旧と照合。fresh新CPUで旧importなしの収集開始→観測→評価→確定→学習。独立レビュー/主担当gate。
  - Requirements: 3.2
- [ ] 3. 固定環境の全回帰を確認する
  - 新moduleと依存guardの追加が既存testと固定旧実装の値へ影響しないことを同じcommitで示す。全pytest/旧11・最終3golden（主担当実測＋JUnit、基準はsteering/agent-handoff.md）、品質/型/pip/diff、固定旧差分空、承認/source hash・実測を記録。独立レビュー/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終GOで正本・再開案内/roadmap更新、commit/pushする。
