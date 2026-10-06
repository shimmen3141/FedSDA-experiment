# 実装task revision 1

- [x] 1. 参照モデルの固定の組立と依存境界をTDDで実装する
  - testを書いたら実装より前にREDを実行して記録する。GREEN後、実装をstubと誤実装へ一時的に差し替えてtestの失敗を確認し、元へ戻してhashを照合する。実旧_snapshot_reference_modelsと、同じ乱数状態からの処理後torch乱数・参照IDの順序・全parameter値・出力を対照（class2/4、保有一覧3種）。独立性（1.2）: 共有部と全parameterの実体が別、保有モデルの更新が参照へ及ばない。履歴平均（1.3）: 件数0/1/2/多数・平均0・統計未登録を、実旧_begin_forward_validation（候補の学習だけ無効化）のsessionと対照。保有モデル・統計・Python/NumPy乱数の不変（1.4）。拒否でtorch乱数が未消費: 2.1 型、2.2 保有0件、2.3 非有限parameter（一覧の途中のモデルを含む）。AST注入RED→6symbol exact guard GREEN。品質/型/独立レビュー/主担当gate。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3,3.1
- [x] 2. 固定から観測・評価・確定までを接続する
  - 実旧のsession開始（候補の学習だけ無効化）→観測→確定までと、新の固定→損失収集の開始→観測→評価→確定を照合（統計の与え方3通り、class2/4）。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）で共同更新→固定→保有モデルをさらに共同更新（参照は固定されたまま）→観測と確定を実旧と照合。fresh新CPUで旧importなしの固定→観測→評価→確定。独立レビュー/主担当gate。
  - Requirements: 3.2
- [x] 3. 固定環境の全回帰を確認する
  - 新moduleと依存guardの追加が既存testと固定旧実装の値へ影響しないことを同じcommitで示す。全pytest/旧11・最終3golden（主担当実測＋JUnit、基準はsteering/agent-handoff.md）、品質/型/pip/diff、固定旧差分空、承認/source hash・実測を記録。独立レビュー/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終GOで正本・再開案内/roadmap更新、commit/pushする。
