# 実装task revision 1

- [ ] 1. 帰属確定標本の吸収の組立と依存境界をTDDで実装する
  - 実旧_absorb_into_storeと、標本列・順序・identity/割当概念計数/統計全field（クラス初出順）を対照（class2/4、吸収先3種、標本0/1/複数、概念IDあり/None混在/なし）。他モデル・学習計数・parameter/grad/optimizer・乱数の不変（1.5）。拒否時不変: 2.1 model_idとowner型、2.2 標本列と概念ID列、2.3 未保有（空列を含む）、2.4 複数行record・特徴/ラベル不正（列の途中を含む）。呼出順（2.5）。旧の途中失敗時の部分更新を実旧で再現しimplementation-findingsへ記録。AST注入RED→6symbol exact guard GREEN。RED→GREEN、品質/型/独立Luna/主担当gate。
  - Requirements: 1.1,1.2,1.3,1.4,1.5,2.1,2.2,2.3,2.4,2.5,3.1
- [ ] 2. 確定処理の棄却分岐と吸収後の学習を接続する
  - 実旧_finalize_forward_validationの棄却分岐と新の吸収を照合（標本・計数・統計、現在ID不変）。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）の実NNで共同更新→吸収→各自の標本storeを使う共同更新を実旧と照合。fresh新CPUで旧importなしの吸収→学習。独立Luna/主担当gate。
  - Requirements: 3.2
- [ ] 3. 固定環境の全回帰を確認する
  - 新moduleと依存guardの追加が既存testと固定旧実装の値へ影響しないことを同じcommitで示す。全pytest/旧11・最終3golden（主担当実測＋JUnit、基準はsteering/agent-handoff.md）、品質/型/pip/diff、固定旧差分空、承認/source hash・実測を記録。独立Luna/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終Luna GOで正本・再開案内/roadmap更新、commit/pushする。
