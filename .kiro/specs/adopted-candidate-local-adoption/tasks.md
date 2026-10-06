# 実装task revision 2

- [x] 1. 採用時のローカル状態更新の組立と依存境界をTDDで実装する
  - oracleは実旧clientと実ForwardValidationSessionで実行する実_finalize_forward_validationの採用分岐（設計時に実行可能性を確認済み。手順の再構成による代替は使わず、実行できなくなった場合は設計へ戻してレビューを取り直す）。実旧採用分岐と、一時ID/一覧/値/統計/保留・待機/計数/標本列・順序・identity/現在ID/変更record/採番次値を対照（class2/4、保留標本0/1/複数、計数0/正、連続2回）。保留標本の追加で既存/新モデルの損失統計と割当概念計数が変わらないことを実旧と照合（1.4）、既存owner項目・評価標本・乱数の不変（1.4）。拒否時不変は、本関数の検証点（2.1 owner型、2.2 計数と保留標本、2.3 標本store/計数store/現在IDの使用済みID）と登録の拒否代表入力（2.4 一覧/統計/送信保留の使用済みID、保有0件、待機、登録先owner型、候補/管理器/特徴/ラベル/非有限parameter）の各点で、採番次値と全owner状態を観測。呼出順（2.5）。判定記録・switch位置・適応イベント・通知を新関数が行わないこと（3.1）は、返り値が変更recordだけであることとexact依存guardで確認。AST注入RED→13symbol exact guard GREEN。RED→GREEN、品質/型/独立Luna/主担当gate。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3,2.4,2.5,3.1
- [x] 2. 採用後の学習と登録確認を接続する
  - 12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）の実NNで、共同更新→候補の独立学習→実旧確定処理/新採用→採用後の共同更新→正式ID確認→共同更新を実旧と照合。fresh新CPUで旧importなしの採番→採用→後続学習→確認。独立Luna/主担当gate。
  - Requirements: 3.2（1.4の乱数不変と状態保持を継続学習でも確認）
- [x] 3. 固定環境の全回帰を確認する
  - 新moduleと依存guardの追加が既存testと固定旧実装の値へ影響しないことを同じcommitで示す。全pytest/旧11・最終3golden（主担当実測＋JUnit、基準はsteering/agent-handoff.md）、品質/型/pip/diff、固定旧差分空、承認/source hash・実測を記録。独立Luna/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終Luna GOで正本・再開案内/roadmap更新、commit/pushする。
