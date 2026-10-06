# 実装task revision 3

- [x] 1. 採用候補の初期ローカル登録の組立をTDDで実装する
  - 実旧の共有部構成clientの登録＋旧FedSDA待機設定と対照。保有一覧順×現在ID保有/非保有の反映先選択、class2/4、singleton/欠落クラス、別ID既存保留の置換、共有部値/全保有モデル出力/統計全field/snapshot値と独立性/待機・ready/候補optimizer reset、不変owner・RNGを検証。一時ID/待機/owner型/使用済みID3箇所/保有0件/候補・管理器・特徴・ラベル・非有限parameterの拒否で全状態不変、API呼出順を観測。RED→GREEN、品質/型/独立Luna/主担当gate。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3,2.4,2.5,2.6,2.7,3.1
- [x] 2. 依存境界と登録後の学習・登録確認を接続する
  - AST注入RED→GREEN（13symbol exact guard）。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）の実NNで共同更新→候補の独立学習→登録→登録後の共同更新を実旧と照合し、loss/全値/grad/optimizer state/RNGの一致を確認。既存の登録確認で一時ID→正式IDへ付け替えた後の継続も確認。fresh新CPUで旧importなしの登録→後続学習。独立Luna/主担当gate。
  - Requirements: 1.4,3.1,3.2
- [ ] 3. 固定環境の全回帰を確認する
  - 全pytest/旧11・最終3golden、品質/型/pip/diff、固定旧差分空、承認/source hash・JUnit/実測を記録。独立Luna/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終Luna GOで正本・再開案内/roadmap更新、commit/pushする。
