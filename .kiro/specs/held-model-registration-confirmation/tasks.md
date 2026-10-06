# 実装task revision 1

- [x] 1. 保有モデルの登録確認組立をTDDで実装する
  - 実旧保有済み分岐/非負current分岐、先既存/新先、補助元欠落、入力/型/モデル欠落拒否で全owner不変、順序/取得済みrecord保持を検証。RED→GREEN、品質/型/独立Luna/主担当gate。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3,3.1
- [ ] 2. 依存境界と登録確認後の学習を接続する
  - AST注入RED→GREEN。12NN条件3共同更新で初回後confirm、ID/全owner/数値/optimizer/Random保持を実旧と照合。fresh新CPUで旧importなしの確認→後続学習。独立Luna/主担当gate。
  - Requirements: 1.4,3.1,3.2
- [ ] 3. 固定環境の全回帰を確認する
  - 全pytest/旧11・最終3golden、品質/型/pip/diff、固定旧差分空、承認/source hash・JUnit/実測を記録。独立Luna/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終Luna GOで正本・再開案内/roadmap更新、commit/pushする。
