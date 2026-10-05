# 実装タスク

- [x] 1. 指定回数の抽出と共同更新を接続し、実旧反復と照合する
  - 明示保有ID対応と反復を実装し、missingmoduleの実RED→GREENを確認する。
  - 二値/多クラス、3optimizer、共有更新/凍結、0/1/4回、batch1/3・逆順対応表で実旧反復へ全loss/parameter/grad/optimizer/終端RNGを照合する。
  - 完了は実旧sampling/forward/loss/backward/optimizerを用いる対照テストの成功で確認する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.3, 3.1_
  - _Boundary: 借用対応記録・反復executorと実旧対照test_

- [x] 2. 無操作・契約拒否・失敗後の保持を検証する
  - 対応表/回数preflight、0の未アクセス、未参加payload、sample入力拒否、frozen/借用参照と環境保持を確認する。
  - 空共有・None共有optimizerを新NNで検証し、後段の実更新拒否でも完了更新と消費RNGを巻き戻さない証拠を示す。
  - 完了は拒否前のRNG/NN/optimizer不変と、失敗後の保持を比較する対象テスト成功で確認する。
  - 既存の実装が契約を満たす場合はtest-onlyとし、架空のREDを作らない。
  - _Requirements: 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_
  - _Boundary: 反復executorの契約検証_
  - _Depends: 1_

- [ ] 3. 依存境界と全回帰・統合証拠を揃える
  - 新2moduleのexact禁止/許可import testを先に追加し、実RED→AST guard GREENを確認する。
  - fresh CPU反復smoke（旧非import）、対象/全pytest・基準ローカル旧11最終3golden、Ruff/型/依存検査・旧固定差分/内容hashを確認する。
  - 全12条件の実測/採否と完成・後続境界を記録し、Luna taskAPPROVED/featureGOをもって完了を確認する。
  - _Requirements: 3.1, 3.2_
  - _Boundary: AST・統合証拠（依存test/対象spec/roadmap）_
  - _Depends: 1, 2_

## Implementation Notes
標本順と対応表順を混同しない。旧samplerの観測で余分なdrawを行わない。全乱数をfinally復元。
反復の回数算出/optimizer生成・reset/FIFO所有/診断/counterを取り込まない。
