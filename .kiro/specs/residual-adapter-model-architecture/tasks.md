# タスク: Residual Adapterモデル構造
共通source/testを扱うため全taskは順次、(P)なし。CPU shared venvと既存設定基盤は実装済み。
検証環境はdocs/experiments/refactoring-baseline.md。各task後に独立Luna reviewと主担当fresh検証を行う。

- [x] 1. 最終モデル構造と旧直接照合を実装する
  - 対象testを先に作りmissing-module REDを実測後、三つのnn.Moduleと構築前検査、forward/共有特徴経路を実装してGREENにする。
  - 全state・CPU RNG消費順・binary/multiclass・rank制限・空幅・共有注入を実旧oracleへ照合する。optimizer builderだけtest-only no-opとする。
  - 完了は対象pytest成功、指定3productionとtest境界内、Luna reviewと主担当fresh検証で観測する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 3.1_
  - _Boundary: NN architecture（指定3production・対象test）_
  - _Depends: なし（設定基盤は完了済み）_

- [ ] 2. 拒否・勾配・共有と既存部品への接続を検証する
  - test名追加が必要なら命名revisionをLunaレビューしてから対象testだけを追加する。
  - forged設定/型・値域/共有実構造/forward契約を拒否し、構築前RNG/入力/shared不変、empty batch、storage共有/独立、ambient dtype/device/grad保持を検証する。
  - zero展開と非zero展開で実旧への入力・全Parameter勾配比較、新forward→既存確率/mean loss、既存snapshot選択→独立新モデル標準load_state_dictをtest-only接続する。
  - 完了は対象pytest成功とLuna review、主担当fresh検証。test-only追加にfake REDを要求しない。
  - _Requirements: 1.4, 1.5, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1_
  - _Boundary: Test-only integration（対象testのみ）_
  - _Depends: 1_

- [ ] 3. 依存境界と全回帰・完了証拠を揃える
  - AST exact3moduleの禁止/許可注入testを先に追加してRED後、指定symbolのallowlistを更新してGREENにする。
  - fresh CPUモデルforward smoke（旧importなし）、対象test＋AST、全tests旧11/最終3goldenを実行する。旧production/golden/旧比較test差分なしを照合する。
  - integration-validation.mdとroadmapへ12条件・実配置・検証結果・学習や新全体runの未移植範囲を記録する。再現した旧問題だけfindings台帳へ記録する。
  - 完了はLuna task review/最終feature GO、主担当fresh完了検証とmetadata更新で観測する。
  - _Requirements: 4.1, 4.2_
  - _Boundary: AST tests / integration evidence（依存test・対象spec・roadmap・実証findingsのみ）_
  - _Depends: 1, 2_

## Implementation Notes
日本語はapply_patchで編集し、PowerShell→Python stdinで生成しない。
torch.set_default_deviceで永続global modeを作らず、meta環境testにはtorch.device contextを用いる。
旧prefixへのmappingはtest-only。NNにoptimizer・method progressionを追加しない。
