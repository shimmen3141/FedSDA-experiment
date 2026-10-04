# タスク: 候補パラメータ初期化
全タスクは順次。共通production/testファイルを扱うため(P)なし。
共有venv/pytest/pythonpathとCPU torchは準備済み。実行環境はdocs/experiments/refactoring-baseline.md。

- [ ] 1. 初期化設定と独立snapshot作成を実装する
  - 対象testを先に書きmissing-module REDを確認し、三方式/同率/空/順序/型を実旧unbound helperへ直接照合してGREENにする。
  - 全入力検査とdetach cloneを実装し、snapshotを返す一責務に限定する。Shared風完全stateはtest入力で表現する。
  - 完了は指定2productionファイルと対象testの実pytest成功、Luna task reviewと主担当fresh検証で観測する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1_
  - _Boundary: CandidateParameterInitialization（指定2productionとtestのみ）_
  - _Depends: なし（既存設定検証基盤は実装済み）_

- [ ] 2. 拒否・独立性と警報後評価からの接続を検証する
  - 対象testだけを追加し、後段異常/forged設定/未知ID/重複/構造不整合/非有限とコピー双方向/grad/RNG/default dtype/device保持を検証する。
  - 既存候補評価の平均lossを明示tupleへ対応付けて初期化元を選び、適合失敗した評価元も候補初期値として使えることを確認する。採用・生成・学習をproductionへ追加しない。
  - 完了は対象pytest成功とLuna review、rootfresh検証。テストだけの追加にfake REDは要求しない。
  - _Requirements: 2.1, 2.2, 2.3, 3.1_
  - _Boundary: Test-only integration（対象testのみ）_
  - _Depends: 1_

- [ ] 3. 依存境界・全回帰と統合証拠を完成する
  - exact2module依存の禁止/許可注入testを先に追加しRED確認後にallowlistを更新する。
  - CPU fresh smoke、対象testとAST、全tests旧11/最終3goldenを実行し、旧production/golden差分なしを確認する。
  - integration-validation.mdとroadmapへ9条件/配置/未移植範囲/旧発見の有無を記録する。実証済み発見だけ既存台帳へ記録する。
  - 完了はLuna task review、全spec integration GO、rootfresh完了検証、spec metadata更新で観測する。
  - _Requirements: 1.4, 3.1, 3.2_
  - _Boundary: Architecture tests / integration evidence（AST test・対象spec・roadmap）_
  - _Depends: 1, 2_

## Implementation Notes
日本語ファイルはapply_patchで編集する。PowerShell→Python stdinの日本語文字列は破損する場合がある。
環境testでtorch.set_default_deviceは使わずtorch.device contextを使う。

