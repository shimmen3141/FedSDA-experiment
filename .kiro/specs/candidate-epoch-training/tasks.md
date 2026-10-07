# 実装タスク revision1

- [ ] 1. 候補のエポック学習条件を型で宣言する
  - 全6fieldの必須/値域/frozen/kw_onlyをtest先行RED後に実装し、正常値と不正値を確認する。
  - 学習率を二重定義せず、既存metadata検査を利用する。
  - _Boundary: 学習設定moduleと対象設定test_
  - _Requirements: 1.1, 1.2, 1.4, 3.1_

- [ ] 2. 区間の事前検査と固定エポック反復を実装する
  - 学習moduleに結果record・事前検査・dataset反復helperをtest先行RED後に実装する。公開方式dispatchは次taskで追加する。
  - class2/4×3optimizerで全区間学習、0epoch、末尾batchを実旧fixedメソッドへ対照し、dataset/parameter/grad/optimizer/RNG/計数を確認する。
  - 設定/区間/owner/binding/gradの拒否時に全状態が不変であることを確認する。
  - _Boundary: epoch学習moduleの事前検査/反復helper、対象固定/拒否test_
  - _Depends: 1_
  - _Requirements: 1.1, 2.1, 2.3, 3.1_

- [ ] 3. 検証損失による停止と方式dispatchを実装する
  - 公開入口をtest先行RED後に追加し、無作為分割・丸め・非改善停止・parameterのみ復元・小区間fallback・skip・0epochの方式差を実旧へ対照する。
  - class2/4×3optimizerで厳密閾値と最良snapshotのみ復元を確認する。stub/復元省略/optimizerリセット/乱数差の差し替えを検出して復元hashを照合する。
  - _Boundary: epoch学習moduleの公開入口と停止処理、対象方式/復元test_
  - _Depends: 2_
  - _Requirements: 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1_

- [ ] 4. 候補生成から学習と継続更新へ接続する
  - test-onlyで初期値選択/生成→epoch学習→次batch更新を実旧へ照合し、parameterのみ復元後のoptimizer/gradを含めて確認する。
  - fresh新CPUで生成と学習を実行し、旧importなしとsession未完成の範囲を記録する。
  - _Boundary: 生成からの接続test、fresh CPU検証_
  - _Depends: 3_
  - _Requirements: 2.1, 2.2, 2.3, 3.3_

- [ ] 5. 依存境界と基準環境の全回帰を統合検証する
  - 明示的な統合検証task。2moduleのAST注入RED→exact guard→GREEN、許可symbolsと禁止runtime/config/旧/乱数再設定を確認する。
  - 実装commitで全pytest/JUnit/Ruff/Pyright/pip/固定旧差分/承認hashを照合し証拠を記録する。
  - 全task承認後に別Lunaセッションのfeature最終GOを受ける。
  - _Boundary: AST境界と統合検証の記録_
  - _Depends: 4_
  - _Requirements: 3.2, 3.3_

