# 実装タスク

- [x] 1. 準備済み分類器の標本別損失評価を実装する
  - 未実装公開APIのimport RED後、全入力の先行検証・一forward・出力検証・binary/multiclass損失を実装する。
  - 一標本/非連続/逆順の実旧損失一致、型/shape/value/parameter/output拒否、値/grad/flags/RNG/default/gradmode保持と独立結果を確認する。
  - 完了は対象単体GREEN・Ruff/format/Pyright・Luna Task APPROVEDで判断する。
  - _Boundary: 一分類器の標本別評価と対象単体test_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1_

- [x] 2. 準備後の損失から初期統計への接続を照合する
  - test-onlyでclass2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件に既存共同更新・実旧準備/新採用共有反映を接続する。
  - 評価結果を既存batch初期統計へ渡し、実旧登録の全統計fieldとexact比較する。一標本fallbackも照合する。
  - 完了は同じモデル値/損失/全統計一致、評価前後のoptimizer state不変、対象GREEN・品質・Luna Task APPROVEDで判断する。
  - _Boundary: 上位対応のtest-only統合_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 2.3, 2.4, 3.1_

- [ ] 3. 依存境界と全回帰を検証する
  - ASTのexact symbol許可/禁止注入を先行REDにし、bare importも拒否するguardでGREENにする。
  - fresh新CPUで実分類器→損失→初期統計を実行し、全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧/golden差分、sourcehashと全11要件traceを記録する。
  - 完了は実測とLuna Task APPROVEDで判断する。全checkbox完了後のfeature GOは別ゲートとする。
  - _Boundary: 依存guard・対象spec・steeringの統合検証_
  - _Depends: 1, 2_
  - _Requirements: 3.2, 3.3_

## Implementation Notes
主担当が番号を指定して順次実装し、実GPT-6 Lunaの独立レビューで各taskを承認する。命名承認前のsrc/test作成なし。
Torch fork_rng/旧設定monkeypatch・必要時のglobal Random復元。日本語文書はapply_patchで保存し、PowerShellのstdin pipeを通すPython scriptはASCIIだけを使う。
共有../../venv、OMP/MKL各1。全pytest/PyrightはWindows tmp ACL/子Python起動のためrequire_escalated。旧production/両goldenは変更しない。
