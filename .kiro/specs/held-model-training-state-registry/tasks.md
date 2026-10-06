# 実装タスク

- [ ] 1. 保有状態の登録・置換・現在参照取得を実装する
  - 未実装公開APIのimport RED後、対応の登録前検証、初出順/同ID置換、readonly record/snapshot、現在optimizerのbinding生成を実装する。
  - 負/ゼロ/正/大整数、型/対応/parameter不正の拒否、未登録取得、値/grad/state/RNG不変、reset後の新旧binding参照を確認する。
  - 完了は対象単体GREEN、Ruff/format、Luna Task APPROVEDで判断する。
  - _Boundary: 保有状態registryと対象単体test_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3_

- [ ] 2. 実旧登録から共同学習への明示接続を照合する
  - test-onlyで実旧登録のmodel dict境界と、新registryの登録/同ID置換順を対照する。旧損失統計やpendingは本registryの完了範囲へ含めない。
  - class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件で、登録/置換/reset→現在binding→既存抽出/反復を3step接続する。
  - 完了は全loss/NN値/grad/optimizer state/終端Random一致のGREEN、品質、Luna Task APPROVEDで判断する。
  - _Boundary: 上位対応のtest-only統合_
  - _Depends: 1_
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 3.1_

- [ ] 3. 依存境界と全回帰を検証する
  - AST禁止/許可注入を先行REDにし、新moduleのexact依存guardでGREENにする。
  - fresh新CPUで登録→reset→現在binding→共同更新を実行し、全pytest旧11/最終3golden・Ruff/format/Pyright/pip・旧固定差分/sourcehash・全11要件traceを記録する。
  - 完了は実測記録とLuna Task APPROVEDで判断する。全checkbox完了後のfeature GOは別ゲートとする。
  - _Boundary: 依存guard・対象spec・steeringの統合検証_
  - _Depends: 1, 2_
  - _Requirements: 3.2_

## Implementation Notes
番号を指定して主担当が順次実装、各taskを実GPT-6 Lunaが独立レビューする。命名承認前のsrc/test追加なし。
Torch fork_rng・旧設定monkeypatch・Python global Randomのfinally復元を使う。
共有../../venv、OMP/MKL各1。全pytest/PyrightはWindows tmp ACL/子Python起動のためrequire_escalated。旧production/両goldenは変更しない。
