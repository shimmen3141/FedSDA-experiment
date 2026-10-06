# 実装task

graph独立Luna PASS済み。requirements/design revision1・naming revision3が正本。主担当で順次実行する。

- [x] 1. 分類器snapshotの契約をTDDで実装する
  - 型/全parameter環境・有限性、独立storage/native順/全値、複数呼出、状態保持、実旧get_params対照をtest先行RED→production GREENで確認する。
  - 対象pytest、Ruff、Pyright、実diffの独立Luna承認を完了gateにする。
  - Requirements: 1.1,1.2,1.3,2.1,2.2,3.1
- [x] 2. 学習後のsnapshotと候補初期化をtest-only接続する
  - 12条件（class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結）で実旧/新共同更新後のsnapshot全値を対応し、既存initializer→新native復元とoptimizer状態保持を確認する。
  - production変更なし。test-only REDはN/A。対象pytest、品質、独立Luna承認をgateにする。
  - Requirements: 1.1,1.2,3.1,3.2
- [ ] 3. 依存と固定基準を統合検証する
  - exact symbol許可/拒否注入を先にRED→guard GREEN。fresh native CPU smokeでsnapshot→initializer→復元と旧importなし。
  - 全pytest（旧11/最終3golden含む）、Ruff/format/Pyright/pip、diff、固定旧/golden/test差分なし、source hash/JUnitをintegration-validation.mdへ記録する。
  - task独立承認後にcheckboxを更新し、全taskの整合性・要件coverageを別のLuna feature GOで確認する。現在地を更新し、適切な単位でcommit/push。
  - Requirements: 3.2,3.3
