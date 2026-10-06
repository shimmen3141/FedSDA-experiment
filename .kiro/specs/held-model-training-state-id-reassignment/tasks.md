# 実装task

要求/設計revision1・命名revision2・Luna graph PASS/task APPROVEDを正本に、依存順に実行する。

- [ ] 1. registryの単一ID付替えをTDDで追加する
  - 実旧confirmの順序/衝突/欠落/同ID、汎用signed、両ID拒否、古い記録/参照・optimizer蓄積保持/後続resetのtestを先にREDにする。
  - 既存ownerへ一APIと拒否項目名引数のみ追加、GREEN/既存registry対象/品質・独立Luna実diffレビューで完了。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3
- [ ] 2. 付替え前後の共同更新と統計をtest-onlyで接続する
  - 12条件（class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結）で3共同更新。最初の更新後に実旧confirmと新registry ID付替えを実行し、同optimizerで継続する。
  - loss/全parameter/grad/個別と共有optimizerをexact照合。統計は上位で別途付替え、registry操作単独で統計/pending/snapshotを変えないこと、3乱数/defaults保持を検証。
  - production追加なし、RED N/A。対象pytest/品質/独立Lunaレビューで完了。
  - Requirements: 2.3,2.4
- [ ] 3. 既存境界と固定goldenを統合検証する
  - 既存AST（変更/RED N/A）、fresh新CPU registry/optimizer/統計/pending接続と旧非importを確認。
  - 全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧差分空、承認hash/source hash/JUnitを実測記録。
  - 独立Luna task承認と主担当gateでtask3をcheckする。
  - Requirements: 2.4

## 全task完了後のfeature gate

別feature最終Luna GO（8条件/契約/参照/所有/設計/境界/ファイル計画）を得て、再開案内・roadmapを更新しcommit/pushする。
