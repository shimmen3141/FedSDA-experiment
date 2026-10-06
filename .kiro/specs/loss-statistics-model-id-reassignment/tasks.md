# 実装task

要求・設計・命名revision1とLuna graph PASS/task APPROVEDを正本として依存順に実行する。

- [x] 1. 統計storeの単一ID付替えをTDDで追加する
  - 順序/元欠落/先衝突/同ID/汎用signed/0件・class順、両ID拒否と状態保持、独立取得値・後続観測のテストをREDにする。
  - 既存storeに一APIだけ追加。実旧BaseClient.confirm_model_registrationの統計値/モデル順を照合。負元IDの実旧対照と汎用signed直接期待値を区別する。
  - 対象pytest/品質/独立Luna実diffレビューをgateとする。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2
- [x] 2. 正式ID確認と保留解除をtest-onlyで接続する
  - class2/4×delay1/2の4条件で実旧登録/生統計更新/ready/確認と、新producer/初期統計/store/pending/現在統計取得/付替え/明示clearを照合。
  - 付替えだけでは保留record・残回数・登録時snapshotを変更しないこと、現在統計全field・parameter/grad/RNG/既定環境の保持を確認。production追加なし、RED N/A。
  - 対象pytest/独立Lunaレビューをgateとする。
  - Requirements: 2.3,2.4
- [ ] 3. 既存境界と固定goldenを統合検証する
  - 依存追加がないことを既存ASTで確認（guard変更・RED N/A）。fresh新CPU snapshot/store/pending/付替え/clear接続・旧非importを確認。
  - 全pytest旧11/最終3golden、Ruff/format/Pyright/pip/diff、固定旧・golden差分空、source hash/JUnitを実測記録。
  - 独立Luna taskレビュー後に主担当gateでtask3をcheckする。
  - Requirements: 2.4

## 全task完了後のfeature gate

別feature最終GO（8条件/設計/ファイル計画/所有）を得て案内更新・commit/pushする。
これはtask3の完了条件ではなく、全task完了後の最終手順とする。
