# Implementation Plan

- [x] 1. client設定と初期状態の独立した組立を実装する
  - 実旧__init__対照と不正設定/状態の拒否testを先に書く。値/grad/optimizer/mode、複数clientの分離、3RNG不変が確認できれば完了。
  - _Boundary: runtimeの設定・assembly_
  - _Requirements: 1.1, 1.2, 1.3, 2.1_
- [x] 2. 組立済みclientから観測・round・終端へ接続する
  - 実旧の連続操作と異常blockを先にtestする。各段の全owner・optimizer・記録・乱数の一致と、概念不干渉/readyの読取りが完了条件。
  - _Boundary: runtimeのclient操作統合_
  - _Depends: 1_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
- [ ] 3. 新構成だけの接続・独立レビュー・全回帰を完了する
  - exact依存と共用fresh接続を追加し、独立指摘採否・全pytest/JUnit・品質検査を確認して完了する。
  - _Boundary: 接続・検証_
  - _Depends: 2_
  - _Requirements: 4.1_

## Implementation Notes
- 要求/design/task graph gateを主担当inlineでPASS、全10要求を3taskへ対応付けて委任承認。kiro-impl --review off、子エージェントなし、全task後に独立CLI1回。NN状態/複数部品の接続はHaiku優先だが、前specまでの週間上限が続く場合は代替Luna medium。
- RED前に実旧__init__/空process_batch/local_train(0)/flushを実行しPASSを確認。新module未存在の収集エラーをREDには数えない。
- task 1: stubのNotImplementedErrorを確認（35件。Adam設定引数のtest側誤りを修正後もRED）。実装後、test側の計数snapshotメソッド名/Adam選択肢を実定義に修正して35 passed（3.74秒）。実旧deepcopyはgradをNoneにするため、初期gradの保持案を実旧に合わせて要求/設計へ補正。初期入力gradは変更しない。両optimizer・2/4クラス・no_grad・複数client・拒否前forward/copyなし・3RNG不変を確認した。
- task 2: 観測/round/flush stubで18 failed（NotImplementedError）、実装後に53 passed（4.64秒）。test helperの引数、旧新規Adamのconfig条件を明示し、旧の未学習IDの0 entryを新計数ownerの欠落=0へ対応付けた。旧process_batch/local_train/flush本体を代役なしで通し、更新の戻り値だけwrapperで記録。各段のparameter/grad/optimizer/store/統計/予測/遷移と3RNGを照合した。追加の境界値/送信ready/optimizer分離testを含む最終接続検証へ進む。
- 最終接続検証: 3120 passed（対象55件＋依存3064件＋共用fresh smoke1件、90.80秒）。追加件数0の評価storeは空のcollectionを保持する上流契約にtestを合わせた。Ruff check/format（307ファイル）、Pyright 0 errors/0 warnings、pip check、names成功。freshでは学習済みAdamから組み立て、2/4クラスで観測7件→6回の実学習→残余flush/二度目no-opを確認。source/testを凍結し独立レビューへ出す。独立レビュー・全pytestは未実施。
