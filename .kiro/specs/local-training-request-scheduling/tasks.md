# 実装タスク

- [x] 1. 学習要求の保留・回数算出・成功後消化を実旧へ照合する
  - missingmoduleの実RED後、機能別固定設定と単一counterを実装する。
  - interval1/2/5×L0/1/3の要求/端数flush/二重flushを実旧BaseClientの操作へ照合し、各eventのpending/呼出し順/予算を比較する。
  - 不正設定/ack・frozen/readonly・0予算・任意精度整数・例外後再試行を検証する。
  - 完成は全eventの実旧一致と拒否前後state保持の対象test成功、Luna実装APPROVEDで確認する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 3.1_
  - _Boundary: 要求counter/固定設定とその実旧対照test_

- [ ] 2. 算出予算を実共同学習へ明示接続して同値性を確認する
  - test-onlyで新schedule→前回の反復executor→成功ackを接続し、旧train_step/flush→実旧共同更新と比較する。
  - 32条件（class2/4、Adam/SGD、interval1/3、L0/2、共有更新/凍結）の混合要求列で、各境界の全NN/grad/optimizer/RNGと予算を確認する。
  - 不参加の正常skipでも保留を消化し、0成功lossを更新失敗と扱わないことを検証する。
  - 完成は実旧数値本体を置換しない比較test成功とLuna実装APPROVEDで確認する。
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 2.3, 3.1, 3.2_
  - _Boundary: 上位外側のtest-only明示接続、既存部品production変更なし_
  - _Depends: 1_

- [ ] 3. 依存境界と全回帰・統合証拠を揃える
  - 二moduleのexact禁止/許可import注入を先行し実RED→AST guard GREENを確認する。
  - stdlib fresh schedule smokeと新CPU反復接続/旧非import、対象/全pytest（ローカル旧11/最終3golden）・Ruff/format/Pyright/pip・旧固定差分/源内容hashを確認する。
  - 全11条件の実測・旧所見・限界/残る境界を保存し、Luna TaskAPPROVED/featureGOで完成を確認する。
  - _Requirements: 3.1, 3.2_
  - _Boundary: 依存test/対象specとroadmap、旧/golden変更なし_
  - _Depends: 1, 2_

## Implementation Notes
要求はtrain_step一回が一件、標本/optimizer更新の総数とは異なる。
0予算でも間隔到達/明示flushでpending>0なら正常完了確認を行う。未参加の空lossも正常成功として消化する。
globalPythonを旧呼出し時に借用Random開始へ合わせ、finally復元する。Torch状態もfixture/testで復元する。
学習例外時にackしない。retryは外側明示、既済NN更新はrollbackしない。

