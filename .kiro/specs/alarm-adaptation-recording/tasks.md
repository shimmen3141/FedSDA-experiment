# 実装タスク revision3

- [x] 1. 適応記録ownerと警報完了の記録を実旧対照つきで実装する
  - 命名承認後にtestを追加してRED、source実装でGREEN。実旧の5結果・2/4classと全field/件数/位置、拒否時不変、snapshot、記録以外の状態不変を照合する。
  - 再利用と異なるIDの双方向対応、入力recordの後続破壊から保存copyの独立、上流5結果との集合一致を確認する。
  - 注入契約のREDを経て両resolverへ実importのexact guardを追加する。対象test・境界suite・Ruff・srcのPyrightと独立レビュー承認で完了。
  - _Boundary: evaluationの記録ownerとruntime変換、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1_

- [ ] 2. 検出力と新実装単独の接続を確認する
  - 実sourceのfield/件数/位置の破壊と検査を更新後へ移す変異を1つずつ実行し、元byteへ戻す。未検出はtestで補う。
  - fresh CPUで旧/test importなしに新上流完了と記録ownerを接続し、2/4class×5結果を確認する。証拠と復元後GREENを独立承認されて完了。
  - _Depends: 1_
  - _Boundary: testと検出力/独立動作の証拠_
  - _Requirements: 2.1, 2.2, 2.3, 3.1_

- [ ] 3. 固定基準の全回帰とfeature最終GOを確定する
  - source/testをcommitして全pytest/JUnit、旧11/最終3golden、Ruff/Pyright/pip check、承認hash/固定旧diff/source hashを実測し、独立レビューへ照合を依頼する。
  - 別fresh reviewerのfeature最終GO後に進捗と再開案内を更新する。新全体runと通知/sessionは未完了と記録する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 3.2_
