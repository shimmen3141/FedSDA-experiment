# 実装タスク

- [x] 1. 警報後の固定参照損失収集と規定件数の状態を実装する
  - 既存設定・提案位置・参照ID順を明示入力し、提案次標本から同じ件数の系列をfloat化して保持する。全入力検証を更新前に終え、ready後は拒否する。
  - 旧session直接oracleでtarget2/3/5、固定ID順/逆順dictとfloat列・count/readyを比較する。constructor・型/範囲/位置/参照欠落/余剰・過剰追加の拒否とstate不変を確認する。
  - 到達後もsnapshotを繰返し参照して損失・位置が維持されることを確認し、採否起動・model/payload操作・終端消去の依存とcallbackを持たない境界を検証する。
  - LEGACY-004へ旧部分更新と新原子的拒否の再現テストを接続する。完了は対象成功→Luna APPROVED→主担当fresh再検証。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 5.1, 5.3_
  - _Depends: 承認済み要件/設計/命名、既存Python/pytest環境・CandidateModelTrainingAndAcceptanceSettings_
  - _Boundary: post_alarm_candidate_loss_collection.pyと同所有test、LEGACY-004記録_

- [x] 2. 旧client到達時機と既存採否部品への接続を検証する
  - 旧unbound observeで固定snapshot参照がliveモデル消失後も観測され、同じtarget到達回でfinalizeすることを比較する。
  - 変更不能snapshot・別実体・input dict変更・共有RNG/default dtype/device・keywordを確認する。途中/完了状態を消去せず返す。
  - 収集snapshotを既存採否APIへ明示変換して、旧参照選択/採否と比較する。完了は対象成功→Luna APPROVED→主担当fresh再検証。
  - _Requirements: 4.1, 4.2, 5.1, 5.2_
  - _Depends: 1、既存post-alarm-candidate-loss-evaluation public API_
  - _Boundary: 明示test統合。対象testのみ、production採否/モデル依存を増やさない_

- [ ] 3. 依存境界と全golden・独立起動を統合検証する
  - exact部品だけ同機能設定を許可し、torch/NumPy/旧/globalconfig/採否/FIFO/runtime/別収集の禁止注入を実行する。
  - 全tests・旧11/最終3goldenと旧import/torch/NumPyなしsmokeを実行し、15/15条件・cross-task・design境界・blockedなしを記録する。
  - 完了はLuna APPROVEDと主担当fresh検証、最終feature GO。roadmapと部分完成/後続責務を更新し、新FedSDA全体runの完成と混同しない。
  - _Requirements: 5.3_
  - _Depends: 1, 2_
  - _Boundary: 明示統合。対象test/依存境界test/spec記録/roadmap_

