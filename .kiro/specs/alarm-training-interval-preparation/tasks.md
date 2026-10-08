# 実装タスク revision4

逐次実行。既存基準環境、分割・評価保存・吸収・区間解決と旧oracleを再利用する。test/helper/局所名は外部下書きのAST抽出と追加命名承認を実装開始前に完了する。新moduleを作るtaskでexact依存guardと注入契約も追加し、未登録moduleを残さない。全taskの独立承認後、別fresh feature最終GOを行う。

粒度の根拠: runtime外部下書きは約160行、core testは既存実NN oracleを再利用したパラメータ化による約300行で、独立した複数機能の新実装ではない。Task 2の目安は実装・対象実測・独立レビューを含め1〜2時間。Task 3も既存区間解決/共同更新の公開test helperを使う6条件の接続検証で1〜2時間。Task 7の全pytestは直前実測332秒、証拠集計・品質・レビューを含め1時間程度を見込む。実測がこの規模を大きく超える場合はtask分割を再判断する。

- [x] 1. 不変記録と依存境界を統合する
  - 明示的なfoundation統合task。recordのfield、frozen/kw_only、借用Tensor、tuple列順、空の結果と前区間吸収先なしを先にtestしREDを記録してから実装する。
  - 入力recordは型/値検査をruntimeへ残し、概念IDは診断情報として保持する。結果は区間の位置・標本・概念IDの対応を保持する。
  - 対象の2moduleはIndexedObservedTrainingSampleとPreparedAlarmTrainingIntervalsの定義moduleのみ。これら2つだけのexact依存注入testを先行し、ImportFrom直接解決とImport拒否の両resolverへこの2つを登録する。まだ存在しないruntimeは登録も検証もせずTask 2へ残す。record/test/guardがGREENで独立レビュー承認されることを完了条件とする。
  - _Boundary: 位置付き入力と準備結果record_
  - _Requirements: 1.1, 1.3, 1.4_

- [x] 2. 区間準備と前区間の保存・吸収を実装する
  - 先に実旧処理との対照test・拒否testを書いてREDを記録し、共通検査、位置列の完全照合、公開区間分割、前区間全件の公開損失評価による事前検査、評価保存→吸収、結果返却を実装する。
  - 正規/負ID、前区間空、空FIFO、spanが短い/等しい/長い、capacity+1、概念ID None/あり、保存0/部分/全数と容量超過で、旧と標本順・値・統計/計数・Python最終stateが一致する。
  - owner/Random/record/位置/span/概念IDの型値、未保有、前区間先頭/後半の不正payloadで拒否され、保存・全owner・3乱数が不変。変化区間の分類器依存検査を後続へ残す境界もtestする。
  - 現行分類器への参照が事前評価と吸収で同じであることを確認する。FIFO、帰属、parameter/grad/optimizerと他モデルを変えない。
  - 全分岐を一度に完成させる短い組立関数であり、正常対照と拒否検証は同じ入力境界として扱う。拒否未実装などの途中状態をコミットしない。
  - runtimeのexact依存注入testを先行しguard/両resolverへ登録。実旧保存/吸収をstubせずGREENとなり、独立レビュー承認されることを完了条件とする。
  - _Boundary: 区間準備runtime_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3_

- [x] 3. 前区間準備から区間解決・学習までを検証する
  - 明示的なtest-only統合task。前区間保存→吸収→吸収後履歴統計での変化区間評価の順を記録wrapperで確認し、元の処理を実行する。
  - 2/4class×再利用/維持/候補開始を実旧_resolve_driftと照合し、続く共同更新と候補観測を含む全状態・parameter/grad/optimizer・3乱数を比較する。
  - 新準備が返したchange_interval_observationsからtraining_sampleとobserved_concept_idのtupleを取り出し、既存resolve_alarm_change_intervalのchange_interval_training_samplesとchange_interval_sample_concept_idsへ渡す。呼出し直前の吸収後統計を照合し、結果の区間開始位置とpayload対応も検査する。既存test_alarm_change_interval_resolutionのbuild_alarm_change_interval_resolution_oracleとassert_alarm_change_interval_resolution_matches_legacy、共同更新のrun_legacy_joint_updateを再利用する。新しい本番client接続は追加しない。
  - 最小件数直前/ちょうどとcapacity+1、保存前事前検査も接続条件に含む。旧の最小件数未満FIFO保持と新準備のFIFO不消費を区別して比較する。
  - 完了条件: 実旧対照と接続が実測され、独立レビュー承認される。
  - _Boundary: 準備と既存区間解決・学習の接続_
  - _Requirements: 2.1, 2.4, 2.5, 3.2, 3.3_

- [x] 4. 代表変異で検証の検出力を確認する
  - 明示的なtest-only検証task。区間逆転、保存/吸収順の逆転、事前検査欠落、余分なRandom消費、FIFO消費、診断概念IDの誤使用の代表source変異を外部スクリプトで適用する。
  - 変異ごとに対象testが失敗して誤動作を検出し、元source byteへ必ず復元した後に対象testがGREENになることを実測する。
  - 完了条件: 検出した変異と復元後のhash/GREENが記録され、独立レビュー承認される。通常sourceへ変異や検証用分岐は残さない。
  - _Boundary: 区間準備testの検出力_
  - _Requirements: 1.1, 2.1, 2.4, 3.2, 3.3_

- [x] 5. exact依存境界を検証する
  - source実importと新3moduleのexact guard/注入許可集合が一致し、wholemodule/private/禁止上位/再export/child/相対逸脱を拒否する。
  - 完了条件: 対象＋AST suiteが成功し、許可集合一致を独立レビュー承認される。
  - _Boundary: exact依存境界_
  - _Requirements: 1.4, 3.3_

- [ ] 6. 新CPUで独立動作を検証する
  - 旧/test importなしのfresh CPUで、2/4classの位置付き区間準備、評価保存/吸収、不消費・乱数を確認する。
  - 完了条件: fresh CPU smokeが成功し、実スクリプトと結果を独立レビュー承認される。
  - _Boundary: 新独立動作_
  - _Requirements: 1.4, 3.3_

- [ ] 7. 基準環境で全回帰と証拠を検証する
  - 明示的な統合検証task。対象実装commitで全pytest/JUnit、旧11・最終3golden、Ruff/Pyright/pip checkを実測し、固定旧source/golden差分が空であることを確認する。
  - 対象specのintegration-validation.mdへ要求12項目、各承認revision/hash、source hash、tested commit、taskレビュー、未検証範囲と追加forwardの制約を残す。
  - 完了条件: 主担当実測とJUnit照合を独立担当が確認し、task承認される。全pytestの独立再実行は既存合意に従い必須としない。
  - _Boundary: 基準環境の全回帰_
  - _Requirements: 3.3_
