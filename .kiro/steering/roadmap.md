# リファクタリングの進行

## 全体方針

旧基準`748c3aa`と同じGit履歴の`refactor/architecture`ブランチで、新APIへ移行する。
クリーンアーキテクチャの依存方向・差し替え境界・機能別配置を採用する。
詳細の正本は`docs/research/refactoring-policy.md`。新srcのパッケージ境界と機能ごとの設定型を段階的に実装している。

## specの候補と依存順

1. `configuration-foundation`: 初回は最終構成に必要な型付き設定と検証。選択肢、preset、保存表現は必要になる段階で追加する。
2. `single-run-execution`: SINEデータ供給と単一runの実行順序。12実行タスクを完了し、2026-10-03の統合検証はGO。承認・検証証拠は同specのspec.jsonとreview.mdを参照する。
3. 最終FedSDAの予測・割当・監視・候補検証: 最初は`fixed-share-prediction-weights`で予測重み状態と更新を独立移植する。続いてモデル出力混合、ClassESR監視、候補の将来損失判定、FIFO帰属・client調整を責務別に仕様化し、1・2へ接続する。
4. モデル・学習・サーバ同期・統合: 3の境界に沿って移植する。
5. 評価・新成果物・golden比較と必要なbaseline・ablation: 各移植単位から段階的に接続する。
6. 新CLI・掃引・文書を仕上げ、新ブランチから参照用旧構成を除く。

3の最初のspecは`fixed-share-prediction-weights`。続く`weighted-class-probability-prediction`も3実装タスクを完了し、Lunaによる最終統合判定GO。すでに得られた出力の確率化・混合・クラス判定・観測後損失に限定する。先の関数・変数を全て今決めることはせず、担当単位の実装前に一覧化する。
機能境界と状態所有者を先に整理し、最小設定を小規模な実行経路へ接続する。
汎用の設定基盤を全て完成させてから手法を移植する順序にはしない。

## 現在の状態

- worktreeとcc-sdd導入: 完了。
- アーキテクチャ規約・命名レビュー手順: 作成済み。
- 要求・設計・初回task: 人間承認済み。初回13タスクに既存成果の改名task 9.1を追加した。
- 設定基盤の初回承認範囲と後続未完了範囲: `../specs/configuration-foundation/spec.json`が正本。
- 初回範囲: 基礎例外、各機能の固定条件、値・組合せ検証、不変な部分集約型。実験実行や完全な設定型は含めない。
- 正本一覧は対象specのREADME。命名再検討表は候補・履歴であり、実装する正式名はnaming.mdのみ。
- 単一run基盤: `../specs/single-run-execution/README.md`。12実行タスクを完了。旧SINEとの準備後データ・乱数・処理順の照合と既存goldenを含む全テストを確認した。新しい学習・判断処理の研究指標は未移植。
- 予測重みの移植は`../specs/fixed-share-prediction-weights/README.md`。4実装タスクはLuna承認・検証済み、最終統合判定はGO。全testsは2172 passed / 3 skipped、旧production・golden差分なし。承認・feature GOの正本は同spec.json。モデル混合・学習と全体runはこの部品の完了範囲に含めない。
- 分類予測の数値部品は`../specs/weighted-class-probability-prediction/README.md`。3 tasks完了、全20条件をLunaが確認し最終統合GO。2026-10-04の全testsは2281 passed / 3 skipped、旧golden11ケース・最終golden3ケースを更新せず通過。二値/多クラスの確率化・混合・予測・平均有界損失と、重み状態へのテスト接続が完了範囲。新FedSDA全体run、モデルforward・学習は未移植。
- 全体・正解クラス別損失監視は`../specs/class-conditional-loss-monitoring/README.md`。3 tasks完了、全20条件をLunaが確認し最終統合GO。2026-10-04の全testsは2390 passed / 3 skipped、旧golden11ケース・最終golden3ケースを更新せず通過。単一e-SR・ClassESR混合・global候補位置・reset・明示損失接続が完成範囲。モデル統計からのbaseline推定、警報後操作、新全体runは後続範囲。
- 候補の将来損失評価は`../specs/post-alarm-candidate-loss-evaluation/README.md`。3 tasks完了、15条件をLunaが確認し最終統合GO。2026-10-04の全testsは2504 passed / 3 skipped、旧11/最終3goldenを更新せず通過。外部収集済みlossによる現行優先の既存適合選択・二分区間候補採否・診断値が完成範囲。収集session・model操作は未移植。旧採否/理由の丸め不整合はresearch.mdへ別修正候補として記録した。
- 保留標本位置FIFOは`../specs/pending-training-data-assignment/README.md`。3 tasks完了、14条件をLunaが確認し最終統合GO。2026-10-04の全testsは2580 passed / 3 skipped、旧11/最終3golden更新なし。位置の明示追加・超過解放・非破壊分割・全消費とpublic監視span接続だけが完成範囲。モデル帰属・payload・学習・警報後進行・終端方針は未移植。
- 旧実装の不具合・改善候補の正本は`../../docs/research/implementation-findings/README.md`。候補理由の丸め差、短い警報後の割当済み標本残留、終端FIFO末尾の未帰属を別記録で追跡する。今回の移植で修正しない。
- 警報後の候補/参照損失収集は`../specs/post-alarm-candidate-loss-collection/README.md`。3 tasksを実装し15条件の証拠を確認。2026-10-04の全testsは2664 passed /3 skipped、旧11/最終3golden更新なし。提案次位置からの固定参照loss系列、原子的拒否、規定件数到達とimmutable copy・既存採否への明示test接続が範囲。設定モジュールと宣言型だけへ依存許可を限定した。承認と最終統合判定の正本は同spec.json/review.md。
- 旧sessionが不足参照入力で部分更新する事実はLEGACY-004へ追跡記録を追加。正常clientへの影響は未確認で旧productionは変更しない。
- 次は候補開始・終了の進行、警報後のモデル帰属/client調整と、その前提となるモデル統計・学習を依存順に仕様化する。未作成specの命名・実装を先取りしない。

SINE goldenの参照条件と所属はsingle-run-executionのreference-inventory.mdへ棚卸し済み。
今後は必要な追加契約を確認し、型付き設定と単一run基盤へ手法の処理部を接続する。
`configuration-foundation`の後続計画4〜8は未承認・未完了のまま保持する。

## 承認と検証

2026-10-03の追加指示により、今後は要求・設計・task・命名・実装について、gpt-6-lunaレビューと主担当の有用指摘の反映を承認として進める。レビューと採否・承認対象を記録し、毎段階で人間承認を再要求しない。
命名を変えた場合は承認を解除し、変更箇所をレビューする。
移植対象のgoldenと機能テストを各単位の検証へ接続する。
