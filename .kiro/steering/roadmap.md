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

- 共有特徴抽出部への再接続は`../specs/shared-feature-extractor-attachment/README.md`。全3tasks完了・命名revision3・Luna最終feature GO、対象33/AST込み540、全4033 passed/3 skipped/1既存warning。品質/fresh新CPU/旧固定差分/全7条件traceを確認。参照交換/接続検証と上位のoptimizer操作を分離する。実旧12条件×3stepの全loss/NN値/grad/optimizer state一致、旧owner非流用/不変を確認。共有元選択・同期parameterロード・候補登録・モデル一覧の所有・新client/全体runは後続。新たな旧正常系不具合は観測していない。

- optimizer状態の所有と明示リセットは`../specs/parameter-optimizer-state/README.md`。全3tasks完了、命名revision3/Luna最終feature GO。固定parameter列/設定を借用し現在optimizerと成功後交換を所有、値/gradと旧借用binding保持。実旧reset12条件・NN24条件×3stepを照合、対象47/AST込み554、全4000 passed/3 skipped/1既存warning、品質/fresh新CPU smoke/旧固定差分確認。全9要件/hash/採否は対象specが正本。モデル所有/共有接続/正式登録/reset時機/新client/全体runは後続。新たな旧正常系不具合は観測していない。

- モデル別学習標本の保持は`../specs/model-training-sample-storage/README.md`。全3tasks完了、命名revision1/Luna最終feature GO。追加順/空列/重複/借用payload/snapshot構造分離/サーバID一回対応を独立移植。実旧保存16条件・抽出48条件×3反復を照合、対象94/AST込み581、全3933 passed/3 skipped/1既存warning、品質/fresh新CPU smoke/旧固定差分確認。全10要件/hash/採否は対象specが正本。正式登録pop/上書き、モデル/optimizer所有・同期、統計/評価store、新client/全体runは後続。新たな旧正常系不具合は観測していない。

- ローカル学習要求の保留と実行回数は`../specs/local-training-request-scheduling/README.md`。全3tasks完了・命名revision1/Luna最終統合GO。要求counter/間隔/試行予算/正常終了の明示確認を独立移植。実旧9条件の要求列と32条件の実NN反復接続でcounter/予算/順序/loss/parameter/grad/optimizer/RNGを照合。対象70/AST込み539、全3821 passed/3 skipped/1既存warning、stdlib/new CPU4-step smokeを主担当・Lunaが確認。旧source/golden変更なし、全11要件/hash/採否は対象specが正本。完全run設定/CLI、要求・警報/ラウンドflush位置、標本/optimizer所有、通信/診断・新client/全体runは後続。新たな旧正常不具合は観測していない。

- 保有モデルの共同学習反復は`../specs/held-model-joint-training-iterations/README.md`。全3tasks完了・命名revision2/Luna最終統合GO。借用モデルID対応と毎回新抽出→共同更新を接続し、実旧72条件で全batch/loss/parameter/grad/optimizer/RNGをexact照合。対象94/AST込み533、全3721 passed/3 skipped/1既存warning、主担当・Lunaのfresh CPU4回Adam更新/旧非import成功。0/skip/拒否・後続失敗時の先行更新保持も確認。全12要件・hash・採否の正本は対象spec。optimizer/標本所有・回数算出/pending/interval・候補/同期/診断counter・PCGrad・新全体runは後続。新たな旧正常不具合は観測していない。

- 保有モデルの学習バッチ抽出は`../specs/held-model-training-batch-sampling/README.md`。全4tasks完了・Luna最終統合GO。順序付きモデル別観測列から保有/件数で参加選別し、借用Randomで復元なし抽出とcatを実装。対象56/AST込み459、全3591 passed/3 skipped/1既存warning、主担当/LunaのfreshCPU抽出→共同更新smoke成功。FIFO位置解決/IDからNN・optimizer対応はtest-onlyで実旧12条件×3stepへ完全照合した。全12要件・hash・採否・実測は対象specへ記録。標本ストア所有/追加/帰属/ID統合・反復・optimizer所有/reset・新全体runは後続。新たな旧正常不具合は観測していない。

- 一回共同更新は`../specs/joint-model-parameter-update/README.md`。全3tasks完了・Luna最終統合GO。確定済み参加batch列から共有forward一回・標本数加重BCE/CE・backward一回・共有step→個別step入力順を移植。対象77/AST込み438、全3493 passed/3 skipped/1既存warning、主担当/LunaのfreshCPU smoke成功。空共有はoptimizer=Noneで個別学習を独立検証し、旧LEGACY010は未修正/通常過去影響未確認。参加抽出は上記specで独立移植した。反復・optimizer所有/reset・候補進行・通信/診断counter・PCGrad・新全体runは後続。全17要件・hash・採否・実測の正本は対象spec.json/review.md。

- optimizer設定/生成は`../specs/parameter-optimizer-construction/README.md`。全3tasks完了・Luna最終統合GO。Adam standard/AMSGradとlrのみSGDを別型にし、明示Parameter列へ標準optimizerを生成。実旧groups/state/3step全値を照合、新NN共有/個別tuple接続、346件・全3364 passed/3 skipped・fresh CPU smokeを検証。9条件・hash・採否/実測の正本は対象spec。LEGACY010へ空共有optimizer生成失敗を記録、旧未修正/通常過去影響未確認。単回共同更新は上記specで完了。共有optimizer所有/reset・学習反復・候補進行・新全体runは後続。

- 最終モデル構造の移植は`../specs/residual-adapter-model-architecture/README.md`。全3tasksを完了しLuna最終統合GO。共有特徴抽出/非線形残差/classifierのCPU32生成・forward、実旧state/RNG/gradient、test-only確率/snapshot接続を実装。対象＋AST319件、全tests3284 passed/3 skipped、CPU fresh smokeを検証した。承認hash・12条件・実測・採否は対象specに記録。追加の旧不具合は観測していない。学習・optimizer・共有付替え・候補進行・新全体runは後続。

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
- 有界損失の不変集計は`../specs/bounded-loss-moments/README.md`。一系列の件数・保存平均・偏差平方和、旧演算順の追加、n2以上の平均/標本分散と上位への明示test接続を実装。2026-10-04の全testsは2728 passed /3 skipped、旧11/最終3golden更新なし。依存境界・stdlib独立起動を確認、最終統合判定と承認の正本は同spec.json/review.md。モデル/class所属、batch seed、用途別baseline方針、学習・登録・統計merge、新全体runは後続。
- クラス統計の旧説明不整合と、不正class入力で旧全体統計が部分更新する事実をLEGACY-005/006へ記録。正常client/過去成果への影響は未確認。旧productionの修正と今回の数値移植を分ける。
- 用途別損失基準値は`../specs/loss-baseline-selection/README.md`。監視n1/clip、警報区間再利用n2/平均0除外、警報後履歴n2/平均0保持を、単一集計から選ぶpure方針として実装。上位への明示test接続、入力検査コピー・共有状態とexact上流型依存を検証。2026-10-04の全testsは2829 passed /3 skipped、旧11/最終3golden更新なし。全回帰と最終判定の正本は同spec.json/review.md。モデル/class統計map・seed/merge・snapshot時機・候補進行・新全体runは後続。
- モデル全体・クラス別損失統計は`../specs/model-and-class-loss-statistics/README.md`。明示seed/一括置換、独立snapshot、帰属損失の原子的更新を実装し、旧各更新への直接照合と監視/履歴参照への明示test接続を確認。2026-10-04の全testsは2885 passed /3 skipped、旧11/最終3golden更新なし。承認と最終判定の正本は同spec.json/review.md。モデル実体・class上限、batch seed算出、merge/ID変更/削除時機・学習・新全体runは後続。
- LEGACY-006には旧不正class入力による部分更新と新storeの原子的拒否の同入力テストを追記。旧production自体は未修正で、正常clientや過去成果への影響は未確認。
- batch損失からのモデル初期統計は`../specs/batch-loss-statistics-initialization/README.md`。外部計算済みloss/labelsから旧torch reduction順の全体/class集計を生成し、singleton非対称・class昇順/欠落と旧24ケースを照合、store/update/baselineへの明示test接続を確認。2026-10-04の全testsは2965 passed /3 skipped、旧11/最終3golden更新なし。承認/最終判定の正本は同spec.json/review.md。事前学習の逐次Welfordseed、model prepare/forward/学習/登録/送信、merge・新全体runは後続。
- 旧空batch登録でNaN初期統計を保持する事実をLEGACY-007に再現・同入力対照とともに追跡。正常client/過去成果への影響は未確認、旧productionは未修正。
- モデルID対応後の損失統計選択は`../specs/model-id-mapped-loss-statistics-selection/README.md`。一回ID対応/max全体件数・先着同数/欠落・zeroだけのserver補完、モデル/class順と旧18ケースの全field照合、独立コピーとstore/update/baselineへの明示test接続を検証。2026-10-04の全testsは3041 passed /3 skipped、旧11/最終3golden更新なし。モデル・学習データ・予測重み・現在の帰属IDの対応、登録/通信/server集計・新全体runは含めない。今回新たな旧正常不具合は未観測、既存LEGACY001–007は修正状態を変えない。承認/最終判定の正本は同spec.json/review.md。
- 次は候補開始・終了の進行、警報後のモデル帰属/client調整と、その前提となるモデル学習・統計mergeを依存順に仕様化する。未作成specの命名・実装を先取りしない。
- サーバ向け損失平均集約は`../specs/server-loss-mean-aggregation/README.md`。参加選別済みのclient順に統計件数で平均を加重し、M2zero/空・zero時Noneを両旧サーバ20ケースへ照合。上位whole server保持/置換→ID補完→store更新/baselineを明示test接続。2026-10-04の全testsは3103 passed /3 skipped、旧11/最終3golden更新なし。参加判定・モデルパラメータ集約/通信/状態保存・新全体runは後続。極大件数の範囲超過/除算失敗はLEGACY008へ再現と新拒否を記録、通常影響未確認/旧production未修正。承認・最終判定の正本は同spec.json/review.md。

SINE goldenの参照条件と所属はsingle-run-executionのreference-inventory.mdへ棚卸し済み。
今後は必要な追加契約を確認し、型付き設定と単一run基盤へ手法の処理部を接続する。
`configuration-foundation`の後続計画4〜8は未承認・未完了のまま保持する。

## 承認と検証

2026-10-03の追加指示により、今後は要求・設計・task・命名・実装について、gpt-6-lunaレビューと主担当の有用指摘の反映を承認として進める。レビューと採否・承認対象を記録し、毎段階で人間承認を再要求しない。
命名を変えた場合は承認を解除し、変更箇所をレビューする。
移植対象のgoldenと機能テストを各単位の検証へ接続する。
- 候補パラメータ初期化は`../specs/candidate-parameter-initialization/README.md`。全3tasksを完了しLuna最終統合GO。現在の学習先/評価済み最小loss/全保有等平均から独立snapshotを返す部品を実装し、全15dtype・順序・copyを旧helperへ直接照合した。test-onlyの公開候補評価mean接続とexact依存/CPU fresh smoke、全tests3182 passed/3 skippedを検証済み。承認・hash・9条件の証拠は対象specに記録した。LEGACY009へ極大float32平均の非有限化を記録、旧productionは未修正・通常影響未確認。モデル生成/適用/optimizer/学習/候補session進行/新全体runは後続範囲。
