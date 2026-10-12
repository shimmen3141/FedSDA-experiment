# Implementation Plan

- [x] 1. 単一clientの逐次学習を移植する
  - 実旧clientをoracleとして予測、損失、全保持、pending予算、optimizerと乱数をtest先行で照合する。
  - 不正入力は変更前に拒否し、真概念の入替えで学習が変わらないことを観測する。
  - _Boundary: FedAvg client_
  - _Requirements: 1.1, 1.2, 1.4, 2.3, 4.2_
- [x] 2. 初期準備とserver同期を全体runへ接続する
  - 完全設定、実旧事前学習/同期、共通executorと計測を接続する統合task。
  - 六datasetと更新/同期の境界で全体runの予測・最終値・optimizer・統計・保持標本・通信・乱数を実旧と照合する。
  - _Boundary: FedAvg settings, factory, server, execution integration_
  - _Depends: 1_
  - _Requirements: 1.1, 1.3, 1.4, 2.1, 2.2, 3.2, 4.2_
- [x] 3. 共通指標と成果物を接続する
  - 共通評価へ接続し、予測と生成末尾、通信/計算差分、統計/学習量の表を保存する。
  - 実runのJSON/NPZ roundtrip、存在する片方の上書き拒否、stale記録拒否、ゼロ処理shapeを確認する。
  - _Boundary: FedAvg metrics, artifact integration_
  - _Depends: 2_
  - _Requirements: 3.1, 3.2, 3.3, 4.2_
- [x] 4. 既存掃引入口で比較可能にする
  - 正式構成名の登録、設定型/実行/保存/再開dispatch、説明、共用fresh、依存境界を接続する。
  - 混合計画でFedDrift専用軸の非増殖、既存ID不変、逐次/別process実行・再開・CSV・読取専用計画を観測する。
  - 独立実装レビュー後に指摘反映、全pytest、固定旧/golden不変、品質/fresh/Linuxを検証する。
  - _Boundary: sweep integration and validation_
  - _Depends: 3_
  - _Requirements: 4.1, 4.2_

## Implementation Notes
要求・設計の主担当gateとtask graphのinline sanity確認を行った。実装前独立CLI仕様レビューはMajor1修正後PASS、spec.jsonにsessionを記録した。

仕様レビュー初回Major1を採用: serverが呼ぶclient専用配布操作の型/入力/呼出し元とexecutorとの関係をdesignへ明記。重大箇所の再確認PASS。

Task1: 実旧Obliviousの予測/全体損失/学習/optimizer/データ/Random対照、拒否不変、概念不干渉14passed。REDは予測記録空と実旧1件の不一致、module収集失敗ではない。

Task2: 6dataset×Adam/SGDの実旧setup/全体run対照を含む26passed。SGD oracleがweight_decayを持たない型を読むtest helperの誤りを修正。乱数、caller Torch隔離、集約統計まで一致。

Task3: 全体run・準備/round/終端計測、共通指標、JSON/NPZ保存と拒否を含む43passed。REDは全体run戻り値未接続。準備の標本数は7×2epoch=14で、client数で二重計数しない期待値へ修正。

Task4実装: FedAvg対象57passed（件数重み不均等/空参加、配布拒否、設定再検査を追加）、既存FedDrift掃引＋依存境界を含む3185passed、Pyright0。この時点では全pytest・独立実装レビュー前であり、最終結果は以下に記録する。


独立実装レビュー: Haiku 5.5 medium指定、session 8bdfae19-a09d-4e82-b0f5-c451e10b58a2、is_error=false/実モデルclaude-haiku-5-5、PASS（Blocker/Majorなし）。3148passed（FedAvg57＋依存3091）、fresh直接exit0/Ruff成功を独立実行。全pytest/Linux/Pyright/names/identityは独立再現していない。16flowsの表示は既存8適応シナリオ×2で、追加した全体runを含む総数ではない。
Minor1を採用しCSV名をrun_summary.csvへ修正。Minor2/3を採用し6dataset既定値の実旧対照と、FedAvg実行途中の失敗位置/cause/成功済み状態/乱数隔離のtestを追加。Minorだけなので再レビューは不要。
任意Aのoptimizer設定アクセサ追加は既存共通型のAPIを変えるため今回見送り。現在の所有stateの固定設定を借用する役割に限定する。任意Bのモデル数1は単一モデル構造の不変条件から導く値（モデル集合を持たない）でありdesignに明記済み。任意Cの統計n=0補充は事前学習標本数>=1で到達不能のため変更しない。レビュー本文の98b98b1b6b40はFedSDAのIDを指し、FedDriftではない。


レビュー修正後: FedAvg対象64passed。Ruff lint/format（346files）、Pyright0、pip check成功。WSL UbuntuでFedAvg57＋既存FedSDA環境golden3=60passed、追加の既定値対照6passedと実行途中失敗1passed。共用freshのFedAvg実run/計測/保存は主担当と独立担当ともexit0。文書の3手法比較コマンドはprint-planで3run/全missingを確認し、出力directoryを作っていない。変異testは既定の省略。新実装用goldenの追加とLEGACY/IMPROVE適用は後続。

最終検証: 固定Windows venvの全pytestは13724passed/8skipped/9warnings、701.21秒、exit0。JUnitは13732件、failure/errorとも0（../../venv/refactoring-tests/fedavg-experiment-run-full.xml）、ログは同directoryのfedavg-full.log。既存test_regressionとtest_proposed_regression、追加FedAvg64件、共用freshも成功。固定旧748c3aaに対する旧実装・旧回帰test・旧golden・toolsの差分は空。

検証対象source/test commitはbf7a5a0bf3902b916a47fe911f6b5373b1d8ca3d。Pythonと既存goldenの450pathをLF正規化したSHA256は19d132d62a6692a014b6319711f9d225ea202e4876a07a24b4f859575dbafc85。完了文書commit後にidentityで同一性とclean treeを確認する。新しいFedAvg goldenは未追加であり、今回の旧動作照合は実旧を直接実行した対照testによる。
