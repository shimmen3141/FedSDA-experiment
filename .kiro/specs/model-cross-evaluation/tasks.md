# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 記録と設定
- [x] 1.1 診断の記録のownerと、束の標本の上限を足す
  - 診断の記録（記録の型と検査、追加、写し）と、スカラーの設定のfieldを足す。束を作る箇所（既存のtestと共用script）を直す。新しいmoduleの依存の許可集合を登録する。
  - 完了: 記録のownerの単独のtest（順、検査、不正で不変）、束のtest、依存境界のsuiteが成功する。
  - _Boundary: CrossEvaluationRecordStore、ClientCrossEvaluationRecord、FedsdaRunClientScalarSettings_
  - _Requirements: 3.3, 3.4, 6.2_

- [x] 2. クロス評価
- [x] 2.1 clientの評価を、実旧のclientの評価との対照つきで実装する
  - 旧の設定の差し替えへ、評価標本の追加の件数と、評価の標本の上限を足す。同期したラウンドの途中の状態で、実旧のclientの`evaluate_model`・`evaluate_model_diagnostics`と照合する対照testを先に書く。
  - 評価の関数、結果の型、clientの2つの操作を実装する。依存の許可集合を登録する。
  - 完了: clientの対照が、2値・多クラスで、モデルの全部の組について、結果と乱数の一致を示す。拒否のtest、評価が変えない状態のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: evaluate_candidate_model_on_target_model_samples、ClientModelCrossEvaluation、ModelPairCorrectnessCounts、FedsdaRunClientの2つの操作_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.3, 5.2, 6.2_
- [x] 2.2 サーバのクロス評価を、実旧のサーバのクロス評価との対照つきで実装する
  - ラウンドの対照（登録→集約→クロス評価→配布→再較正）を先に書く。クロス評価の関数と結果の型を実装する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、表・対の集計・3つの診断の記録・通信量・乱数・clientの状態の一致を示し、要求4.2の経路を通ったことを確かめるtestが成功する。拒否と途中の失敗のtest、依存境界のsuiteが成功する。
  - _Depends: 2.1_
  - _Boundary: cross_evaluate_global_models、ModelCrossEvaluation、CrossEvaluationLossSums、ModelPairUniqueCorrectnessCounts_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 5.1, 5.3, 5.4, 6.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptで、新規モデルが登録されたラウンドにクロス評価を行う
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、集約の後にクロス評価を行い、表の形と、通信量と記録が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.2_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 6.1_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 4.1, 6.1, 6.2_

## Implementation Notes

- 調査の結果、旧の「クロス評価→クラスタリング→統合」を、クロス評価（本spec）と、クラスタリングと統合（次のspec）の2つに分けた。境界は、損失の統計の表と、対ごとの正誤の集計（`ModelCrossEvaluation`）。
- 手順からの逸脱: clientの評価（2.1）とサーバのクロス評価（2.2）は、sourceを先に書き、その直後に対照testを書いた（testを先に書く手順からの逸脱）。
- 実装中の発見: (1)実旧のサーバを最終構成の枝（クラス別つきの正誤の比較）へ通すには、旧の設定の`FEDSDA_CLUSTERING_DECISION`を`class_functional_confidence`にする必要があった（既定は`distance`。サーバの生成時に読まれる）。旧の設定の差し替え（`set_legacy_configuration`）へ、最終構成のクラスタリングの設定を足した。既存の対照は、クラスタリングを行わないので、影響を受けない。(2)旧は、上限で標本を抜き出した後で、5件未満なら件数0を返す（上限が4以下だと、常に評価されない）。新も同じ順。(3)多クラスの対照の条件では、評価標本が、評価に使える件数（6件以上）に達しなかった（評価標本は、正式IDのモデルへ、警報の区間の標本を吸収したときだけ足される）。実旧と新を同時に進める下書きのtest（commitしていない）で、多クラスの16条件を調べて確かめた。経路の網羅のtestは、「評価標本を使う」経路だけ、2値の条件で通っていればよいとした。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。乱数の消費と、旧の評価の数値の対応なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `765f84b3-7d0a-444f-a975-b3fa154f4e5b`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`c207322..bab0c60`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし、Minor 1件、任意 5件）。対象test（51 passed）、依存境界・fresh process・束のtest（3179 passed）、Ruff、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない。旧の`FunctionalPairStats`の定義は、testが読む属性名から推定した、と申告している。
- 指摘の採否（Minorと任意だけなので、再レビューは受けていない）: (1)Minor: 旧は、対象が現行モデルで評価標本が足りないとき、学習データの辞書を既定値つきで読み、空の列を作る。新は作らない→新は変えない（読取りで状態を変えない）。実旧で再現するtestを足し（`test_cross_evaluation_of_current_model_does_not_create_empty_training_collection`。生成直後のclient）、[LEGACY-017](../../../docs/research/implementation-findings/legacy-017-cross-evaluation-creates-empty-training-collection.md)として記録し、設計の「旧と違う点」へ、起きる条件と影響を書いた。ラウンドの対照（毎ラウンドのクロス評価、8条件×15ラウンド）では、この状態は起きていない。(2)任意: 要求4.2の「新規モデルが登録されたラウンドごと」を、対照の実際（毎ラウンド）に合わせた。(3)任意: パラメータの非有限値は、乱数を消費した後で拒否される→設計へ書いた（ownerは変わらない。グローバルモデルのownerには非有限が入らない）。(4)任意: 要求5.2の対象のモデルIDは、型だけを確かめる（保有していないIDは、件数0の結果）→要求の文面を直した。(5)任意: サーバと全clientへ同じ乱数生成器を渡す必要がある→設計と再開案内へ書いた。(6)任意: 途中の失敗のtestが、通信量を固定値の代役に差し替えて照合している→グローバルモデルとclientの状態を直接比べる形へ書き直した。
- レビューの後の変更: 上の(1)(6)のtestのほか、命名の照合（`spec_checks.py names`）が、集計の補助class（非公開）の2つのメソッドを公開の名前として挙げたので、補助classを非公開の関数にした（挙動は変えていない。サーバの対照8条件で確かめた）。
- 検証（Windows基準環境、`ee55c17`。この後は、sourceとtestを変えていない）: 全pytest 11043 passed / 3 skipped / 2 warnings、exit 0（前spec 10991＋診断の記録 31＋クロス評価 21）。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtest 2件はskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、変更した全sourceで「未登録: なし」。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: クロス評価は、まだラウンドの中から呼ばれていない（共用scriptの代役と対照testが呼ぶ。クラスタリングと統合のspecで、新規モデルがあるラウンドの、集約と配布の間へつなぐ）。対照は、毎ラウンド、クロス評価を行う（旧の実際のラウンドでは、新規モデルがあり、モデルが2つ以上のラウンドだけ）。clientの上限と、評価の標本の上限の、設定の置き場所は仮（引数と、束のスカラーの設定）。旧の計算量・所要時間の記録、Cached方式、非劣性の検証のための損失差は、移植していない。診断の記録の保存は、診断の保存を決めるspecで扱う。WSLでは実行していない。
