# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 既存のownerと束への追加
- [x] 1.1 ownerの操作を足す
  - 保有モデルの全置換え、損失統計の全置換え、グローバルの損失統計の一覧、サーバによる統合の適応記録（結果種別と位置の列）、共有部のoptimizerの状態の保持者を足す。新しいmoduleの依存の許可集合を登録する。
  - 完了: 足した操作の単独のtest（順、検査、不正で不変）と、変更したmoduleの既存のtest、依存境界のsuiteが成功する。
  - _Boundary: HeldModelTrainingStateRegistry、ModelAndClassLossStatisticsStore、GlobalModelRepository、AdaptationRecordStore、SharedParameterOptimizerStateHolder_
  - _Requirements: 2.1, 2.4, 3.1, 3.3, 6.2_
- [x] 1.2 clientが、共有部のoptimizerの状態を保持者で持ち、作り直すモデルのoptimizerの設定を束で受け取る
  - ownerの記録の共有部のoptimizerの状態を保持者にし、共有部と構造の参照を、現在の学習帰属のモデルから読む。束へ、作り直すモデルのoptimizerの設定を足す（既存の設定と同じ種類であることを確かめる）。束を作る箇所と、共有部のoptimizerの状態を読む箇所の、既存のtestと共用scriptを直す。
  - 完了: clientと束の既存のtest（実旧clientとの対照を含む）、事前学習・登録と集約のtest、共用script、依存境界のsuiteが、変更の後も成功する。束の新しいfieldのtestが成功する。
  - _Boundary: FedsdaRunClient、FedsdaRunClientOwners、FedsdaRunClientSettings_
  - _Requirements: 3.3, 3.4_

- [x] 2. 受取りと配布
- [x] 2.1 clientの受取りを、実旧のclientの受取りとの対照つきで実装する
  - 登録と集約の対照のoracleの状態から、実旧のclientの`apply_server_mapping`と新の受取りへ、同じID対応とグローバルモデルを与えて照合する対照testを先に書く。
  - 入力の検査、統計の選択、分類器とoptimizerの状態の生成、適応記録、統計・評価標本・学習データ・計数の付け替え、保有モデルの置換えとつなぎ直し、現在の学習帰属の付け替え、clientの操作を実装する。依存の許可集合を登録する。
  - 完了: ID対応の受取りの対照が、現在の学習帰属が付け替わる場合・複数のモデルが1つへ集まる場合・評価標本が上限を超える場合で、全状態と乱数の一致を示す。拒否のtest、受取りが変えない状態のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2_
  - _Boundary: apply_global_model_distribution、GlobalModelDistributionApplication、FedsdaRunClient.apply_global_model_distribution_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.2, 5.1, 5.2, 5.3, 5.5, 6.2_
- [x] 2.2 サーバの配布を実装し、登録→集約→配布のラウンドを実旧と照合する
  - 下りの通信量の記録と、全clientへの受渡しを実装する。ラウンドの対照（標本処理→保留中の学習→登録→集約→配布→送信待ちの進行）を追加する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、2値・多クラス、複数の標本列、学習率の2つの設定が違う条件で、ラウンドごとに、配布の後のサーバとclientの全状態と乱数の一致を示し、要求4.3の経路を通ったことを確かめるtestが成功する。配布の拒否と途中の失敗のtest、依存境界のsuiteが成功する。
  - _Depends: 2.1_
  - _Boundary: distribute_global_models_to_clients_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.3, 5.1, 5.4, 5.5, 6.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptで、ラウンドごとに登録・集約・配布を行う
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、ラウンドごとに登録・集約・配布を行い、配布の後に、全clientが全グローバルモデルを同じ値で保有し、1つの共有部につながっていることと、下りの通信量が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
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

- 手順からの逸脱: 1.1と1.2は、1つのcommitにした（`1e0d63e`）。受取りは、ID対応つきの対照testをsourceより先に書いたが、拒否と「変えない状態」のtestは、sourceの後に書いた。
- 実装中に設計を直した点: (1)共有部のoptimizerの状態は、作り直す全モデルについて作る（つなぎ直しの部品が、各モデルの共有部とoptimizerの対応を検査するため。使うのは、つなぎ先のものだけ）。(2)つなぎ直しを、適応記録と統計の更新より前へ移した（つなぎ直しの検査で拒否されたときに、ownerが変わらないようにする。成功時の結果は同じ。`53765a5`）。(3)集約の非公開の関数だった、パラメータの分割を、公開の名前`split_shared_and_concept_specific_parameters`にして、配布からも使う（命名表へ足した）。
- 対照の条件: 実旧と新を同時に進める下書きのtest（commitしていない）で、16条件×14ラウンドの状態を調べて、条件を選んだ。評価標本の抜出しは、既定の上限（12件）では起きなかったので、上限を4件にした条件を足した。学習率の2つの設定を違う値にすると、標本列によっては新規モデルが作られなくなるので、複数のグローバルモデルができる条件を選んだ。
- 既存のtestの変更: clientの統計の照合（`assert_run_client_matches_legacy`）は、クラス別の統計を持たない旧の統計（サーバの集約で作られ、配布でそのまま置かれるもの）を受け入れる。束の「別の設定型」の拒否のtestは、2つ先のfieldの値を使う（optimizerの設定の2つが同じ型のため）。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。NNの作り直しと乱数の消費、複数のownerの更新順なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `7ffcacf1-d9f6-4135-bfd3-310ba1dd4455`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`bcef33c..53765a5`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし、Minor 4件、任意 1件）。対象test（48 passed）、依存境界・fresh process・束のtest（3179 passed）、Ruff、共用scriptを独立に実行した。全pytest・Pyright・`spec_checks.py`は実行していない。共用scriptの差分の全行と、旧のモデル生成と新の分類器の構成の細部は読み切っていない（乱数の消費の一致は、対照testの結果に依拠）と申告している。
- 指摘の採否（Minorだけなので、再レビューは受けていない）: (1)Minor: clientの受取りの検査でしか分からない不正（現在の学習帰属が、配布にないIDへ移るID対応）では、最初のclientが拒否しても、下りの通信量は残る→容認する（サーバは、clientの状態を事前に検査しない。旧と同じ順）。設計へ書き、testで固定した（`test_distribution_keeps_communication_volume_when_first_client_rejects`。通信量のほかは、サーバもclientも乱数も変わらない）。(2)Minor: サーバの拒否の条件が型だけ→(1)のtestを足した。拒否のloopのtestは、例外の型と、状態の不変を確かめる（どの検査で止まったかは、条件ごとには確かめていない）。(3)Minor: 一時IDのモデルの概念固有部のoptimizerは、その状態自身の設定で作り直される→記録だけ。レビュー担当は「学習率が違う条件でも独立には確かめていない」としたが、学習率の2つの設定が違う条件（クラス数4・区間長30・seed 3）のラウンドの対照が、一時IDのモデルを持つclientへの配布を通り、実旧の`attach_backbone`の後のoptimizer（学習率を含む`state_dict`）と照合している。設計へ1行足した。(4)Minor: 経路の網羅の判定の2つ（学習データを持たないモデルが混ざる集約、採番だけが行われた一時IDのモデル）は近似→記録だけ（前者は、配布の後の学習データの件数で、次のラウンドの集約の状態を判定する。後者は、送信保留がなく一時IDのモデルを保有することで判定する）。(5)任意: 公開の定数`SERVER_REMAP_ADAPTATION_OUTCOME`が命名表にない、位置の列と結果種別の語がそろっていない→不採用（定数とfieldは、命名表の対象外。語は、結果種別が「統合で付け替わった」、位置の列が「サーバが付け替えた位置」で、どちらもremapを含む）。
- 検証（Windows基準環境、`30da913`。この後は、sourceとtestを変えていない）: 全pytest 10948 passed / 3 skipped / 2 warnings、exit 0（前spec 10894＋ownerの操作 17＋束と結果種別の既存testの増分 5＋配布 32）。JUnitは10951 testcase、failure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtest 2件はskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、変更した全sourceで「未登録: なし」。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 集約後の予測重みと診断証拠の再較正は未移植なので、対照は、旧のサーバの`run_round`ではなく、登録・集約・配布を直接呼ぶ形で照合している（旧の最終構成の実際のラウンドとは、再較正の分だけ違う）。ID対応を作る処理（クロス評価・クラスタリング・統合）は未移植で、ID対応つきの対照は、testが作ったID対応を、実旧と新へ同じく与えている。サーバの配布は、ID対応で集められる側のグローバルモデルを外さない（外すのは統合の処理）。配布されるパラメータの有限性は検査しない（集約が非有限を拒否するので、グローバルモデルのownerには入らない）。段5より後の失敗（段1の検査を通った入力では起きない想定）では、済んだ段が残る。実行の枠のサーバの操作の実体は、まだない。WSLでは実行していない。
