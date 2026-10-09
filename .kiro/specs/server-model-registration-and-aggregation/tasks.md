# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. サーバが持つ状態のowner
- [x] 1.1 通信量の記録のownerを実装する
  - 上り・下りの、モデル転送数、メッセージ数、パラメータの値の数、バイト数を数える。依存の許可集合を登録する。
  - 完了: ownerの単独のtest（各操作の加算、値の数とバイト数、各不正入力の拒否と不変）と、依存境界のsuiteが成功する。
  - _Boundary: CommunicationVolumeRecordStore、CommunicationVolumeSnapshot_
  - _Requirements: 2.1, 2.2, 2.3, 7.2_
- [x] 1.2 グローバルモデルのownerを実装する
  - 初期モデル、採番、モデルIDごとのパラメータと統計（写しで受け渡す）、登録の来歴を持つ。依存の許可集合を登録する。
  - 完了: ownerの単独のtest（初期状態、採番、写しの独立、設定した順、来歴の昇順と上書き、各不正入力の拒否と不変）と、依存境界のsuiteが成功する。
  - _Boundary: GlobalModelRepository、GlobalModelRegistrationRecord_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.2_

- [x] 2. 登録と集約
- [x] 2.1 新規モデルの登録を、実旧のサーバとの対照つきで実装する
  - 実旧の事前学習・サーバ・client 3つと、新のclient 3つ・ownerを作り、ラウンドごとに照合する対照testの土台を先に作る。
  - 入力の検査と、client順の採番・来歴の記録・正式IDの確認を実装する。
  - 完了: 対照testで、登録の後の、次の正式ID・来歴・各clientの全状態が実旧と一致する（集約を含む全項目の一致は2.2の完了で確かめる）。拒否のtest、登録の途中の失敗のtestが成功する。
  - _Depends: 1.2_
  - _Boundary: register_ready_client_models、RegisteredClientModel_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.3, 6.4_
- [x] 2.2 clientのモデルの集約を実装する
  - 対象のIDの決定、clientごとの参加モデル、共有部と概念固有部の重み付き平均、統計の平均、通信量、全部の計算の後の反映を実装する。依存の許可集合を登録する。
  - 完了: 対照testが、2値・多クラス、複数の標本列で、ラウンドごとに、グローバルモデルのIDの順・全パラメータ・損失統計、次の正式ID、来歴、通信量の全項目、集約の件数、各clientの全状態、乱数の状態の一致を示し、要求5.2の経路を通ったことを確かめるtestが成功する。拒否のtest、計算の途中の失敗で何も変わらないことのtest、依存境界のsuiteが成功する。
  - _Depends: 1.1, 2.1_
  - _Boundary: aggregate_client_models_into_global_models、ClientModelAggregation_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 6.1, 6.2, 6.4, 7.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptで、ラウンドごとに登録と集約を行う
  - clientの流れの、何もしないサーバの代役を、登録と集約を行う代役に替える（配布は行わない）。
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、複数のclientの標本処理の後で登録と集約をラウンドごとに行い、グローバルモデルが更新されたことと、上りの通信量が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.2_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 7.1_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 5.1, 7.1, 7.2_

## Implementation Notes

- 手順からの逸脱: 1.1・1.2・2.1・2.2は、1つのcommitにした（`22d0ae6`）。sourceの下書きの後にtestを書いた（testを先に書く手順からの逸脱）。
- 対照の条件: 最初の6条件では、採番だけが行われる経路（LEGACY-016）を通らなかったので、実旧だけで条件を探して、通る条件（概念の区間長9・seed 7、区間長10・seed 50）を入れた。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。NNのパラメータの平均と、複数のclientの状態の更新順なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `567b284b-fb91-4052-b42c-5941d0047de0`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`363e07b..4f324ba`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし、Minor 3件、任意 2件）。対象test（64 passed）、依存境界とfresh process（3044 passed）、Ruff、共用script、`spec_checks.py names`を独立に実行した。testファイルの全行と、依存境界testの差分は読んでいないと申告している。
- 指摘の採否（5件とも採用。文書だけの修正なので、再レビューは受けていない）: (1)Minor: 非有限のパラメータを、旧は平均し、新は集約で拒否する→設計の「旧と違う点」へ書いた（写しの部品の契約をそのまま使った結果。発散したrunを続行させる必要が出たら、その時点で扱いを決める）。(2)Minor: 状態の報告を、clientの数で数える前提（最終構成のclientは、すべて状態を報告する）→設計へ書いた。(3)Minor: 要求5.2の「学習データを持たないモデルを含む集約」が、経路の網羅のtestに入っていない→要求と設計を、実態に合わせて直した（生成直後の集約を実旧と照合している。参加するclientがいる集約に混ざる状態は、配布がないと起きないので、配布のspecの対照で確かめる）。(4)任意: 初期モデルの来歴の表し方の違い→設計へ書いた。(5)任意: LEGACY-016の「予測には参加し続ける」は実行で確かめていない→記録を、コードの読みによる、と直した。
- 検証（Windows基準環境、`4f324ba`。レビューの後は、sourceとtestを変えていない）: 全pytest 10894 passed / 3 skipped / 2 warnings、exit 0（前spec 10830＋2つのowner 36＋登録と集約 28）。JUnitは10897 testcase、failure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtestはskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 配布がないので、clientのモデルはラウンドをまたいで発散したまま集約される（対照は、その状態で旧と一致することを確かめている）。参加するclientがいる集約に、学習データを持たない非負のIDのモデルが混ざる状態は、通っていない。非有限のパラメータの集約は、旧と挙動が違う（上の(1)）。実行の枠のサーバの操作の実体は、まだない。WSLでは実行していない。

