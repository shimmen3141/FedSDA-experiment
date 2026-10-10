# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. サーバの操作
- [x] 1.1 サーバのownerの記録、組立て、3つの操作を実装する
  - サーバの操作のtest（軽量メッセージ、同期の関数の呼出しと結果の保持、終端で不変、組立てと操作の拒否）を先に書く。依存の許可集合を登録する。
  - 完了: サーバのtestと、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunServer、FedsdaRunServerOwners、assemble_fedsda_run_server_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.2, 6.2_

- [x] 2. 参加者の初期準備と全体run
- [x] 2.1 設定の束とfactoryを、実旧の全体runとの対照つきで実装する
  - 旧の設定の差し替えへ、全体runの条件を足す。旧の全体の流れを、旧の部品を旧の順に呼んで実行するhelperと、新の全体run（test専用の中継で、真の概念を渡す）との対照testを先に書く。
  - 設定の束、factoryを実装する。依存の許可集合を登録する。
  - 完了: 全体runの対照が、複数のseedと条件で、概念列・観測列・サーバの全状態・診断の記録・各clientの全状態・runの乱数の最終状態の一致を示し、要求4.2の経路を通ったことを確かめるtestが成功する。真の概念なしの全体run、繰返し、束とfactoryの拒否のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: FedsdaRunParticipantSettings、FedsdaRunParticipantFactory_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 5.1, 5.3, 6.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptで、factoryと実行の枠による全体runを実行する
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、全体runを実行し、区間の進行の件数と、サーバとclientの状態の対応を確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.1_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 6.1_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 4.1, 6.1, 6.2_

## Implementation Notes

- 調査の結果、全体runの接続を、「実行の枠への接続」（本spec）と、「指標の導出とgoldenとの照合」（次のspec）の2つに分けた。Linuxでの再現性の確認は、goldenを扱う次のspecで行う。
- 手順からの逸脱: サーバの操作（1.1）と、束・factory（2.1）は、sourceを先に書き、その直後にtestを書いた（testを先に書く手順からの逸脱）。2つのtaskを1つのcommitにした（`aabbe87`）。全体runの対照は、最初の実行で、当初の全条件が通った（REDの実行は記録していない）。
- 実装中の発見: (1)新の全体run（`execute_stream_protocol_run`＋`FedsdaRunParticipantFactory`）は、実旧の全体run（旧の全体の流れを、旧の部品を旧の順に呼んで実行したもの）と、9条件で、概念列・観測列・サーバの全状態・2つの診断の記録・各clientの全状態・runのPythonとNumPyの乱数の最終状態が一致した。部品のspecで見つからなかった差は、出なかった。(2)統合が起きる条件と、終端で未完了の候補検証が回収される条件は、実旧だけの全体runを、それぞれ64条件・76条件進めて選んだ（下書きのtest。commitしていない）。(3)`FedsdaRunClient`は、実行の枠の契約の型（`RunClientOperations`）と、戻り値の型と任意の引数が違うので、factoryで`cast`している（値は変更しない）。
- 真の概念の扱い（主担当の判断。ユーザーへ確認する事項）: 実行の枠の契約（clientへ真の概念を渡さない。single-run-executionの要求で、testがある）は、変えていない。このため、新の全体runでは、真の概念に依存する診断（真の概念別の診断証拠、割当概念の計数、標本ごとの記録の概念、クラスタリングの真の概念の一致）は、行われない。対照testは、test専用の中継（`ConceptInjectingClientOperations`）で真の概念を渡して、診断まで実旧と照合している。真の概念を渡さない全体runが、これらの診断だけが違うことも確かめている。
- 独立レビュー（2回、2026-10-10）: どちらもClaude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。全体の流れと乱数の消費の、旧との対応なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。
  - 1回め: session `8d59f46c-ae22-4363-a1ed-2ccfcb11e102`、対象`7094fa1..58e9463`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 1件、任意 4件）。対象test（18 passed）、依存境界とfresh process（3044 passed）、Ruff、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない。
  - 2回め（Majorの修正の確認。別session）: session `cb3b7eac-24ae-4858-ba00-5cc60e03c2db`、対象`58e9463..ec46cd8`。判定は`FIX: APPROVED`（指摘なし）。対象test（14 passed、skipなし）とRuffを独立に実行し、足した2つの経路の判定が、新の終端の回収の記録と、旧の`finalize_incomplete_forward_validation`に対応することを、コードを読んで確かめた。
- 指摘の採否: (1)Major: 要求4.2の経路のうち、「候補の採用」と「終端での未完了の候補検証の回収」が、経路の網羅のtestの必須集合になかった→採用。対照testが、新の適応記録の結果種別を、通った経路へ足すようにし、必須集合へ2つを足した。候補の採用は、もとの8条件で通っていた。終端での回収は通っていなかったので、通る条件（seed 17、標本250件）を足して9条件にした。新の経路の有無が、実旧の判定（理由`insufficient_forward_data`）の有無と、条件ごとに一致することも確かめる。(2)任意: 乱数生成器についてのコメントが、assertと合っていない→コメントを消した（サーバとclientが同じ乱数生成器を使うことは、runの乱数の最終状態の、実旧との一致で確かめている）。(3)任意: クラスタリングを有効にする条件の固定（`on_new_model`）が、設計にない→設計の「旧と違う点」へ書いた。(4)任意: 束の再検査の例外の報告が2通り→設計へ規則を書いた（sourceは変えない）。(5)任意: 真の概念に依存する項目の名前が、読取りの実際の項目名か、確かめられなかった→名前が実際の項目名であることと、少なくとも3つの項目が実際に違っていることを確かめるassertを足した。
- 検証（Windows基準環境、`ec46cd8`。この後は、sourceとtestを変えていない）: 全pytest 11299 passed / 3 skipped / 2 warnings、exit 0（前spec 11280＋全体runの対照ほか 14＋サーバの操作 5）。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtestはskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新しい2つのsourceで「未登録: なし」。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 指標（精度、検出の適合率・再現率、通信量の集計ほか）の導出と保存、goldenとの照合は、次のspec。旧の全体runのうち、照合していないのは、ラウンドごとの計測（`telemetry`）と、指標の計算より後の部分。設定の束（`FedsdaRunParticipantSettings`）は、runtimeに置く仮の形で、完全なrun設定（保存表現、preset）からの組立ては、まだない。SINE以外のdatasetは扱わない。WSLでは実行していない。
