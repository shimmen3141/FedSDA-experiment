# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 実行と表
- [x] 1.1 設定から、計測つきの全体runと指標の導出までを行う入口
  - 別々に呼んだ場合との一致、乱数を進めないこと、同じ設定からの同じ結果、拒否のtestを先に書く。
  - _Boundary: fedsda_experiment_run_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 1.2 記録の表
  - goldenの条件のfixture（指標の導出のtest）を使い、表だけから旧の31の配列を作って実旧と照合するtest、ラウンドごとのモデル数の照合（goldenの3ケースと、小さい条件）、表の形・型・行の数の検査を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. 保存
- [x] 2.1 要約と表の保存・読込み
  - 往復、上書きの拒否、書きかけを残さないこと、非有限の値、形式の版のtestを先に書く。保存用の辞書への変換へ、非有限の値の扱いを足す。
  - _Depends: 1.2_
  - _Boundary: fedsda_run_artifact_files、settings_serialization_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 2.2 goldenの3ケースを、設定から実行・保存・読込みして、実旧と照合する。共用scriptへ足す
  - _Depends: 2.1_
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.2_

## Implementation Notes

### 実装（task 1.1〜2.2）

- 手順からの逸脱: 3つのsource（実行の入口、記録の表、保存・読込み）を、testより先に書いた。testは、その後で書き、実旧と照合した。testが見つけた誤りは、下の「testが見つけたこと」。testのfileは、tasksの3つではなく、1つ（`test_fedsda_experiment_run_artifacts.py`）にまとめた（goldenの条件のfixtureを共有するため）。
- 実行の入口は、既存の計測つきの全体runと、指標の導出を、そのまま呼ぶ。全体runの実行と、各ownerは、変えていない。
- 保存用の辞書への変換（`convert_settings_to_plain_mapping`）へ、非有限のfloatを`None`にする指定を足した（指定しなければ、これまでどおり拒否）。
- 適応の表の粗い区分`adaptation_action`は、8つ（旧の`action`の8つの値と、1対1）。全部の結果種別に区分があることを、型の定義（`AdaptationOutcome`）と照らすtestで確かめる。
- 依存境界: 3つのmoduleを、moduleごとの許可集合へ登録した。
- 移行の一覧（docs/research/legacy-run-output-migration.md）へ、このspecで決まった、旧→新の配列の対応と、値の名前の対応を書いた。

### testが見つけたこと

- ラウンドごとのグローバルモデル数の、最初の求め方（初期の1＋登録の累計−統合で吸収された数の累計）は、小さい条件の1つで、実旧と違った（3と2）。登録されても、同じラウンドの集約で、グローバルモデルに残らないモデルがある。同期の配布の記録（clientごとの、受取りの後の保有モデルのID）から数える形へ直した（clientごとの保有モデル数も、同じ記録から。最初の案の「次のラウンドの最初の標本の、処理の前の保有モデル数」は、使わない）。
- 旧の「クラスタリング待ちのモデル数」（`round_pending_clustering_count`）は、最終構成のサーバのclassが、その属性を持たないので、いつも0だった。表に入れず、移行の一覧で「最終構成では使わない項目」へ移した（0であることは、goldenの3ケースと、小さい条件の照合で確かめる）。

### 照合の結果（Windowsの基準環境）

- goldenの3ケース（sine2・sea2・mnist2）を、完全なrun設定（既定値から、縮小した項目だけを置き換えたもの）から実行: 指標が、計測つきの全体runと指標の導出を別々に呼んだ場合と、`==`で一致。呼出し側の乱数は進まない。
- 表だけから作った、旧の保存形式の31の配列が、実旧が保存した配列と、形・型・値で一致（保存して、読み込んだ表でも一致。実行環境のgoldenの記録——形とSHA-256——とも一致）。
- ラウンドごとのモデル数（グローバル、clientごとの保有）が、goldenの3ケースの、実旧が保存した配列と一致。モデルの登録・統合を通る小さい3条件（登録されたモデルがグローバルモデルに残らない場合を含む）でも、実旧の全体runの、同期の後の値と一致。
- 保存と読込み: 往復で、要約の辞書と、配列の名前・形・型・値が同じ。上書きの拒否と、途中で失敗したときに何も残さないことを確かめた。
- 共用script: 完全なrun設定（最終構成の既定値）から、実行・保存・読込みを行う（旧実装とtestのmoduleを読み込まない）。

### 独立レビューと検証

- 独立レビュー（1回、2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `3786d47b-a1bf-4ef4-b449-1858fef9935d`、対象`3cbaf5f..030cdce`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Major・Minorなし。任意 6件）。レビュー担当は、対象のtest（35 passed・skipなし——goldenの3ケースの、実旧と新の実行を含む——）、変換と依存境界のtest（3065 passed）、Ruff、共用scriptを独立に実行した。適応の粗い区分と旧の`action`の対応を、旧のsource（`clients/fedsda.py`・`clients/base.py`）と照合した（8つとも一致）。全pytest・Pyright・`pip check`・`spec_checks.py`・WSLは実行していない（基準どおり）。レビューの後、作業ツリーは空だった。
- 指摘の採否（すべて任意）: (1)非有限の値を`null`にするのは、指標の全部（定常の標本がないときの定常精度を含む）で、コメントは計算量のまとめだけと書いている→コメントを直した。(2)旧の`action`には、最終構成では現れない値（`episode_suppressed`、前向きの検証を使わない方針での`create`・`create_rejected`）がある→移行の一覧へ注記した（扱いは、(A2-変換)で決める）。(3)未照合の残りの配列の数の書き方→直した。(4)読込みが、表のファイルの名前の`".."`、形式の版の`True`・`1.0`を通す→採用（拒否する。testを足した）。(5)testの補強: 表の置換の失敗の位置を足した。小さい条件の「登録されたモデルが、グローバルモデルに残らない」ことを、直接確かめるassertを足した（条件ごとに、最初の案の数え方が、実旧と違う・同じ、を期待として持つ）。`history_routing_soft_active`が定数であること、保有モデル数の合計の照合が同じ記録の2つの読み方であることは、そのまま（前者は、最終構成で常に真。後者は、表の内容の確認）。(6)公開の定数2つを、naming.mdへ足した。Blocker・Majorがないので、確認のレビューは行っていない（反映は、読込みの検査の強化、コメント、test、文書。`e3f2233`）。
- WSLで成功（`030cdce`、WSL Ubuntu）: このspecのtestと、前のspec（完全なrun設定、変換）のtestが、82 passed・skipなし（goldenの3ケースが、WSLのgoldenとも一致）。
- 検証（Windows基準環境、`e3f2233`。この後は、specの文書とsteeringだけを変えた）: 全pytest（並列、`-n 8 --dist loadfile`）11937 passed / 4 skipped / 3 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。skipの4件は、POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件。レビューの前の`030cdce`では、11937 passed / 4 skipped。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新規・変更のsourceで「未登録: なし」。
- 変異テストは実行していない（既定）。

### 未検証・残る制約

- 保存する項目は、照合済みの31の配列に当たるもの、ラウンドごとのモデル数、標本ごとの保有モデル数、ラウンドごとのモデルの計算、指標（`FedsdaRunMetrics`の全部）、完全な条件。「記録から導く（未照合）」の残り（移行の一覧の、指標124・配列85のうち、このspecで扱ったラウンドごとのモデル数の2つを除く、指標124・配列83）は、(A2-2)以降。
- 同じ保存先へ、同時に書く2つのprocessは、扱っていない（保存先の名前を、runごとに一意にするのは、(A3)）。
- 来歴（commit、実行のコマンド）は、要約に入れていない（(A3)）。
- 旧の既定の規模（client 10、標本5000）での実行と保存は、していない。

