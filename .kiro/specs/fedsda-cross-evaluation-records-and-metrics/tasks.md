# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 指標
- [x] 1.1 モデル対の予測の相補性の計算
  - 乱数の記録の列での、実旧の`pair_diagnostic_summary`との照合と、拒否のtestを先に書く。
  - _Boundary: cross_evaluation_metric_calculations_
  - _Requirements: 1.1, 1.2_

- [x] 1.2 指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: FedsdaRunMetrics_
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2. 記録の表
- [x] 2.1 clientの評価ごとの表と、クラスごとの表
  - goldenの3ケースと、小さい条件の、実旧が保存した配列との照合を先に書く。
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 3.1, 3.2_

- [x] 3. 記録と検証
- [x] 3.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [x] 3.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 1.2, 2.1, 3.1_

## Implementation Notes

### 実装（task 1.1〜3.1）

- 手順: 計算（1.1）は、testを先に書いた（sourceがない状態での実行は、していない）。指標への追加（1.2）と表（2.1）は、testの期待を先に直し、表の形のtestが失敗することを確かめてから、sourceを直した（指標の照合のtestが、先に失敗することは、個別には確かめていない）。
- 指標は、`evaluation/`の関数が、サーバの記録の列から直接作る（`runtime/`の導出のmoduleは、足していない）。
- 依存境界: 新しいmoduleを、moduleごとの許可集合へ登録し、importが変わった2つのmoduleの許可集合を、実際のimportへ合わせた。

### 照合の結果（Windowsの基準環境）

- 計算: 乱数で作った記録の列（40条件。正誤を比べなかった評価、標本数が0の評価、正誤を比べた評価の全部が標本数0の場合を含む）で、実旧の`pair_diagnostic_summary`と一致。
- 指標: 5項目が、goldenの3ケースの実旧の実行結果と、小さい条件（7条件×設定2通り）の実旧のサーバの集計と、`==`で一致。対応表の取り違えの検査: 取り違えを検出できない対は、ない。どの条件でも0のままの指標は、ない。
- 表: 旧の28の配列（`cross_evaluation_*` 11、`cross_evaluation_class_*` 10、`model_pair_*` 7）が、goldenの3ケースと、統合を通る小さい条件（3条件）で、形・型・値で一致。旧の配列のうち、この3つのまとまりの名前を持つものを、全部比べたことも、testで確かめた。照合が通った場合: 正誤を比べた評価、比べなかった評価（候補と対象が同じモデル、と、違うモデルで標本が足りない評価の両方）、クラスごとの行、標本数が0の評価。通っていない場合は、ない。

### 独立レビューと検証

- 独立レビュー（1回、2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `b279d591-9aab-4e15-a2a5-65f209f3da85`、対象`047d803..59f613c`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし。Minor 1件、任意 3件）。レビュー担当は、計算のtest（42 passed）、表のtestのうちgoldenと実旧を使わないもの（21 passed・2 skipped——全条件を実行したときだけ確かめるtest——）、依存境界、Ruffを独立に実行し、計算と表を、旧のsource（`servers/clustering.py`の`pair_diagnostic_summary`と、クロス評価の記録の箇所、`clients/base.py`）と照合した。実旧との照合のtest、全pytest、Pyright、WSLは実行していない（依頼文で、主担当が実行すると伝えた）。レビュー担当の逸脱: 読取りに、許可していないshellのコマンド（`grep`・`sed`）を使った（ファイルは変えていない。レビューの後、作業ツリーは空だった）。
- 指摘の採否: (1)Minor（文書）: 移行の一覧の、モデル対の指標の行に、表の見出しの行がなく、表として描かれない。置いた場所も、別のまとまりの段落の下だった→採用（見出しを足して、独立した段落にした）。(2)任意: クラスごとの表の`evaluated_sample_count`が、評価の全体の標本数と同じ名前で、意味が違う→採用。`class_evaluated_sample_count`にした（testの期待を先に直し、失敗を確かめてから、sourceを直した）。(3)任意: 新実装は、正誤とクラスごとの正誤を、いつも集める。旧は、設定による→設計の「旧と違う点」へ記載（表が旧と一致するのは、最終構成の実行）。(4)任意: 計算の関数が、記録の`__post_init__`を直接呼ぶ→見送り（手で壊した値を、使う前に検査し直す、既存の書き方——指標の導出の`run_metric_settings.__post_init__()`ほか——と同じ）。
- Blocker・Majorがないので、確認のレビューは行っていない（手順どおり）。
- WSLで成功（`59f613c`、WSL Ubuntu）: このspecの3つのtestファイルが、125 passed・skipなし（goldenの3ケースと小さい条件の、指標と表が、WSLでの実旧の実行とも一致）。指摘の反映の後は、WSLでは実行していない（変えたのは、列の名前1つと、文書）。
- 検証（Windows基準環境）: レビューの前の`59f613c`で、全pytest（並列、`-n 8 --dist loadfile`）12267 passed / 4 skipped / 3 warnings、exit 0。指摘の反映の後の`1179be0`で、12267 passed / 4 skipped / 4 warnings、exit 0（1回目の実行。同じcommitでの2回目の実行は、12267 passed / 4 skipped / 3 warnings。warningは、既知のtorchの警告——nested tensorの2件と、TypedStorageの1件——で、1回目の4件目の内容は、記録していない）。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。skipの4件は、POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新規・変更のsourceで「未登録: なし」。共用scriptは`FRESH PROCESS SMOKE PASSED`（112列）。この後は、specの文書とsteeringだけを変えた。
- 変異テストは実行していない（既定）。

### 未検証・残る制約

- 「記録から導く（未照合）」の残りは、指標40・配列19（予測の集計。移行の一覧の「残り」）。
