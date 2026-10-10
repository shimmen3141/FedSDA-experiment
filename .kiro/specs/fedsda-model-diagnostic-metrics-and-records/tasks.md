# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 計算
- [x] 1.1 値のまとめ、同じ概念のAUC、クラスタの割当と真の概念の一致の比較
  - 乱数の列での、旧と同じ式・実旧の関数との照合と、拒否のtestを先に書く。
  - _Boundary: model_diagnostic_metric_calculations_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. 指標
- [x] 2.1 モデル別の診断の指標の導出と、指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との、全部の項目の照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_model_diagnostic_metric_derivation、FedsdaRunMetrics_
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. 記録の表
- [x] 3.1 登録・クラスタリングの列と、clientの保有モデルごとの表
  - goldenの3ケースと、小さい条件の、実旧が保存した配列との照合を先に書く。
  - _Depends: 2.1_
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 3.1, 3.2_

- [x] 4. 記録と検証
- [x] 4.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [x] 4.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.1, 3.1, 4.1_

## Implementation Notes

### 実装（task 1.1〜4.1）

- 手順: 計算（1.1）は、testを先に書いた（sourceがない状態での実行は、していない）。指標の導出（2.1）は、照合のtestの編集と、sourceを、同じコマンドで入れた（testが先に失敗することは、確かめていない——逸脱）。表（3.1）は、testの期待を先に直し、失敗を確かめてから、sourceを直した。
- AUCを定義できないときの値を、NaNからNoneへ変えた（設計の「旧と違う点」）。きっかけ: 最初の実装（NaN）で、同じ結果を2回導出して比べる既存のtestが、失敗した（NaNは、自分自身と等しくない）。計算の関数の戻り値は`float | None`。
- サーバの同期のラウンドごとのモデル数の数え方を、表のmoduleから、導出のmoduleの`count_models_after_server_round_synchronizations`へ移した（値は同じ。前のspecの照合が、そのまま成功）。
- 保有モデルごとの学習量は、表と指標が、同じ関数（`collect_client_held_model_training_amounts`）から作る。指標は、clientの保有の順（旧の`client.models`の順）で、モデルごとに足す。表は、clientの中を、モデルIDの昇順に並べる（旧の配列の順）。
- 依存境界: 新しい2つのmoduleを、moduleごとの許可集合へ登録し、importが変わった2つのmoduleの許可集合を、実際のimportへ合わせた。計算のmoduleは、NumPy（`asarray`・`float64`）を使う（旧と同じ式で、平均・標準偏差を計算するため）。

### 照合の結果（Windowsの基準環境）

- 計算: 乱数で作った列（値のまとめ30条件、AUC 40条件、クラスタの割当の比較30条件）で、旧と同じ式・実旧の`_lower_score_auc`と一致。
- 指標: 26項目が、goldenの3ケースの実旧の実行結果と一致。モデル数のまとめ以外の23項目は、小さい条件（7条件×設定2通り）の、実旧の`_add_model_diagnostic_results`の値とも一致（すべて`==`。旧のNaNは、新のNone）。
- 対応表の取り違えの検査: どの条件でも同じ値のままの指標は、ない。取り違えを検出できない対は、クラスタの割当の適合率・再現率・F1の3つの間だけ（実行した条件では、いつも同じ値。この3つは、計算のmoduleが、4通りの数から作り、乱数の列で、旧と同じ式と照合してある）。
- 表: 足した17の列が、goldenの3ケースと、統合を通る小さい条件（3条件）の、実旧が保存した配列と、形と値で一致。照合が通った場合: 吸収されたモデル、最も近い相手のないモデル、判定の値が距離と違う対、未登録のモデルの保有、3つ以上のモデルのクラスタ、真の概念の一致・不一致。

### 独立レビューと検証

- 独立レビュー（1回目、2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `ba4cb5ef-4e47-4508-8d47-8ad515ada7cf`、対象`e306725..ccd6959`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 1件、任意 1件）。レビュー担当は、計算のtest（121 passed）、表のtestのうちgoldenと実旧を使わないもの（21 passed・1 skipped——全条件を実行したときだけ確かめるtest——）、依存境界、Ruffを独立に実行し、計算・導出・表を、旧のsource（`experiment.py`、`servers/clustering.py`、`model_lineage.py`、`clients/fedsda.py`）と照合した。実旧との照合のtest、全pytest、Pyright、WSLは実行していない（依頼文で、主担当が実行すると伝えた）。レビュー担当の逸脱: 読取りに、許可していないshellのコマンド（`grep`・`sed`・`cat`）を使った（ファイルは変えていない。レビューの後、作業ツリーは空だった）。
- 指摘の採否: (1)Major: AUCの計算が、NaNの入力で`ValueError`を出す。要求1.2と旧（`isfinite`でない値を除く）は「数えない」→採用。NaNを拒否せず、有限でない値（無限大、NaN）として数えない。testは、NaNでの`ValueError`の期待を削り、既知の値のtestと、実旧との乱数の照合の値の候補へ、NaNと負の無限大を足した（testを先に直し、22件の失敗を確かめてから、sourceを直した）。実行の結果は、変わらない（記録の距離は、NaNを持てない）。(2)任意: 保有モデルの順が、旧の`client.models`の順と同じ、という設計の記述を、直接確かめていない→採用。小さい条件のtestで、clientごとに、実旧の順と比べる（14の条件の全部で、同じ）。設計の記述を、それに合わせた。
- 修正の確認（別session、2026-10-11）: Claude Haiku 5.5。session `f38968ff-27c4-40b7-8ddb-60072e3a2edf`、対象`ccd6959..04f5378`。判定は`FIX CONFIRMATION: APPROVED`（指摘なし。任意の注意: 保有モデルの順の比較は、実行の終わりの状態だけ——指標は、実行の終わりに計算するので、主張の範囲には足りる）。計算のtest（119 passed）とRuffを独立に実行した。
- WSLで成功（`ccd6959`、WSL Ubuntu）: このspecの3つのtestファイルが、202 passed・skipなし（goldenの3ケースと小さい条件の、指標と列が、WSLでの実旧の実行とも一致）。指摘の反映の後は、WSLでは実行していない（変えたのは、AUCの計算のNaNの扱いと、testの補強）。
- 検証（Windows基準環境）: レビューの前の`ccd6959`で、全pytest（並列、`-n 8 --dist loadfile`）12225 passed / 4 skipped / 3 warnings、exit 0。指摘の反映の後の`04f5378`で、12223 passed / 4 skipped / 3 warnings、exit 0（NaNでの`ValueError`の2件を削った）。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。skipの4件は、POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新規・変更のsourceで「未登録: なし」。共用scriptは`FRESH PROCESS SMOKE PASSED`（91列）。この後は、specの文書とsteeringだけを変えた。
- 変異テストは実行していない（既定）。

### 未検証・残る制約

- 真の概念の一致が分からない対（−1）は、実旧との照合では、通っていない（どの条件にも、現れない）。表での表し方は、記録を足したtestで確かめた。
- モデル数のまとめ（3項目）の、実旧との照合は、goldenの3ケースだけ（小さい条件の実旧の実行は、ラウンドごとのモデル数の、旧の集計を通らない。ラウンドごとの値そのものは、前のspecで、小さい条件でも照合してある）。
- 「記録から導く（未照合）」の残りは、指標45・配列47（移行の一覧の「残り」）。
