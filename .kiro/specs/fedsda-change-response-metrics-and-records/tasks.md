# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 保存する表と列の名前
- [x] 1.1 名前を直す（表`server_round`、`held_model_count_at_prediction`、`first_new_concept_sample_index`）
  - testの期待する名前を先に直し、失敗を確かめてから、sourceを直す。
  - 完了: 前のspecのtestが、新しい名前で成功する。
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. 指標
- [x] 2.1 対応づけ・分類・変化点の誤差の計算
  - 乱数の列での、実旧の関数との照合と、拒否のtestを先に書く。
  - _Boundary: change_response_metric_calculations_
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2.2 応答の指標の導出と、指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との、全部の項目の照合を先に書く。
  - _Depends: 2.1_
  - _Boundary: fedsda_change_response_metric_derivation、FedsdaRunMetrics_
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. 記録の表
- [x] 3.1 警報の表、監視の値の表、適応と判定の列
  - goldenの3ケースの、実旧が保存した配列との照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 4.1, 4.2_

- [x] 4. 記録と検証
- [x] 4.1 移行の一覧を更新する
  - _Requirements: 5.1_

- [x] 4.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.2, 3.1, 4.1_

## Implementation Notes

### 実装（task 1.1〜4.1）

- 名前の見直し（task 1.1。ユーザーの指摘: 表`round`は簡潔すぎる。短さより、分かりやすさと、実態との一致を優先する）: 主担当が見直して、改善の余地があると判断したものを直した——表`round`と、ラウンドごとの計算の2つの表を、1つの表`server_round`へ（列は、`global_model_count_after_synchronization`・`client_held_model_count_after_synchronization`・`local_processing_*`・`synchronization_*`）。`sample_prediction.held_model_count`→`held_model_count_at_prediction`。`concept_change.sample_index`→`first_new_concept_sample_index`。ほかの表と列は、記録の型とfieldの名前のままで十分と判断した。保存の形式を使う既存のfileがないので、形式の版は1のまま。前のspec（fedsda-experiment-run-artifacts）の文書は、書き換えていない（設計の表の名前は、このspecのdesign.mdの「名前の見直し」が正）。
- 手順: 名前の見直しは、testの期待を先に直すつもりだったが、置換のscriptの失敗で、sourceの置換が先に入った（古い名前のtestが、新しいsourceに対して失敗することを確かめてから、testを直した）。計算（2.1）は、testを先に書いた（sourceがない状態での実行はしていない）。指標の導出（2.2）と表の列（3.1）は、sourceを先に書き、その後で、実旧との照合のtestを書いた（逸脱）。
- 適応の粗い区分の対応（`ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME`）を、表のmoduleから、指標の導出のmoduleへ移した（指標と表の両方が使う）。
- 旧の指標のうち、ほかの値から計算できるもの（真の変更の数・対応した数・対応しなかった数・見逃し率・誤検出率、「参照の再適合」の合計）は、新の指標に持たない。testの対応が、計算して照合する。
- 依存境界: 新しい2つのmoduleを、moduleごとの許可集合へ登録し、importが変わった2つのmoduleの許可集合を、実際のimportへ合わせた。

### 照合の結果（Windowsの基準環境）

- 計算: 乱数で作った列（72条件と、60条件）で、対応づけ・対応づかなかったイベントの分類・変化点の推定の誤差が、実旧の`match_events`・`_classify_unmatched_events`・`_change_point_errors`と一致。乱数の条件は、4つの区分を、すべて通る。
- 指標: 旧の`compute_metrics`が返す指標のうち、照合済みの33指標に入らない52項目と、監視の値の最大が、goldenの3ケースの実旧の実行結果と、小さい条件（7条件×設定2通り）の実旧の`compute_metrics`の戻り値と、`==`で一致。旧の`compute_metrics`の戻り値の、全部のキーが、照合済み・応答の指標・最終構成では使わない指標（5項目。値は0）のどれかに入ることを、testで確かめた。
- 表: 足した列（18の旧の配列に当たる）が、goldenの3ケースの、実旧が保存した配列と、形と値で一致（欠損は、新旧とも、整数は−1、損失はNaN）。
- 名前の見直しの後、前のspecの照合（31の配列、ラウンドごとのモデル数、保存と読込み）が、新しい名前で成功。

### 独立レビューと検証

- 独立レビュー（2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。対象`b21458b..6440083`。起動は2回になった——1回目（session `597f9d7b-8e61-4512-9bd2-44d975d42257`）は、レビュー担当が検証コマンドを裏で実行し、保存された出力が、最後の通知への返答（判定`CHANGES_REQUESTED`、Blocker・Majorなし、Minor 3件・任意2件、とだけ書いたもの）だけになり、指摘の本文が残らなかった（sessionは保存していないので、復元できない）。2回目（session `7ed93530-421a-4af7-a98d-bfd7eb5caccf`。同じ依頼文に、裏で実行しないこと、最後のメッセージに全文を書くことを足し、全部のメッセージが残る出力形式にした）の判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし。Minor 2件、任意5件）。採否は、2回目の指摘で決めた（1回目の指摘の本文は、読めていない）。
- レビュー担当（2回目）が独立に実行したもの: 計算のtest（157 passed）、表のtestのうちgoldenと実旧を使わないもの（20 passed）、依存境界（3043 passed）、Ruff。実旧との照合のtest（約8分の2つ）、全pytest、Pyright、WSLは実行していない（依頼文で、主担当が実行すると伝えた）。計算と指標の導出を、旧のsource（`metrics.py`、`provisional_model.py`、`experiment.py`、`clients/fedsda.py`）と照合した。レビュー担当の逸脱: 読取りに、許可していないshellのコマンド（`grep`・`sed`・`cat`）を使った（ファイルは変えていない。レビューの後、作業ツリーは空だった）。
- 指摘の採否: (1)Minor: 小さい条件で、監視の値の最大を照合していない→採用（実旧の`_add_telemetry_results`の値と`==`で比べる）。(2)Minor: 値が0のままの指標は、対応表の取り違えを検出できない疑い→採用。照合のたびに旧の値を積み、最後のtestが、実行した全部の条件（goldenの3ケースと、小さい条件14）で、旧の値が同じになる指標の対と、0のままの指標を数える。実測: 0のままの指標はない。取り違えを検出できない対は2つだけ——`provisional_interval_count_mean`と`provisional_training_count_mean`（最終構成では同じ値。新は1つの値で持つ）、`provisional_resolution_delay_mean`と`provisional_validation_count_mean`（最終構成の検証は、提案の次の標本から1標本ずつ進むので、同じ値になる。もとの2つの列は、表のtestで、別々に、旧の配列と照合している）。(3)任意: 理由と採否が、境界の丸めで食い違いうる（旧と同じ）→設計へ注記。(4)任意: 長さの違う列で、旧は飛ばし、新は例外→設計の「旧と違う点」へ記載。(5)任意: 名前の基準を、残りの名前にも当てる→採用。`training_model_switch.sample_index`→`switch_sample_index`、クラスタリングの2つの表の`round_index`→`server_round_index`、`model_registration.registered_round_index`→`registered_server_round_index`（testの期待を先に直し、失敗を確かめてから、sourceを直した）。`resolution_sample_index`は、名前はそのままで、未完了の行での意味を設計へ書いた。(6)任意: 件数の書き方→直した（値を照合した配列は18。ほかに、`provisional_validation_sources`が常に`forward`であることを確かめた。移行の一覧へ、内訳を書いた）。(7)任意: 適応の粗い区分の対応の定数を、`evaluation/`へ置く→見送り（粗い区分は、指標と表のための、旧の`action`に当たるまとめ方で、使うのは`runtime/`の2つのmoduleだけ）。testのimportは、定義しているmoduleからに直した。
- Blocker・Majorがないので、確認のレビューは行っていない（手順どおり）。
- WSLで成功（`6440083`、WSL Ubuntu）: このspecの3つのtestファイルが、231 passed・skipなし（goldenの3ケースの応答の指標と列が、WSLでの実旧の実行とも一致）。指摘の反映の後は、WSLでは実行していない（変えたのは、列の名前と、testの補強）。
- 検証（Windows基準環境）: レビューの前の`6440083`で、全pytest（並列、`-n 8 --dist loadfile`）12097 passed / 4 skipped / 3 warnings、exit 0。指摘の反映の後の`ded63bf`で、全pytest 12098 passed / 4 skipped / 3 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。skipの4件は、POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新規・変更のsourceで「未登録: なし」。この後は、specの文書とsteeringだけを変えた。
- 変異テストは実行していない（既定）。

### 未検証・残る制約

- 「記録から導く（未照合）」の残りは、指標71・配列64（移行の一覧の「残り」）。
- 旧の`provisional_validation_sources`（最終構成では、常に`forward`）と、`adaptation_episode_ids`（常に−1）に当たる列は、持たない。
