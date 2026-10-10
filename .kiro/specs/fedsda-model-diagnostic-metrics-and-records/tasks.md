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

- [ ] 4. 記録と検証
- [x] 4.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [ ] 4.2 独立レビューの指摘を反映し、全回帰を通す
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

### 未検証・残る制約

- 真の概念の一致が分からない対（−1）は、実旧との照合では、通っていない（どの条件にも、現れない）。表での表し方は、記録を足したtestで確かめた。
- モデル数のまとめ（3項目）の、実旧との照合は、goldenの3ケースだけ（小さい条件の実旧の実行は、ラウンドごとのモデル数の、旧の集計を通らない。ラウンドごとの値そのものは、前のspecで、小さい条件でも照合してある）。
- 「記録から導く（未照合）」の残りは、指標45・配列47（移行の一覧の「残り」）。
