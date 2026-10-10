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

- [ ] 3. 記録と検証
- [x] 3.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [ ] 3.2 独立レビューの指摘を反映し、全回帰を通す
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

### 未検証・残る制約

- 「記録から導く（未照合）」の残りは、指標40・配列19（予測の集計。移行の一覧の「残り」）。
