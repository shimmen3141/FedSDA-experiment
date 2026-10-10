# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 計算
- [x] 1.1 区間ごとの正解の割合と、クラスの間のまとめ
  - 乱数の列での、実旧の関数・旧と同じ式との照合と、拒否のtestを先に書く。
  - _Boundary: prediction_diagnostic_metric_calculations_
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. 指標
- [x] 2.1 予測の診断の指標の導出と、指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との、全部の項目の照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_prediction_diagnostic_metric_derivation、FedsdaRunMetrics_
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. 記録の表
- [x] 3.1 標本ごとの予測の列と、clientごとの重みの更新の回数の表
  - goldenの3ケースと、小さい条件の、実旧が保存した配列との照合を先に書く。
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 4. 記録と検証
- [x] 4.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [ ] 4.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.1, 3.1, 4.1_

## Implementation Notes

### 実装（task 1.1〜4.1）

- 手順: 計算（1.1）は、testを先に書いた（sourceがない状態での実行は、していない）。指標の導出（2.1）は、導出のmoduleを先に書き、照合のtestの編集と、指標の型への追加を、同じコマンドで入れた（testが先に失敗することは、確かめていない——逸脱）。表（3.1）は、testの期待を先に直し、表の形のtestが失敗することを確かめてから、sourceを直した。
- 導出の最初の実行で、小さい条件のtestが、14件、testの側の誤り（照合の呼出しを、旧の集計を作る行より前に置いた）で失敗した。呼出しの位置を直した。sourceは、変えていない。
- 依存境界: 新しい2つのmoduleを、moduleごとの許可集合へ登録し、importが変わった2つのmoduleの許可集合を、実際のimportへ合わせた。計算のmoduleは、NumPy（`mean`・`std`）を使う（旧と同じ式で、クラスの間の平均と標準偏差を計算するため）。

### 照合の結果（Windowsの基準環境）

- 計算: 乱数で作った列（区間ごとの正解の割合72条件——変更位置が標本の列より後にある場合、区間が重なる場合、数える標本を与える場合を含む——、クラスの間のまとめ40条件）で、実旧の`_routing_window_accuracies`、旧と同じ式と一致。
- 指標: 旧の40の指標（37項目の対応と、最終構成ではいつも0の3項目）が、goldenの3ケースの実旧の実行結果と、小さい条件（7条件×設定2通り）の実旧の`_add_model_diagnostic_results`の値（回復の窓を、条件の値にして呼ぶ）と、`==`で一致。実際の予測の正解の割合は、既存の指標`prediction_accuracy`とも同じ値。
- 対応表の取り違えの検査: どの条件でも0のままの指標は、ない。取り違えを検出できない対は、`routing_mixture_accuracy`と`routing_switching_accuracy`だけ（最終構成では同じ値。新は、1つの値で持つ）。
- 表: 旧の19の配列（標本ごと6、clientごと4、clientとクラスごと9——標本ごとの表から数える——）が、goldenの3ケースと、統合を通る小さい条件（3条件）で、形と値で一致。予測重みの側の、clientごとの4つの回数は、小さい条件で、実旧のclientの値と一致。観測した真の概念の列は、評価用の概念列と一致。照合が通った場合: 実効モデル数が1より大きい標本、正否の列が互いに違う標本、globalの診断重みの、モデルの集まりの変更・概念での再始動・集約後の再較正、予測重みの最大重みのモデルの交代。通っていない場合は、ない。

### 未検証・残る制約

- 真の概念が分からない標本（`observed_concept_id`がNone）は、実験では現れないので、実旧との照合では通っていない（表での−1と、集計から外すことは、コードの上の扱い）。
- `sample_prediction.global_diagnostic_prediction_is_correct`と`.highest_confidence_model_prediction_is_correct`は、旧に、標本ごとの配列がない。合計（指標）と、クラスごとの数（確信度が最大のモデル）で、照合した。
- 「記録から導く（未照合）」の区分は、これで、全部、照合済み（移行の一覧）。
