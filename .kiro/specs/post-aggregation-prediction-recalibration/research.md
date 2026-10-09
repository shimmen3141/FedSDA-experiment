# Research & Design Decisions

## Summary

- **Feature**: `post-aggregation-prediction-recalibration`
- **Discovery Scope**: Extension（移植済みの部品をつなぎ、診断証拠へ2つの操作を足す）
- **Key Findings**:
  - 旧の最終構成の再較正（`fifo_replay`）は、保留中の標本の損失の列を1つ作り、globalのAdaHedgeとFixed-Shareを、その列で再生し、真の概念別のAdaHedgeを再始動する。
  - Fixed-Shareの再生と、標本ごとの有界損失の計算は、移植済み。足りないのは、AdaHedgeの診断証拠の、集約後の再生・再始動と、3つの計数である。
  - 再較正を足すと、対照を、旧のサーバの`run_round(t, clustering_enabled=False)`そのものにできる。

## Research Log

### 旧の再較正と、対応する新部品

- **Sources Consulted**: 旧`servers/shared_backbone.py`（25〜30行）、`clients/shared_backbone.py`（80〜168行）、`expert_routing.py`（AdaHedge 143〜193行、Fixed-Share 496〜518行）、`clients/fedsda.py`（1296〜1308行。ルータの生成）、`experiment.py`（1267〜1300行、1558〜1595行。計数の集計と保存）。
- **Findings**:

| 旧の段 | 新部品 |
| --- | --- |
| 損失の列（`_fifo_routing_loss_sequence`） | 保留中の標本は`PendingSampleObservationStore`、損失は`evaluate_classifier_per_sample_bounded_losses`（2値は出力と観測ラベルの差の絶対値、多クラスは1−観測クラスの確率。旧と同じ式） |
| globalのAdaHedgeの再生（`expert_router.replay_after_aggregation`） | 診断証拠へ操作を足す（本spec） |
| 真の概念別のAdaHedgeの再始動（`oracle_concept_expert_routers`の`restart_after_aggregation`） | 診断証拠へ操作を足す（本spec） |
| Fixed-Shareの再生（`switching_expert_router.replay_after_aggregation`） | `FixedSharePredictionWeightController.replay_observed_losses_after_aggregation` |

  - 旧は、共有部の特徴を、モデルIDが最小のモデルで1回だけ計算して、各モデルの概念固有部へ渡す。新の損失の部品は、モデルごとに分類器全体を1回呼ぶ。全保有モデルは1つの共有部につながっているので、同じ入力に対する同じ演算で、値は同じになる（対照testで確かめる）。
  - 旧のAdaHedgeの再生は、証拠を消してから、行ごとに`probabilities(losses)`と`update(losses, probabilities)`を呼ぶ。行のモデル集合が証拠と違えば、`probabilities`が集合を同期する（消した直後は空なので、モデル集合の変化の回数は増えない）。新の`get_diagnostic_weights_before_loss_observation`と`update_evidence_after_loss_observation`が、同じ同期を行う。
  - 旧の計数: AdaHedgeは`aggregation_restart_count`・`aggregation_recalibration_count`・`aggregation_recalibration_sample_count`（ほかに、最終構成では増えない`aggregation_recalibration_check_count`・`skip_count`）。`experiment.py`が、globalのルータの計数を集計して保存する。
  - `context_expert_routers`と`shadow_meta_routers`は、最終構成（文脈`switching`）では空のままで、新は持たない。計算量の記録（`_record_model_compute`）は未移植。
  - 再較正の方式: 旧の既定は`none`、最終構成は`fifo_replay`。新の設定（`prediction_weight_recalibration_after_aggregation_policy`）は、`recompute_buffer_losses_and_replay_weight_updates`だけを持つ。
- **Implications**: 再較正は、損失の列を計算し、既存の部品と、足した操作を、旧の順に呼ぶ関数にする。

### oracle

- 登録→集約→配布の対照（tests/refactoring/test_global_model_distribution.pyの`run_synchronization_round_in_both`）の、旧の側を、実旧のサーバの`run_round(round_index, clustering_enabled=False)`にする。旧の設定の差し替えへ、再較正の方式を足す（既存の対照は`none`のまま）。
- 単一の診断証拠は、実旧の`AdaHedgeRouter`の`replay_after_aggregation`・`restart_after_aggregation`と照合する。
- clientの全状態の照合（`assert_adahedge_matches_legacy`）へ、3つの計数を足す。

## Design Decisions

### Decision: 全clientへ順に呼ぶ関数は置かない

- **Context**: 旧は、サーバの`run_round`が、全clientの再較正を呼ぶ。
- **Selected Approach**: clientの操作`FedsdaRunClient.recalibrate_prediction_state_after_aggregation`までを本specで作り、全clientへ順に呼ぶのは、呼出し側（対照test、共用script。後で、実行の枠のサーバの操作の実体）にする。
- **Rationale**: clientをまたぐ状態がなく、順に呼ぶだけである。ラウンド全体の組立て（登録→集約→統合→配布→再較正）は、統合がそろうspecで作る。

### Decision: 損失の列の計算と検査を、状態の更新より前に済ませる

- **Selected Approach**: 検査→損失の列の計算→globalの診断証拠の再生→真の概念別の診断証拠の再始動→Fixed-Shareの再生（旧と同じ順）。診断証拠の再生は、列の全行を検査してから、計数と証拠を変える。
- **Rationale**: 3つの状態は互いに独立で、同じ列を使う。列は、再較正の関数が、検査済みの値から作るので、後の段の検査では拒否されない。

### Synthesis

- **Build vs. Adopt**: 損失の計算とFixed-Shareの再生は既存の部品。新しく書くのは、診断証拠の2つの操作と、全体の順序。
- **Simplification**: 旧の、ほかの再較正の方式（集約後の再始動だけ、首位が変わったときだけの再生）は移植しない（設定に選択肢がない）。

## Risks & Mitigations

- 既存の完了済みのmodule（単一の診断証拠）へ操作を足す — 追加だけにし、既存の操作と契約を変えない。
- 共有部の特徴の計算の回数が旧と違う — 値が同じであることを、実旧との対照（全状態の一致）で確かめる。

## References

- [global-model-distribution](../global-model-distribution/) — 配布と受取り、ラウンドの対照。[adahedge-diagnostic-evidence](../adahedge-diagnostic-evidence/)、[held-adahedge-diagnostic-notification](../held-adahedge-diagnostic-notification/)。
