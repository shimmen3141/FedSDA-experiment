# Research

## 版5と新しい既定の差の下調べ（2026-10-16、Windows）

goldenの版5の全19ケースを、版5の設定と、予測だけを新しい既定（予測に使うパラメータを混ぜる値——現在のモデルだけで予測するケースは局所のまま——、T120）にした設定で実行し、比べた。script: `../../venv/refactoring-tests/v6_probe.py`（gitの対象外）。出力: `../../venv/refactoring-tests/v6-probe-columns.log`（列の単位。最初の出力`v6-probe.log`は表の単位で、仕様レビュー1回目の指摘を受けて取り直した）。

- 全19ケースで、`global_parameters`と`coverage`が同じ。
- FedDrift・FedAvgの4ケースと`fedsda-single-model-prediction`は、全部が同じ。
- 違った記録の表の列（`sample_prediction`の列を除く）: `client_weight_update_counts.fixed_share_weight_leader_switch_count`、`model_registration.maximum_weight_prediction_sample_count`、`server_round.local_processing_concept_specific_part_inference_example_count`、`server_round.local_processing_concept_specific_part_inference_forward_multiply_accumulate_count`、`server_round.local_processing_shared_part_inference_example_count`、`server_round.local_processing_shared_part_inference_forward_multiply_accumulate_count`。
- `sample_prediction`の表が違ったのは、sine2・mnist2の4ケース（`fedsda-sine2-events`、`fedsda-mnist2-decisions`、`fedsda-sine2-reference-reuse`、`fedsda-sine2-alternative-reference-reuse`）。小さい条件の構成のケースは、推論の回数だけが違った。
- 違った指標（`prediction_diagnostic_metrics.`を除く）: `prediction_accuracy`、`stable_period_prediction_accuracy`、`model_computation_counts.`の推論の4項目、`computation_cost_summary.`の9項目（`client_concept_specific_part_forward_multiply_accumulate_count`、`client_concept_specific_part_forward_multiply_accumulate_count_per_held_model_sample`、`client_forward_and_backward_multiply_accumulate_count_per_held_model_sample`、`client_forward_and_backward_multiply_accumulate_count_per_processed_sample`、`client_forward_multiply_accumulate_count`、`client_forward_multiply_accumulate_count_per_held_model_sample`、`client_forward_multiply_accumulate_count_per_processed_sample`、`client_shared_part_forward_multiply_accumulate_count`、`client_shared_part_forward_multiply_accumulate_count_per_processed_sample`）。`fedsda-independent-models`は共有部を持たないので、共有部の項目は違わない。
- 推論の回数が増えるのは、混ぜる予測が、同期の時点の結合予測に加えて、局所の結合予測も計算するため。通信量は変わらない。
