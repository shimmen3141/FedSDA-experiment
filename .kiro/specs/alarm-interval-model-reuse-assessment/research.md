# 調査根拠

旧基準748c3aa。主担当とread-only補助担当session_progress_researchがコードを照合。テストは未実施。

- 旧clients/fedsda.py::_resolve_driftの保有モデル評価ループは、各モデルでper_sample_error→torch.mean→float item、履歴基準なし/零は評価済み候補からも除外、difference<=distance_thresholdを再利用候補へ追加する。_select_reuse_candidateはmin(loss)で同率先着。現行優先ではない。
- clients/base.py::_get_model_statsは欠損またはn<2で0を返す。警報区間はmean=0も除外する。新methods/fedsda/loss_statistics/loss_baseline_selection.py::select_alarm_interval_reuse_baseline_mean_lossが同じ基準を実装済み。
- 新learning/prediction/classifier_bounded_loss_evaluation.py::evaluate_classifier_per_sample_bounded_lossesで実旧と一致するCPU float32の二値/多クラス標本損失を得る。区間平均の組立は未実装。
- 新HeldModelTrainingStateRegistry.snapshot_ordered_held_model_training_statesとModelAndClassLossStatisticsStore.get_model_loss_statisticsは公開の取得口。初期値選択select_candidate_initial_parameter_snapshotへ評価済み候補tupleを渡せる。基準不足モデルをそこへ混ぜない。
- 通常の将来検証は現行優先・平均/ID順があり、この区間再利用判定へ転用しない。
- 旧は前区間の評価保存/現行ID吸収の後で区間を評価する。履歴の更新は今回の外側で行い、渡された時点の統計を使う。評価は保有ID/順序を保持し、負の一時IDも扱う。
- PendingTrainingAssignmentBufferのpartitionとsession開始は実装済み。位置から汎用Tensor標本を確保する接続は後続。初期値選択/候補学習/固定参照を本specで再実装しない。
- 実旧全_resolve_driftの状態更新まで含めた接続は後続。今回のoracleは評価・候補選択部分の観測を明示し、旧最終状態の変化を今回の新評価部品の更新とは取り違えない。

警報区間と将来検証の選択方針差は不具合と断定していない。同率の場合だけ現行を優先する改善仮説は[IMPROVE-001（旧ALGO-001）](../../../docs/research/improvement-candidates/improve-001-current-model-priority-on-reuse-ties.md)へ記録した（未検証/未採用）。本specでは旧の保有順を維持する。
