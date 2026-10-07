# 警報時の学習区間の準備 — 命名案 revision1

ファイル/公開契約の初期一覧。承認状態はspec.jsonを参照する。test/局所名はtask開始前に外部下書きASTで抽出し、追加承認する。

| 名前 | 型・役割・違い |
| --- | --- |
| indexed_observed_training_sample.py | 学習層の位置付き観測標本record。標本の所有者ではない |
| IndexedObservedTrainingSample | frozen入力。位置と観測済み標本と診断概念IDを結ぶ。ObservedTrainingSample自体には位置がない |
| sample_index | 非負builtin int。クライアント観測列の位置。roundやモデル内の位置ではない |
| training_sample | ObservedTrainingSampleの借用参照。Tensorのcopyはしない |
| observed_concept_id | intまたはNone。評価用の真概念ID、分割/モデル選択に使用しない |
| prepared_alarm_training_intervals.py | 前区間の処理が済んだ結果recordを定義 |
| PreparedAlarmTrainingIntervals | frozen返却型。BufferedChangeIntervalPartitionは位置だけの分割、本型はpayloadと前区間処理済みの情報も持つ |
| earlier_observations | tuple[IndexedObservedTrainingSample,...]。変化区間より前、処理順を保持 |
| change_interval_observations | 同tuple。変化区間、未吸収。後続の区間解決へ渡す |
| change_interval_start_sample_index | intまたはNone。FIFO切詰め後の変化区間先頭。検出器の推定開始位置と区別 |
| earlier_interval_absorbed_model_id | intまたはNone。前区間を吸収済みの帰属ID。前区間空ならNone |
| alarm_training_interval_preparation.py | runtimeの区間準備。警報全体の解決ではない |
| prepare_alarm_training_intervals | 位置照合/区間分割/前区間保存と吸収を行いPreparedAlarmTrainingIntervalsを返す。学習optimizer更新はしない |
| pending_training_assignment_buffer | PendingTrainingAssignmentBuffer。保留位置owner、読取りのみ |
| pending_sample_observations | tuple[IndexedObservedTrainingSample,...]。保留順で供給する位置付きpayload列 |
| estimated_change_span_sample_count | 正のbuiltin int。推定変化区間の標本数。実件数とは限らない |
| current_training_model_assignment | CurrentTrainingModelAssignment。現在帰属owner、ID読取りのみ |
| model_evaluation_sample_store | ModelEvaluationSampleStore。前区間の評価標本を保存、学習storeと区別 |
| python_random_generator | exact random.Random。評価抽出用の明示乱数、torch生成器と区別 |
| held_model_training_state_registry | HeldModelTrainingStateRegistry。現行分類器を取得、一覧を更新しない |
| training_sample_store | ModelTrainingSampleStore。前区間学習標本の追加先 |
| model_training_and_assignment_counts_store | ModelTrainingAndAssignmentCountsStore。既存吸収で診断概念の割当計数を更新 |
| loss_statistics_store | ModelAndClassLossStatisticsStore。前区間の全体/クラス別損失を更新 |

既存公開名は元specと同じ意味で再利用する。新永続状態を設けない。private検査helper、test名と引数/局所名は実装前の追加表で役割を承認する。
