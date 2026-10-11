# Design Document: fedsda-adwin-detection

## 変更

- 検出器（新規）: `methods/fedsda/loss_change_detection/adwin_loss_window_detection.py`の`AdwinLossWindowDetector`（観測の結果`AdwinLossWindowObservation`、状態の写し`AdwinLossWindowState`）。旧の`FullScanADWIN`と同じ演算の順。累積和は、窓の先頭からの逐次の和（NumPyを使わない。旧の`np.cumsum`と、同じ値になる）。
- 設定: `LossChangeDetectionSettings.drift_detector_name`へ`adwin`。ADWINの値として、`adwin_confidence_delta`と`adwin_minimum_test_window_sample_count`を足す（提案構成では使わない。最終構成のrun設定は、旧の既定の0.05と10を置く）。窓の上限は、e-SRの、保持する候補の数の上限（`loss_monitor_maximum_retained_candidate_count`）と同じ値を使う（旧も、1つの設定`ADWIN_MAX_WINDOW`を、両方に使う）。
- 監視（`OverallAndTrueClassLossMonitor`）: 検出器がADWINのとき、`_observe_loss_with_adwin`へ分ける。観測の結果へ、`overall_adwin_window_sample_count`（e-SRではNone）を足す。状態の写しへ、ADWINの窓の状態を足す（e-SRの状態は、ADWINではNone）。始め直しは、窓を空にする（基準の平均は、使わない）。
- 標本1件の処理: 監視の観測が、ADWINの窓の標本数を持つとき、検知していなければ、補助の判定（`_current_model_loss_worsened_over_adwin_window`）を行い、真なら警報の処理へ進む。保留の容量を読むため、`PendingTrainingAssignmentBuffer.capacity_sample_count`を足す。
- 検出器の表示名: （`adwin`、どちらの対象）→`ADWIN`。
- 掃引: 構成を2つ足す。損失の監視を替える構成（全体のe-SRを含む3つ）の変更を、1つの表にまとめた。

## 補助の判定の置き場所

旧は、clientの関数が、検出器・保留標本・モデル・損失統計を読む。新の監視は、モデルを持たないので、判定は、標本1件の処理に置く（保留標本、現在のモデル、損失統計、許容量が、そこにある）。監視は、判定に要る、全体の窓の標本数だけを返す。判定を行うかどうかは、設定の名前でなく、監視の観測が窓の標本数を持つかどうかで決める（e-SRは持たない）。

履歴の平均は、警報の区間での再利用の判定と同じ選び方（2件以上で、0でない平均。`select_alarm_interval_reuse_baseline_mean_loss`）で、旧の条件（`hist_mean > 0.0`。統計が2件未満なら0）と同じ。

## 補助の判定を、切る設定を持たないこと

旧は、設定`FEDSDA_ENABLE_FORCED_DRIFT_CHECK`（既定は真）で、補助の判定を切れる。旧のablationの指定は、この設定を変えない。新は、補助の判定を、ADWINでの監視の一部として、いつも行う（切る設定は、持たない）。旧で、この設定を偽にした実行に対応する構成は、新には無い。

## 監視のクラスを分けない理由

監視のクラスは、標本1件の処理ほかが、型として受け取る（前のspecの「クラスの名前を変えない理由」と同じ）。検出器ごとの処理は、クラスの中の関数で分ける。e-SRの検出器は、ADWINの設定でも作る（引数の検査を兼ねる。使わない）。

## e値を持たないこと

ADWINは、e値を持たない。監視の値は、負の無限大とし、標本ごとの監視の値の記録にも、負の無限大が入る。指標`maximum_monitored_log_e_value`は、負の無限大になる（旧は、この指標を作らない）。

## 検証

- `tests/refactoring/test_adwin_loss_window_detection.py`（新規）: 検出器を、旧の`FullScanADWIN`と、損失1つごとに照合する（3つの設定。検知、上限での切捨て、始め直し）。不正な値の拒否。
- `tests/refactoring/test_loss_change_monitoring.py`: ADWINでの監視を、旧の2つのclient（クラス別、全体）の、検出器の更新・変化の候補の長さ・計算の計数と、標本ごとに照合する。クラス別の窓だけが検知する標本を通す。
- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を2つ足す。小さい条件の7つで、実旧の全体runと照合する。窓の上限は、旧の既定（1000）にする（照合の土台の既定の50では、ADWIN自身が検知しない）。照合の中で、ADWINの監視と、補助の判定を包んで、通った経路を数える。
- 照合の土台: 実旧が、監視の値（e値）を持たない構成（`legacy_records_monitored_log_e_values`が偽）では、旧の指標`e_detector_max_log_e`が無いことと、新の値が負の無限大であることを確かめる。
