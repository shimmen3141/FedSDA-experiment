# NEW-005: 最小記述長の選択の計算が、計算回数の指標に数えられていない（誤認。数えられていた）

- 発見日: 2026-10-21。対象: `runtime/alarm_interval_minimum_description_length_selection.py`の`select_alarm_interval_assignment`。発見の経緯: MNISTの警報の増加の調査（[設計の14](../fedsda-unified-concept-assignment-inference.md)）。
- 状態: 誤認として閉じた（2026-10-21、同じ日に実測で確かめた）。

## 経緯

選択の計算（全部の保有モデルでの対数損失の評価と、新しいモデルの学習者の予測と更新）が、計算回数の指標`model_computation_counts`に入っていないと考えて記録した。計測の仕組みを確かめずに、選択のコードに計数の呼出しが無いことだけから判断した誤り。

## 確かめたこと

計測（`learning/models/model_computation_measurement.py`の`measure_model_computation`）は、PyTorchのhookをprocessの全体に登録し、FedSDAの全体runの間（`runtime/fedsda_measured_run_execution.py`）、すべてのモデルの順伝播とoptimizerの更新を数える。選択の計算も、moduleの呼出しとoptimizerの更新で行うので、数えられる。

手元の小さいrun（SINE2・900標本・2 client・間隔50、Git管理外の一時script）で、選択の中だけを別の計測で数え（学習1,290標本、推論2,702標本、更新1,290回ほか）、選択を2回実行させたrunと1回のrunの計数の差が、全部の項目で、選択の中の計数とちょうど一致することを確かめた。

## 規模の参考（上の小さいrun）

runの全体（概念固有部の学習103,858標本、推論27,262標本）に対し、選択の計算は、学習の約1.2%、推論の約10%、共有部の推論（17,848標本）の約22%だった。基準版10の本規模の計算量の値は、この計算を含む。
