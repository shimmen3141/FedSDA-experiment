# Design Document: fedsda-overall-loss-esr-detection

## 変更

- `LossChangeDetectionSettings.loss_monitoring_scope`へ、値`overall_loss`を足す。
- `OverallAndTrueClassLossMonitor`（`methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py`）: 設定の対象が`overall_loss`のとき、全体の検出器の観測を、そのまま監視の観測にする（混合の重みは掛けない。クラス別の検出器は作らない。更新の件数は1）。
- `runtime/fedsda_run_client_settings.py`: 検出の方式と表示名の対応へ、（`e_sr`、`overall_loss`）→`e-SR`を足す。
- 掃引: 構成`fedsda-overall-loss-esr-detection`を足す。

## クラスの名前を変えない理由

監視のクラス（`OverallAndTrueClassLossMonitor`）は、標本1件の処理ほか、多くの関数が、型として受け取る。対象ごとに別のクラスにすると、それらの型と検査を変えることになる。監視の対象は、すでに設定（`loss_monitoring_scope`）として、このクラスへ渡っているので、クラスの中で分ける。名前は、提案構成の対象のままで、全体だけの対象も扱う、と説明を足す。

## 検証

- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を足す（旧の側は、構成名を替える）。小さい条件の7つで、実旧の全体runと照合する。固有の経路のtestを足す。
- `tests/refactoring/test_loss_change_monitoring.py`: 対象が全体だけの監視を、旧の検出器1つ（`BoundedMeanEDetector`）と、標本ごとに照合する（e値、警報、候補の位置、計算の件数、始め直し）。
