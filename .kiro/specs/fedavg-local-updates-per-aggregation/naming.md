# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `fedavg-local-updates-per-aggregation` | 構成名（`--configurations`） | 局所更新の間隔τを集約間隔と同じにしたFedAvg。roundの中は配布されたモデルで予測し、局所更新はroundの終わりにまとめる（ユーザーが選んだ名前） |
| `FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 上の構成名の定数 |

## testの補助の名前（公開しない。参考）

| 名前 | 場所 | 役割 |
|---|---|---|
| `BASELINE_CONFIGURATION_CASE_NAMES` | `tests/refactoring/behavior_golden_cases.py` | 基準版1〜5のgoldenの事例にした13構成の名前 |
| `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN` | 同上 | goldenの事例にせず、専用のtestで守る構成名と、そのtestのファイル |

## 既存の公開名で、役割が変わるもの

無し（`fedavg`は変えない）。
