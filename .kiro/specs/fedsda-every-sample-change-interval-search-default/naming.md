# 命名と役割

命名の方針: 設定の値の名前（`minimum_interval_multiples_with_forward_prequential_code`、`every_sample_with_reverse_prequential_code`）にそろえる。会話で出ただけの語（「R1」）を公開名にしない。

## 新しい公開名

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `MINIMUM_INTERVAL_MULTIPLES_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`（値`fedsda-minimum-interval-multiples-change-interval-search`） | `sweep/experiment_configurations`の定数 | 最終構成の既定から、変化区間の候補の探し方だけを今の扱い（基準版15までの既定）に替えた構成 | `EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`は、基準版16から既定と同じ構成 |
| `non_minimum_interval_multiple_change_interval_selection_count` | goldenの経路の記録（`tests/refactoring/behavior_golden_cases.py`、版16から） | 保有モデルか新しいモデルを選んだ警報のうち、選んだkが最小変化区間の倍数でない数 | runの指標ではない（goldenの経路の記録だけ） |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`（`fedsda-every-sample-change-interval-search`） | 既定から探し方だけを新しい扱いに替えた構成 | 既定と同じ構成（これまでのコマンドと結果の記録のために残す） |
| 最終構成の既定の`change_interval_search` | `minimum_interval_multiples_with_forward_prequential_code` | `every_sample_with_reverse_prequential_code` |
