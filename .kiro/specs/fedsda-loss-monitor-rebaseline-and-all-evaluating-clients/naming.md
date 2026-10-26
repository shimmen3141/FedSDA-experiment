# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `KEEP_BASELINE_UNTIL_ALARM_POLICY_NAME`、`LOWER_BASELINE_ON_LOSS_DECREASE_POLICY_NAME`、`LOSS_MONITORING_BASELINE_UPDATE_POLICIES` | `methods/fedsda/loss_change_detection/loss_change_detection_settings.py`の定数 | 損失の監視の基準の扱いの2つの値（「基準を警報まで保つ扱い」——今の扱い——と「損失の低下で基準を下げる扱い」）の名前と、その組 |
| 基準版13 | goldenの版 | 損失の低下で監視の基準を下げる扱いと、クロス評価の評価clientの上限をclientの数にする扱い（新しい既定）の値 |

設定の項目`loss_monitoring_baseline_update_policy`、観測の項目`baseline_lowered`、状態の写しの新しい項目、runの指標と経路の記録の項目`loss_monitoring_baseline_lowering_count`、
監視器の非公開の下がる向きの検出器と、関数の引数は、設計に書く。

似た名前との違い:
- `SHORT_CHANGE_INTERVAL_ALARM_POLICIES`（既存）: 検出器の合図があり推定の変化区間が短いときに、警報とするかどうか（警報の時機）。新しい組は、警報の間に監視の基準を変えるかどうか。
- `LOWER_BASELINE_ON_LOSS_DECREASE_POLICY_NAME`の「lower」は、基準を下げる向きにだけ変えることを指す（上げない）。研究ツールの変種名の「decrease-rebaseline」と同じ扱い。
- 研究ツールの`LOSS_MONITOR_REBASELINE_VARIANTS`・`loss_monitor_rebaseline_policy`（既存、`tools/`）: 製品の外から監視器を包む研究用の実装。製品の新しい扱いは、監視器の中の正式な実装で、研究ツールの変種は基準版12の上の意味を保つ。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `LossChangeDetectionSettings` | 検出方式・監視対象・誤警報の制御・警報の時機 | 加えて、監視の基準の扱い（ADWINでは今の扱いだけ） |
| `OverallAndTrueClassLossMonitor` | 基準は警報の応答の`reset`でだけ変わる | 設定により、損失の低下を検出したとき、基準を今のモデルの保存平均へ下げる |
| `build_final_configuration_fedsda_run_settings` | 評価clientの上限は3 | 評価clientの上限はclientの数、監視の基準は損失の低下で下げる |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 12 | 13 |
