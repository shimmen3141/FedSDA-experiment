# 命名表: fedsda-complete-run-settings

sourceの公開する名前（module、class、公開の関数）だけを載せる。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `settings_serialization` / `convert_settings_to_plain_mapping` | `core/` | dataclassの設定を、保存用の入れ子の辞書へ変換する |
| `fedsda_run_settings` / `FedsdaRunSettings` | `runtime/` | 最終構成のFedSDAの1 runに要る、全部の設定 |
| `FedsdaRunSettings.execution_settings`・`run_participant_settings`・`model_consolidation_settings`・`run_metric_settings` | 同上 | 実行の枠の設定、参加者の設定の束、統合の設定、指標の設定 |
| `fedsda_final_configuration_run_settings` / `build_final_configuration_fedsda_run_settings` | `runtime/` | 最終構成の既定値を持つ、完全なrun設定を返す |

## 判断が必要な点

- 「run settings」は、既存の`ValidatedExperimentRunSettingsSubset`（検証済みの部分型）・`FedsdaRunClientSettings`・`FedsdaRunParticipantSettings`の語に合わせる。「complete」は型の名前に入れない（部分型のほうが`Subset`と名乗っている）。
- 「final configuration」は、docs/overview/proposed-method.mdの「最終提案構成」。旧の呼び名（Switchingほか）は使わない。
- 「plain mapping」は、JSONにできる値だけの辞書。
