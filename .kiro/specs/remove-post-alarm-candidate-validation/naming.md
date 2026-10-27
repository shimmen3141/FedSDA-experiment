# Naming

| 名前 | 意味 |
|---|---|
| （新しいモジュール、実装担当が決める） | 候補検証のファイルから移す、警報の区間での候補の学習（`TrainedAlarmIntervalCandidate`、`train_candidate_on_alarm_change_interval`）の置き場所。候補検証を指さない名前にする |
| 外すcaseの一覧（定数、実装担当が決める） | goldenの照合から外す、候補検証ありの方針のcase（版ごと） |
| 除く項目の一覧（定数、実装担当が決める） | goldenの照合で除く、消した記録の表の列・指標・設定の項目・実行の段 |

実装担当が名前を決めたら、この表を実装と同じに直す。既存の命名規約（.kiro/steering/）に従う。
