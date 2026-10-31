# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_ORACLE_CONCEPT_ASSIGNMENT_CONFIGURATION_NAME`（値`feddrift-oracle-concept-assignment`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、学習の割り当てを真のコンセプトで行い、予測は直前にラベルを見た標本のコンセプトのモデルで行う構成 | 研究ツール`tools/research_oracle_concept_assignment`の変種（FedSDAの警報と帰属を真の変化で置き換えるもの）とは別。こちらはFedDriftの上の比較手法 |
| `model_assignment_policy`（値`true_concept_assignment`、既定`detection_batch_assignment`） | `runtime/feddrift_run_client_settings.FeddriftRunClientSettings`。`experiment_spec/options.py`では`feddrift_model_assignment_policy`として登録 | 割り当てを真のコンセプトで行う。既定は今のFedDriftの検出バッチの帰属。既定値は保存設定から省き、既存のrun識別子を維持する | FedSDAの帰属方式とは別で、FedDriftの学習・集約の前段を切り替える |
