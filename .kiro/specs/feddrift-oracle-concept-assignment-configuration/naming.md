# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_ORACLE_CONCEPT_ASSIGNMENT_CONFIGURATION_NAME`（値`feddrift-oracle-concept-assignment`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、学習の割り当てを真のコンセプトで行い、予測は直前にラベルを見た標本のコンセプトのモデルで行う構成 | 研究ツール`tools/research_oracle_concept_assignment`の変種（FedSDAの警報と帰属を真の変化で置き換えるもの）とは別。こちらはFedDriftの上の比較手法 |
| 割り当ての方式の設定の値（例: `true_concept_assignment`。実装担当が既存の設定の名前に合わせて決め、ここへ追記する） | FedDriftの設定 | 割り当てを真のコンセプトで行う。既定は今のFedDriftの挙動 | — |
