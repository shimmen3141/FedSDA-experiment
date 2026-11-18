# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `random_concept_schedule` / `generate_random_concept_schedule` | `src/single_stream_hard_assignment_soft_prediction/` | 1本の系列の概念列を、製品のランダムな概念列と同じ規則・乱数の消費の順で作る（再来はランダムな変化の結果として生じる）。概念の数を引数で受け取る | 製品の`generate_random_client_concept_traces`（全client、概念の数をデータセットの定義から取る）の1クライアント分の写し。製品の`federated_learning_experiments.data.concept_schedules.random_concept_schedule_generation`・`random_concept_schedule_settings`（連合学習のclientの概念列と、その設定）とは別のパッケージのmodule。riverとtorchを使わない |
| `river_synthetic_recurring_streams` / `RiverSyntheticStreamDefinition` / `RIVER_SYNTHETIC_STREAM_DEFINITIONS` / `generate_river_synthetic_stream` | 同上 | Riverの合成データの生成器（STAGGER・Agrawal・Random RBF）の再来の系列の定義と、系列の配列（特徴・ラベル・概念）の生成 | 製品の`DATASET_DEFINITIONS`（連合学習のデータ）とは別。名前は重ならない。研究ツール`stream_export`（製品のrunから系列を写す）とは別の作り方 |
| `river_synthetic_stream_command` / `main` | 同上 | 新しい系列を、既存の書き出しと同じ形式（NPZとJSON）で書き出すコマンド | 研究ツール`stream_export`（製品のrunの系列の書き出し、torchを使う）と同じ形式だが、事前学習の標本を書かない。木の版の`single_stream_command`（系列を処理する）とは別 |
