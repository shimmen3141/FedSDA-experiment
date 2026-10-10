# Research & Design Decisions

## Summary

- **Feature**: `fedsda-complete-run-settings`
- **Discovery Scope**: Extension（既存の設定型をまとめる）
- **Key Findings**:
  - 1 runに要る設定は、3つの型に分かれて渡されている。最終構成の値を組み立てるのは、testのhelper（`make_run_client_settings`・`make_run_participant_settings`・`make_golden_condition_settings`）だけ。
  - 旧の最終構成の既定値（2026-10-11に、旧の設定を、最終構成の固定設定で有効化して読んだ値）: FIFO 30、送信待ち1ラウンド、batch 32、変更の最小の間隔5件、e-SRのα 0.001、検出器の候補の上限1000、許容する増加量（γ）0.1、候補の最小改善量1e-4、候補のepoch 30・patience 3・検証の割合0.2、weight decay 1e-3、アダプタのrank 8、事前学習500件・10 epoch・batch 32、対の評価の最小件数5、信頼水準0.95、クロス評価のclientの上限3、学習の間隔1（更新1回）、前向き検証10件、学習率1e-2（通常・新規とも）、評価標本の上限50・追加20、クロス評価の標本の上限50、検出の遅れの許容100件、回復の窓200件、Adam（AMSGrad）、概念の変更: 最小の間隔300件・確率0.0015。規模の既定は、client 10、標本5000、集約間隔50。
  - 旧の回帰testの縮小（`COMMON`・`CASES`）が変えるのは、client数、標本数、事前学習の件数（100）とepoch（mnist2は2）、概念の変更の最小の間隔（100、mnist2は30）と確率（0.015）、回復の窓（50）。

## Design Decisions

### Decision: 完全なrun設定は、既存の3つの設定型を、そのまま束ねる

- **Alternatives Considered**: 平らな1つの型へ作り直す（約40項目）——機能別の設定型と、その検証を、作り直すことになる。挙動を変えない移植の途中で、設定の置き場所を大きく動かすと、照合の範囲が広がる。
- **Selected Approach**: `FedsdaRunSettings`が、実行の枠の設定、参加者の設定の束、統合の設定、指標の設定を持つ。部分の間の一致と、機能の組合せ（既存の`ValidatedExperimentRunSettingsSubset`）を、生成時に確かめる。
- **Follow-up**: 束の中の仮の置き場所（スカラーの束ほか）の整理は、旧実装を外した後で判断する。

### Decision: 統合の設定を、完全なrun設定に入れる

- **Selected Approach**: `ModelConsolidationSettings`（クラスタリングの実行方針、対の比較の方式、linkage、統合の方針）を、完全なrun設定のfieldにする。サーバの実装は、最終構成の方式だけを持つので、この設定は、条件の宣言と、組合せの検証に使う（実行の分岐には使わない）。
- **Rationale**: 保存する条件に、統合の方式が入る。FedDriftほかの手法や、別の方式を足すときに、分岐の入口になる。

### Decision: 最終構成の既定値は、sourceの1つのmoduleに、定数として書く

- **Alternatives Considered**: 設定ファイル（YAMLなど）——読込みと検証が増える。値は、型のある関数で足りる。
- **Selected Approach**: `build_final_configuration_fedsda_run_settings(experiment_run_conditions=...)`が、既定値を持つ完全なrun設定を返す。規模（client数、標本数、集約間隔）とseedとdatasetは、実行条件として、呼出し側が渡す（既定を持たない）。
- **Rationale**: 既定値の出どころが1箇所になる。testが、旧の設定の値と、全部の項目で照合できる。

### Decision: 既定値の変更は、`dataclasses.replace`で行う

- **Selected Approach**: 設定型は不変のdataclassなので、`replace`で一部を置き換えた値を作る（生成時の検証が、もう一度行われる）。入れ子の置き換えを短く書くhelperや、掃引の軸は、(A3)で決める。

### Decision: 保存用の表現は、設定の型から機械的に作る

- **Selected Approach**: `core`に、dataclassの設定を、入れ子の辞書へ変換する関数を置く。dataclassは`{"settings_type": 型の名前, field名: 値, ...}`、tupleはlist、数値・文字列・真偽値・Noneはそのまま。ほかの値は拒否する。
- **Rationale**: 設定のfieldを足したとき、保存の側を直し忘れない。
- **Follow-up**: 辞書から設定へ戻す変換は、必要になったとき（保存した条件からの再実行）に足す。

## Risks & Mitigations

- 既定値の写し間違い — 旧の設定を、最終構成の固定設定で有効化して読んだ値と、全部の項目を照合するtestを置く。goldenの3ケースの条件を、既定値＋縮小の置き換えで作り、既存の（goldenを再現する）設定と一致することを確かめる。
- datasetを足したとき、モデルの既定を足し忘れる — 定義のある全部のdatasetに、既定があることを確かめるtestを置く。
