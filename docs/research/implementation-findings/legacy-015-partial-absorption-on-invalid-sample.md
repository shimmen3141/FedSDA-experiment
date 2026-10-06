# LEGACY-015: 標本吸収が途中の標本で失敗すると先行標本の更新が残る

- 発見日: 2026-10-07。対象: 旧clients/base.py::_absorb_into_store、固定基準748c3aa。
- 発見spec: assigned-training-sample-absorption。種別: 契約外入力での部分更新。状態: 再現済み・未修正。

## 再現と観測

実旧`SharedBackboneClassConditionalESRFedSDAClient`（4クラス、現在のモデル9）へ、3件の標本列の2件目だけ特徴数を3（モデルの入力は2）にして実_absorb_into_store(9, 標本列)を呼ぶ。再現は tests/refactoring/test_assigned_training_sample_absorption.py::test_actual_legacy_absorption_keeps_earlier_samples_when_a_later_sample_fails。
2026-10-07基準venvで実行し、RuntimeError（forwardの形状不一致）と次の状態を観測した。

- train_data_store[9]は2件増える（1件目と、不正な2件目）。
- model_concept_counts[9]の合計は2増える。
- model_stats[9]の件数は1だけ増える（1件目）。2件目は標本と概念だけ追加され、統計は更新されない。3件目は未処理。

旧は標本ごとに、標本追加→概念計数→損失評価→統計更新の順に行い、損失評価の失敗より前の更新を巻き戻さない。不正な標本自体も学習標本に残る。

## 影響と今回の扱い

吸収する標本は、通常の経路では予測・検出を通過済みの形状の正しい標本であり、本再現は契約外入力。正常client経路・過去実験成果への影響は未確認で、過去成果が破損しているとは判断しない。
新実装は全標本の検証と損失評価を状態変更より前に行い、どれか1件でも拒否されれば標本・割当概念計数・統計を変更しない（assigned-training-sample-absorptionの要求2.4/2.5）。正常時の標本列・計数・統計は旧と一致する。旧production・goldenは修正しない。

## 将来の修正候補

旧を保守する場合は、損失評価を標本追加より前に行う。新実装へ移行するなら旧の修正は不要。担当/修正commitは未定。
