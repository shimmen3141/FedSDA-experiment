# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `fedsda-with-candidate-validation`（`CANDIDATE_VALIDATION_CONFIGURATION_NAME`） | 構成の名前 | これまでの既定（候補検証あり、候補の学習で共有部も更新する）。新しい既定から、採否の方針と候補の共有部の2つを替える。`fedsda-without-candidate-validation`（候補検証なし、共有部も更新する）とは、採否の方針だけが違う |
| 基準版7 | goldenの版 | 新しい既定（候補検証なし＋候補の共有部の固定）の値 |

新しいmodule・class・公開の関数は無い（設定の項目の追加は、設計に書く）。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `fedsda`（既定） | 候補検証あり | 候補検証なし＋候補の共有部の固定 |
| `fedsda-without-candidate-validation` | 既定から、候補検証だけを外した構成 | 既定から、候補の共有部の固定を外した構成（設定は、これまでと同じ。名前は過去の結果との連続のために残す） |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 6 | 7 |
