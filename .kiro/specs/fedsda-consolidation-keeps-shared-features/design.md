# Design Document

## Overview
統合の加重平均（`_compute_sample_weighted_mean_parameters`、`runtime/model_clustering_and_consolidation.py`）で、メンバーの共有部がビット単位で同じなら、代表（クラスタの最小のID）の共有部をそのまま使う扱いを足し、既定にする。

| 設定 | これまで | 新しい既定 |
|---|---|---|
| 統合での共有部の扱い（新しい項目`FedsdaRunParticipantSettings.consolidated_shared_feature_policy`） | （項目なし。`average_shared_features_with_concept_specific_parts`に当たる） | `keep_identical_shared_features` |

## Boundary Commitments
### This Spec Owns
参加者の設定の新しい項目と検査、サーバ・同期・統合への受け渡し、`_compute_sample_weighted_mean_parameters`の分岐、最終構成の既定、旧と比べる設定の入口、基準版のhelper（版1〜8の明示、版9）、既定に依るtest、文書。
### Out of Boundary
概念固有部と損失統計の加重平均、クラスタリング、集約、通信量の記録（版8のまま）、FedDrift・FedAvg、固定旧。
### Allowed Dependencies
新しいmoduleは無い。依存境界の許可へ、新しくimportする名前を登録する。
### Revalidation Triggers
統合を通る全test、既定のbuilderを通るtest、掃引のrunの識別子、基準版のhelper。

## Architecture
### 設定と受け渡し（Requirement 1.1、2.1）
- `FedsdaRunParticipantSettings`に必須の項目`consolidated_shared_feature_policy`（値は`average_shared_features_with_concept_specific_parts`と`keep_identical_shared_features`）を足し、`__post_init__`で検査する（値の組と検査の関数は、`model_clustering_and_consolidation.py`に置く。v8の`cross_evaluation_model_transfer_policy`と同じ形）。
- `assemble_fedsda_run_server`・`FedsdaRunServer`・`synchronize_models_in_server_round`・`cluster_and_consolidate_global_models`へ、この値を渡す（`synchronize_models_in_server_round`と`cluster_and_consolidate_global_models`の引数は、既存の呼出しを変えないよう、旧の扱いを既定値に持つ。参加者の設定とサーバの組立ての項目は必須）。

### 統合の加重平均（Requirement 1.2〜1.4）
- `_compute_sample_weighted_mean_parameters`に、引数`keeps_identical_shared_features: bool`を足す。真で、正の件数を持つメンバー（今の平均に参加するメンバー）の共有部（`split_shared_and_concept_specific_parameters`の共有部）が、全部、代表（クラスタの最小のID。件数が0でもよい）の共有部とビット単位で同じ（同じ名前の集合で、各tensorのdtype・形・byte列が同じ）なら:
  - 共有部は代表の値をそのまま返す（写し）。
  - 概念固有部は、今と同じ加重平均（足す順も同じ）。
  - 計算回数（戻り値の、重み付きの和へ足した値の数）は、概念固有部だけを数える。
- それ以外（偽、または共有部が同じでない——代表が、件数が0で、ほかのメンバーと違う共有部を持つ場合を含む——、または件数がすべて0）は、今と同じ。共有部を持たない構造（独立のモデル）では、共有部が空どうしで条件を満たすが、平均する共有部が無いので、値も計算回数も今と同じになる（testで確かめる）。
- 統合の対象は、そのラウンドに集約したモデル（`aggregated_global_model_ids`）で、集約はその全部へ同じ共有部を入れるので、通常は条件を満たす。条件を満たさない場合の扱いは、上のとおり安全側（今と同じ平均）にする。
- 比べ方の関数は、v8で`server_round_synchronization.py`に置いた`_tensors_have_the_same_bits`を、両方が使える場所（`server_model_registration_and_aggregation.py`の、`split_shared_and_concept_specific_parameters`の隣）へ移して共有する（同じ規則を2か所に書かない）。

### 旧と比べる設定の入口（Requirement 3.1）
- `use_legacy_compatible_policies`は、参加者の設定の`consolidated_shared_feature_policy`を`average_shared_features_with_concept_specific_parts`にする。
- 実旧との照合に使う設定の組（`test_fedsda_stream_protocol_run.py`の組立て、fresh script）は、旧の扱いを明示する。

### 基準版（Requirement 3.2、4）
- 版1〜8: 参加者の設定の「後に足された項目」の表（v8で足したもの）へ、`consolidated_shared_feature_policy`＝旧の扱いを、版1〜8とも足す（版8は、この項目だけ）。
- 版9を足す（ケースは版8と同じ21）。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を9へ上げるのは、goldenを置いた後。
- 版8→9の差の説明のtest `test_baseline_9_differs_from_8_only_where_consolidation_occurred`: 全ケースで、FedDrift・FedAvgのケースと、統合が起きないFedSDAのケース（版8の`coverage`の`merged_model_count`が0）は、`run_metrics`（指標の全部）・`record_tables`・`global_parameters`・`coverage`が版8と同じ。`run_settings`は、別のtestで、FedSDAのケースの新しい項目だけが違う（新しい既定の値）ことを確かめる。統合が起きたFedSDAのケース（`merged_model_count`が1以上）が少なくとも1つあること（比較が空でない）。

### 共有部の送り直しが無くなること（Requirement 4.3）
- 新しいtestで、全体run（goldenの`fedsda-sine2-adoption-without-validation`の条件で、統合での共有部の扱いだけを替えた2つ）で、`_record_consolidated_model_resend_volume`を包んで、ID対応が空でない（統合が起きた）呼出しの数と、そのうち統合の後の共有部が先に送った共有部とビット単位で違う呼出しの数を数える。新旧の両方で、統合が起きた呼出しが1回以上あること、新しい扱いでは共有部が違う呼出しが0回、旧の扱いでは1回以上であることを確かめる（旧の扱いで1回も違わなければ、条件を替えて選び直す）。

## 旧と違う点
既定を使う実行のFedSDAの構成で、統合が起きると、代表の共有部の値が、旧の扱いと最大で約1e-7違う（旧の扱いが、同じ値の加重平均の誤差を含む）。その差が後の学習へ伝わるので、統合が起きたrunでは、精度・記録が、わずかに変わりうる。通信量は、共有部の送り直しが無くなる分、減る。

## Validation Design
- 部品: `_compute_sample_weighted_mean_parameters`で、共有部が同じ場合は代表の値の写しを返し、概念固有部は旧の扱いと同じ値、計算回数は概念固有部だけ。共有部が違う場合と、偽の場合は、旧の扱いと同じ。件数がすべて0の場合は、今と同じ。
- 全体run: 上の送り直しのtest。
- 設定の検査、既定、全構成の組立て。
- 基準版: 版1〜8の照合（値を変えずに）、版9の照合、版8→9の差の説明。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜9と、このspecのtest）。

## File Structure Plan
- `src/federated_learning_experiments/runtime/`: `model_clustering_and_consolidation.py`、`server_model_registration_and_aggregation.py`（比べ方の関数）、`server_round_synchronization.py`、`fedsda_run_server.py`、`fedsda_run_participant_factory.py`、`fedsda_final_configuration_run_settings.py`、`core/run_provenance.py`。
- `tests/refactoring/`: 新規`test_fedsda_consolidation_keeps_shared_features.py`、`test_model_clustering_and_consolidation.py`（必要なら）、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v9/`、`test_fedsda_run_client.py`、`test_fedsda_stream_protocol_run.py`、`test_fedsda_run_server.py`、`fresh_process_smoke.py`、依存境界、既定に依るtest。
- 文書: `docs/experiments/refactoring-baseline.md`、`docs/experiments/metrics.md`、`docs/experiments/experiment-sweep.md`、研究記録（IMPROVE-020）、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.4 | 新しい項目、加重平均の分岐、部品のtest |
| 2.1 | 既定、構成のtest |
| 3.1〜3.2 | 旧と比べる入口、版1〜8の明示と照合 |
| 4.1〜4.3 | 版9、版8→9の差の説明、送り直しのtest |
| 5.1 | 文書 |
