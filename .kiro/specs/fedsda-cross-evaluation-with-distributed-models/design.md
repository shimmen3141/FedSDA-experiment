# Design Document

## Overview
サーバの1ラウンドで、クロス評価のためにモデルを送るのをやめ、配布するモデルを使う扱いを足して既定にする。状態の更新（clientの評価、統合、配布の反映）は今のまま動かし、通信量の記録だけを、新しいプロトコルの送り方に合わせる。

| 設定 | これまで | 新しい既定 |
|---|---|---|
| クロス評価のモデルの送り方（新しい項目`FedsdaRunParticipantSettings.cross_evaluation_model_transfer_policy`） | （項目なし。`send_candidate_models_for_cross_evaluation`に当たる） | `reuse_distributed_models_for_cross_evaluation` |

### プロトコルとして成り立つこと
新しい扱いのラウンド（クロス評価があるラウンド）は、次のプロトコルに当たる。

1. clientがモデルを送り、サーバが登録・集約する（今と同じ）。
2. サーバが、全clientへ、集約後の共有部と全グローバルモデルの概念固有部を送る。clientは受け取ったものを保持し、**まだ手元の保有モデルを置き換えない**。
3. サーバが、評価するclientへ、評価の依頼（候補と対象のID）を送る。clientは、受け取ったグローバルモデルを候補として、手元の保有モデル（対象）の標本で評価し、統計を返す（今と同じ値: 候補は集約後のグローバルモデル、対象はclientの手元の保有モデル）。
4. サーバがクラスタリングと統合をする。IDがまとめられたら、全clientへ、ID対応と、値が変わった代表のモデルの概念固有部と損失統計を送る。統合は、共有部を含む全パラメータをメンバーの件数で加重平均する（`_compute_sample_weighted_mean_parameters`）ので、メンバーの共有部が同じ値でも、浮動小数点の演算で、代表の共有部がビット単位で変わりうる。配布する共有部は、全グローバルモデルの最初のモデルのもの（`distribute_global_models_to_clients`）なので、統合の後のその値が、2で送った共有部とビット単位で違えば、共有部も全clientへ送り直す。
5. clientが、受け取ったグローバルモデル（代表は送り直された値）とID対応を反映する（今の配布の反映と同じ最終の状態）。

今の実装は、2の受け取りと5の反映を、統合の後の1回の配布の反映でまとめて行っている。評価（3）は、サーバのグローバルモデルの値を候補として渡すので、2で受け取った値と同じ。したがって、状態の更新の順を変えずに、通信量の記録だけを1〜5に合わせれば、新しいプロトコルの結果と一致する。

## Boundary Commitments
### This Spec Owns
`FedsdaRunParticipantSettings`の新しい項目と検査、サーバ（`FedsdaRunServer`、`assemble_fedsda_run_server`）への受け渡し、`synchronize_models_in_server_round`の通信量の記録の分岐、`cross_evaluate_global_models`と`distribute_global_models_to_clients`の、通信量を記録するかどうかの引数、配布の通信量を記録する関数（新規、`global_model_distribution.py`の中）、最終構成の既定、旧と比べる設定の入口、基準版のhelper（版1〜7の明示、版8）、既定に依るtest、文書。
### Out of Boundary
クロス評価の評価の中身・乱数、クラスタリングと統合の判定と値、clientの配布の反映、上りの通信、FedDrift・FedAvg、固定旧。
### Allowed Dependencies
新しいmoduleは無い。依存境界の許可は、変える必要がない見込み（同じmoduleの中の関数の追加と、引数の追加）。
### Revalidation Triggers
サーバの同期を通る全test、既定のbuilderを通るtest、掃引のrunの識別子（FedSDAの全runで変わる）、基準版のhelper。

## Architecture
### 設定（Requirement 1.1、2.1）
- `FedsdaRunParticipantSettings`に、必須の項目`cross_evaluation_model_transfer_policy`（値は`send_candidate_models_for_cross_evaluation`と`reuse_distributed_models_for_cross_evaluation`）を足す。`__post_init__`で、builtin strで、2つの値のどちらかであることを検査する（`RunSettingsValidationError`）。
- 最終構成の既定は`reuse_distributed_models_for_cross_evaluation`。FedSDAの全構成が引き継ぐ。
- `assemble_fedsda_run_server`と`FedsdaRunServer`へ、この値を渡し、`synchronize_models_in_server_round`の引数にする。

### 通信量の記録（Requirement 1.2〜1.5）
`synchronize_models_in_server_round`の流れ（状態の更新の順は変えない）:

1. 登録、集約（今と同じ）。
2. クラスタリングが有効で、集約したモデルが2つ以上のとき:
   - 新しい扱いなら、クロス評価の前に、`record_global_model_distribution_volume`（新規）で、その時点の全グローバルモデルの配布の通信量（共有部1個×client数、全モデルの概念固有部×client数、モデルの転送回数＝モデル数×client数）を記録する。
   - `cross_evaluate_global_models`を呼ぶ。新しい引数`record_model_transfers`（新しい扱いでは`False`）が`False`なら、メッセージ（依頼と返信）だけを記録し、共有部・概念固有部・モデルの転送を記録しない。評価・乱数・記録は今と同じ。
   - 統合（今と同じ）。
3. 配布の反映: `distribute_global_models_to_clients`を呼ぶ。新しい引数`record_communication_volume`（新しい扱いで、2でクロス評価をしたラウンドでは`False`）が`False`なら、配布の通信量を記録しない（反映は今と同じ）。代わりに、ID対応に、空でなければ（統合が起きたら）、ID対応のメッセージ（client数）と、値が変わった代表のモデル（元と行き先が違う対の、行き先のIDの集合）の概念固有部×client数と、モデルの転送回数（代表の数×client数）を記録する。さらに、配布する共有部（統合の後の全グローバルモデルの最初のモデルの共有部）が、2で記録した共有部とビット単位で違えば（`torch.equal`が偽のパラメータが1つでもあれば）、共有部×client数も記録する。損失統計は、今の配布と同じく、byteを数えない（今の通信量の指標は、配布の損失統計を数えていない）。
   - ID対応は、統合でクラスタの数がモデルの数より減ったときだけ作られ、そのときは全メンバー（単独のクラスタは自分へ）を持つ（`cluster_and_consolidate_global_models`）。したがって「ID対応が空でない」と「統合が起きた」は同じで、今の配布がID対応のメッセージを数えるラウンドと、新しい扱いで送り直しを数えるラウンドは一致する。
4. 再較正（今と同じ）。

- クロス評価をしないラウンド（クラスタリングが無効、または集約したモデルが1つ以下）は、どちらの扱いでも、今と同じ記録になる（配布の記録の関数は、今の配布の中の記録と同じ計算。今の配布の関数は、この関数を呼ぶ形にする）。
- 検査は、どの記録よりも前に行う（新しい引数がbuiltin boolであること、設定の値）。
- 代表のモデルの概念固有部は、統合の後のグローバルモデルの値から分けて数える（共有部は数えない）。

### 旧と比べる設定の入口（Requirement 3.1）
- `use_legacy_compatible_policies`: 参加者の設定の`cross_evaluation_model_transfer_policy`を`send_candidate_models_for_cross_evaluation`にする。
- 実旧との照合に使う設定の組（`valid_run_settings_mapping`、`test_fedsda_stream_protocol_run.py`の組立て、fresh scriptの組立て）は、`send_candidate_models_for_cross_evaluation`を明示する。
- 照合の入口の検査: 実旧と比べる全体runの照合（通信量の指標を比べる）は、旧の扱いでないと通信量が一致しないので、それ自体が検出になる。加えて、`assert_run_client_uses_legacy_compatible_policies`は、clientしか受け取らないので、サーバの値は検査しない（設計上の判断。全体runの照合が通信量で検出する）。

### 基準版のgolden（Requirement 3.2、4。`tests/refactoring/behavior_golden_cases.py`）
- 版1〜7の扱い: 新しい項目は、参加者の設定（`run_participant_settings`）の項目なので、clientの設定の「後に足された項目」の表とは別に、参加者の設定の「後に足された項目」の表を足し、版1〜7とも`send_candidate_models_for_cross_evaluation`にする。`build_case_settings`はFedSDAのケースでこの値を明示し、`comparable_run_settings`は値を確かめてから除く（clientの表と同じ形）。
- 版8を足す（`SUPPORTED_BASELINE_VERSIONS`、`BASELINE_VERSION`、`CASE_NAMES_BY_BASELINE_VERSION`。ケースは版7と同じ21）。必須の経路は版7と同じ。
- `BASELINE_VERSION`を8へ替えるのは、版8のWindowsとWSLのgoldenを生成して置いた後。`BEHAVIOR_BASELINE_VERSION`も同時に8へ。sourceとtestをcommitしてから生成する（中間のcommitでは、版8のgoldenの存在を確かめるtestが失敗する。前の版と同じ）。

### 版7→8の差の説明（`tests/refactoring/test_important_behavior_regression_goldens.py`）
新しいtest `test_baseline_8_differs_from_7_only_in_communication_volume`で、全21ケースについて:
1. 記録の表・`global_parameters`・`coverage`は版7と同じ。
2. 指標は、`communication_volume.`で始まる項目以外が版7と同じ。
3. FedDrift・FedAvgのケースは、指標も全部が同じ。
4. クロス評価が起きたFedSDAのケース（記録の表の`client_cross_evaluation.`の行が1つ以上）は、下りのbyteが版7より少ない。クロス評価が起きなかったFedSDAのケースは、指標も全部が同じ。
5. 少なくとも1つのケースで、下りのbyteが減る（比較が空でない）。

## 既定に依る既存のtestの扱い
| test | 今の前提 | 新しい期待 |
|---|---|---|
| `test_fedsda_final_configuration_run_settings.py`の、既定が「旧の固定設定＋列挙した差」と一致するtest | 項目なし | 列挙した差に、新しい項目を足す |
| 掃引のrunの識別子の期待値 | 今の既定の設定 | 新しい値へ |
| 参加者の設定を組み立てるtest（`test_fedsda_stream_protocol_run.py`ほか） | 項目なし | 旧の扱いを明示 |

実旧と比べるtestは、期待値を変えない（失敗したら、旧と比べる入口の明示の漏れを直す）。

## 旧と違う点
既定を使う実行（掃引を含む）のFedSDAの構成で、クロス評価が起きるラウンドの通信量（下りのbyte、モデルの転送回数、メッセージの数）が変わる。精度・学習・帰属・統合・パラメータは変わらない。2026-10-20までに保存した結果の通信量は、旧の扱い（クロス評価のために候補のモデルを送る）で記録されている。

## Validation Design
- 通信量の記録: 小さい条件（クロス評価と統合が起きる条件）で、2つの値の全体runを比べ、通信量以外の全部の記録・指標・パラメータが同じで、下りのbyteが、クロス評価の下りを引いて代表の再送を足した値（旧の扱いの記録を、ラウンドごとに分けて、新しい扱いの式で計算し直した値）と一致すること。クロス評価をしないラウンドの記録が同じこと。
- 共有部の送り直し: 部品のtestで、統合の後の共有部が先に送った値とビット単位で同じ場合は共有部を数えず、違う場合（testで、統合の後の最初のモデルの共有部の値を変えた場合）は共有部×client数を数えること。全体runで、共有部を送り直したラウンドの数も記録して確かめる（0でも1以上でもよいが、数えた値が式と一致すること）。
- 設定: 新しい項目の検査（不正な値の拒否）。既定の値。全構成の組立てと小さい実行。
- 旧と比べる入口: 既存の実旧との照合（期待値は変えない）。
- 基準版: 版1〜7の照合（値を変えずに）、版8の照合、版7→8の差の説明。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜8と、このspecのtest）。

## File Structure Plan
- `src/federated_learning_experiments/runtime/`: `fedsda_run_participant_factory.py`（設定と受け渡し）、`fedsda_run_server.py`、`server_round_synchronization.py`、`model_cross_evaluation.py`、`global_model_distribution.py`、`fedsda_final_configuration_run_settings.py`、`core/run_provenance.py`。
- `tests/refactoring/`: 新規`test_fedsda_cross_evaluation_with_distributed_models.py`、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v8/`、`test_fedsda_run_client.py`（旧と比べる入口）、`test_run_settings_validation.py`（設定の組）、`test_fedsda_stream_protocol_run.py`、`fresh_process_smoke.py`、既定に依るtest。
- 文書: `docs/experiments/experiment-sweep.md`、`docs/experiments/metrics.md`、`docs/experiments/refactoring-baseline.md`、`docs/overview/proposed-method.md`、`docs/research/improvement-candidates/improve-020-reduce-cross-evaluation-communication.md`、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.5 | 新しい項目、通信量の記録の分岐、通信量の記録のtest |
| 2.1〜2.2 | 最終構成の既定、構成のtest |
| 3.1 | 旧と比べる入口と設定の組の明示、実旧との照合の既存test |
| 3.2 | 版1〜7の明示と照合 |
| 4.1〜4.2 | 版8、版7→8の差の説明のtest |
| 5.1 | 文書の更新 |
