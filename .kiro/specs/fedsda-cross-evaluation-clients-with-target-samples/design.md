# Design Document

## Overview
サーバのクロス評価（`cross_evaluate_global_models`、`runtime/model_cross_evaluation.py`）で、対象のモデルを評価させるclientの候補を、その対象の標本を返せるclientに限る扱いを足し、既定にする。研究ツールの`_sample_holding_cross_evaluation_clients_policy`（`FedsdaRunClient.get_cross_evaluation_held_model_ids`を包み、標本を返せるIDとの共通部分を返す）を、製品の部品として書き直す。

| 設定 | これまで | 新しい既定 |
|---|---|---|
| クロス評価で評価させるclientの選び方（新しい項目`FedsdaRunParticipantSettings.cross_evaluation_client_selection_policy`） | （項目なし。`clients_holding_target_model`に当たる） | `clients_with_target_model_samples` |

## Boundary Commitments
### This Spec Owns
参加者の設定の新しい項目と検査、サーバ・同期・クロス評価への受け渡し、評価させるclientの候補の作り方の分岐、clientの「標本を返せるか」の判定（標本の選び方と共有）、最終構成の既定、候補検証ありの構成の明示、旧と比べる設定の入口、基準版のhelper（版1〜11の明示、版12、経路の記録の新しい項目）、既定に依るtest、研究ツールの基点の固定、文書。
### Out of Boundary
クロス評価の評価そのもの（標本の選び方の規則、件数の上限、損失と正誤の集計）、クラスタリング、統合、クラスタリングを行う同期の規則、通信量の数え方、FedDrift・FedAvg、固定旧、研究ツールの拒否権の変種。
### Allowed Dependencies
新しいmoduleは無い。依存境界の許可へ、新しくimportする名前を登録する。
### Revalidation Triggers
クロス評価を通る全test、既定のbuilderを通るtest、掃引のrunの識別子、基準版のhelper、研究ツールの既存の変種。

## Architecture
### 設定と受け渡し（Requirement 1.1、2.1）
- 値の名前・組・検査の関数を`model_cross_evaluation.py`に置く: `CLIENTS_HOLDING_TARGET_MODEL_POLICY_NAME = "clients_holding_target_model"`、`CLIENTS_WITH_TARGET_MODEL_SAMPLES_POLICY_NAME = "clients_with_target_model_samples"`、`CROSS_EVALUATION_CLIENT_SELECTION_POLICY_NAMES`、`validate_cross_evaluation_client_selection_policy`（builtin strで、2つの値のどちらか。版8の`cross_evaluation_model_transfer_policy`、版9の`consolidated_shared_feature_policy`と同じ形）。
- `FedsdaRunParticipantSettings`に必須の項目`cross_evaluation_client_selection_policy`を足し、`__post_init__`で検査する。`assemble_fedsda_run_server`・`FedsdaRunServer`（必須）、`synchronize_models_in_server_round`・`cross_evaluate_global_models`（既存の呼出しを変えないよう、今の扱いを既定値に持つ）へ渡す。`synchronize_models_in_server_round`は、どの段より前に検査する（今の2つの方針の検査の隣）。

### 標本を返せるかの判定（Requirement 1.3）
- `client_model_cross_evaluation.py`で、今の`evaluate_candidate_model_on_target_model_samples`の中の標本の選び方（評価標本が6件以上ならそれ、足りなければ対象が現在の学習帰属のときだけ、学習データが11件以上ならそれ、どちらでもなければ空）を、非公開の関数`_select_target_model_samples`へ切り出し、評価はこれを使う（上限での抜き出しは今と同じく評価の側で行う。乱数の消費は変わらない）。
- 公開の関数`target_model_cross_evaluation_samples_are_available`（引数は対象のID、評価標本のowner、学習データのowner、現在の学習帰属）は、`_select_target_model_samples`の結果が空でないかを返す。状態を変えず、乱数を使わない。
- `FedsdaRunClient.get_model_ids_with_cross_evaluation_samples()`は、保有モデル（`held_model_training_state_registry`）のIDのうち、上の関数が真になるIDのfrozensetを返す。研究ツールは標本の条件だけを使う（`get_cross_evaluation_held_model_ids()`との共通部分）が、製品は、正誤の比較に対象のモデルが要るので、保有モデルであることも条件にする（候補が必ず評価を行い、0件の記録が起きない）。2つが違うのは、標本を持つが対象のモデルを保有しないclientがあるときだけで、最終構成では配布で全clientが全グローバルモデルを保有するので起きない見込み。研究ツールとの照合（Requirement 5.1）で、候補の選び方が同じことを確かめる。

### 評価させるclientの候補（Requirement 1.2、1.4、1.5）
- `cross_evaluate_global_models`の`holding_clients_by_model_id`を作るところ（`run_clients`の順に、各clientのIDの集合を見て、IDごとのclientの列へ足す）で、今の扱いは`get_cross_evaluation_held_model_ids()`、新しい扱いは`get_model_ids_with_cross_evaluation_samples()`を使う。列の順は、どちらも`run_clients`の順。それ以降（上限を超えたときだけ、その列から`python_random_generator.sample`、メッセージの記録、評価、集計）は同じ。候補が無い対象のモデルは、評価させるclientが0（今の扱いでも、保有するclientが無い対象は同じ）。
- 新しい扱いでは、選んだclientは対象のモデルを保有し、5件以上の標本を選ぶ（下限6・11 ≥ 評価に要る5）ので、必ず評価を行い、0件の記録は起きない（Requirement 1.5）。部品のtestと、基準版12のgoldenの必須の経路で確かめる。

### 最終構成と候補検証ありの構成（Requirement 2.1）
- 最終構成の既定（`fedsda_final_configuration_run_settings.py`）を`clients_with_target_model_samples`にする。候補検証ありの構成（`sweep/experiment_configurations.py`の`fedsda-with-candidate-validation`）は、`clients_holding_target_model`を明示する（版11と同じく、候補検証との組合せは未検証のため）。ほかの構成は既定を引き継ぐ。

### 旧と比べる設定の入口（Requirement 3.1）
- testの旧と比べる設定の入口（`use_legacy_compatible_policies`）と、実旧との照合に使う設定の組（`test_fedsda_stream_protocol_run.py`の組立て、fresh script、`test_fedsda_run_client.py`・`test_fedsda_run_server.py`の組立て）は、今の扱いを明示する。

### 基準版（Requirement 3.2、4）
- 版1〜11: 参加者の設定の「後に足された項目」の表へ、`cross_evaluation_client_selection_policy`＝`clients_holding_target_model`を、版1〜11とも足す。版1〜11のgoldenの全ケースの設定は、`build_case_settings`が、この表の値で新しい項目を置き換えて組み立てる（保存してある設定と比べるときは、組み立てた値がこの値であることを確かめてから除く。版9の`consolidated_shared_feature_policy`と同じ経路）。照合は、既存の版ごとの照合のtest（版ごとの全ケース）を、そのまま全部の版で通す。
- 経路の記録に、版12から`cross_evaluation_record_count`（記録の表`client_cross_evaluation.evaluated_sample_count`の行の数）と`zero_sample_cross_evaluation_count`（そのうち0の行の数）を足す（FedSDAのケースだけ）。
- 版12を足す（ケースは版11と同じ22）。必須の経路: 版12のFedSDAのケースのうち、候補検証ありの構成で実行する4ケース（`CANDIDATE_VALIDATION_PATH_CASE_NAMES`: `fedsda-sine2-events`、`fedsda-mnist2-decisions`、`fedsda-sine2-reference-reuse`、`fedsda-sine2-alternative-reference-reuse`）を除く全部で、`zero_sample_cross_evaluation_count`が0（記録の行が0のケースも0）。ケースごとの必須の経路の関数（`assert_required_paths`）で確かめる。加えて、統合が起きる`fedsda-sine2-adoption-without-validation`と`fedsda-mnist4-warm-started-learner`では、`cross_evaluation_record_count`が1以上であることも必須の経路にする（比べ方が空でない）。版11までの必須の経路は、版12でも同じに求める。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を12へ上げるのは、goldenを置いた後。
- 版11→12の差の説明のtest: 版11と版12を実行し（goldenを使わず、その場で）、FedDrift・FedAvgのケースと、候補検証ありの構成のケースは、`run_metrics`・`record_tables`・`global_parameters`・`coverage`（版12で足した項目を除く）が同じ。新しい既定を引き継ぐFedSDAのケースは、`run_settings`が新しい項目だけ違う。版11の実行で、標本が0件の評価が起きたFedSDAのケースが少なくとも1つある（記録の表から数える）。

### 研究ツール（Requirement 5）
- 照合（既定を変える前に）: 研究ツールの`product-default-sample-holding-cross-evaluation`と、製品の既定（基準版11）に新しい項目だけを`clients_with_target_model_samples`にした全体runを、MNIST2と、SINE2・CIRCLE2・SEA4（各1 seed。クロス評価と統合が起き、評価の候補が上限3を超える同期を含む条件。例: 各1500〜2500標本・10 client・間隔50）で比べる。精度、通信量（`communication_volume`の下り・上りのバイト数と、下り・上りのメッセージの数。クロス評価の依頼・返信のメッセージは評価させたclientの数で数えるので、候補の選び方の違いを捉える）、予測の記録の要約（`prediction_record_digest`）、最終のパラメータの要約（`server_parameter_digest`）、計算回数の指標（`model_computation_counts`、`preparation_model_computation_counts`）が一致することを確かめる。候補の選び方が違えば、メッセージの数か乱数の消費（その後の軌跡）が変わり、要約が一致しない。ただし、上限以下では候補の列の違いが指標に出ないことがあるので、候補の列そのものも直接比べる: 両方のrunで、クロス評価の呼出しごと・対象のモデルごとに、(a) 上限を当てる前の候補のclientの列と、(b) 実際に評価させたclientの列（記録の表`client_cross_evaluation`のclientの列）を記録し、全部の同期で一致することを確かめる。加えて、製品のrunで、クロス評価の呼出しごとに、標本の条件を満たすが対象のモデルを保有しない（client、対象）の組の数を数え、全部の同期で0であること（2つの候補の条件が同じ候補を作る前提）を確かめる（Git管理外の一時script。結果はtasks.mdに記録する）。
- 基点の固定: `_execute_job`の、`product-default`系以外の変種を基準版10の既定へ戻す置き換えに、`cross_evaluation_client_selection_policy="clients_holding_target_model"`を足す（研究ツールの評価のclientを絞る変種は、今の扱いの経路——`get_cross_evaluation_held_model_ids`——を包むので、そのまま働く）。既定を変えた後、製品の既定は`get_model_ids_with_cross_evaluation_samples`で同じ候補の集合を作り、包みの対象の`get_cross_evaluation_held_model_ids`をクロス評価で呼ばないので、`product-default-sample-holding-cross-evaluation`は`product-default`と同じ選び方になる。これを変種の説明に書く。
- 固定の確認: 既定を替えた後、研究ツールの`default`と`default-sample-holding-cross-evaluation`の小さい条件の結果が、替える前と同じことを確かめる（`product-default`系以外の意味が変わらない）。

## 旧と違う点
既定を使う実行のFedSDAの構成で、クロス評価を行う同期のうち、標本を持たないclientが候補にあった対象のモデルでは、評価させるclientと、上限を超えたときの無作為の選択（乱数の消費）が変わる。登録の後の同期でも対の判定の値を持てるので、統合が増え、サーバが持つモデルの数・下りの通信・clientの学習の計算が減ることがある。クロス評価のメッセージの数は、評価させたclientの数で数えるので、0件を返すclientの分だけ減る。

## Validation Design
testは、重要なもの・後で効くもの・複雑なものに絞る（ユーザー指示）。
- 部品（重要）: 実際のclient（既存のtestの組立てを使う）を、標本を返せるclientと返せないclientが混ざる状態にして、`cross_evaluate_global_models`を新旧の扱いで実行し、記録（`client_cross_evaluation`のclientの列）で次を確かめる。(i) 新しい扱いでは、対象ごとの評価したclientが、標本を返せるclientの、clientの順の列（上限以下のとき）と同じ。(ii) 上限を超えるときは、同じseedの`random.Random`で、その列から`sample`した結果と同じ（上限以下のときは乱数を消費しない: 実行の後の乱数の状態が、上限を超えた対象の数だけ`sample`したときと同じ）。(iii) 標本を返せるclientが無い対象は、評価したclientが0でメッセージも0。(iv) 新しい扱いでは0件の記録が無く、今の扱いでは同じ状態で0件の記録がある（比べ方が空でない）。(v) 対象のモデルの評価標本を持つが、対象のモデルを保有しないclient（保有モデルの登録簿から対象を外した状態）は、新しい扱いの候補に入らない。`target_model_cross_evaluation_samples_are_available`と、評価が実際に選ぶ標本が、評価標本（6件の境界）・学習データ（11件の境界）・現在の学習帰属の組合せで食い違わないこと（共有の規則の守り）。
- 設定の検査（後で効く）: 不正な値・型の拒否を、参加者の設定と同期の入口で1つずつ。
- 全体run（複雑）: 研究ツールとの照合（一時script）、基準版11→12の差の説明と版12の必須の経路（goldenのtest）。
- 既存の照合: 基準版1〜11、実旧との照合、全pytest、Ruff、Pyright、WSL（goldenの版1〜12と、このspecのtest）。

## File Structure Plan
- `src/federated_learning_experiments/runtime/`: `model_cross_evaluation.py`、`client_model_cross_evaluation.py`、`fedsda_run_client.py`、`server_round_synchronization.py`、`fedsda_run_server.py`、`fedsda_run_participant_factory.py`、`fedsda_final_configuration_run_settings.py`。`sweep/experiment_configurations.py`、`core/run_provenance.py`。
- `tests/refactoring/`: 新規`test_fedsda_cross_evaluation_clients_with_target_samples.py`、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v12/`、既存の組立て（`test_fedsda_stream_protocol_run.py`、`test_fedsda_run_client.py`、`test_fedsda_run_server.py`、`fresh_process_smoke.py`）、依存境界、既定に依るtest。
- `tools/research_provisional_candidate_expert/run.py`（基点の固定、変種の説明）。
- 文書: `docs/overview/proposed-method.md`、`docs/experiments/refactoring-baseline.md`、`docs/experiments/experiment-sweep.md`、LEGACY-022・IMPROVE-028、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.5 | 新しい項目、判定の共有、候補の作り方の分岐、部品のtest、版12の必須の経路 |
| 2.1 | 既定、候補検証ありの構成の明示、構成のtest |
| 3.1〜3.2 | 旧と比べる入口、版1〜11の明示と照合 |
| 4.1〜4.3 | 版12、版11→12の差の説明、0件の評価の数の経路 |
| 5.1〜5.2 | 研究ツールとの照合、基点の固定と確認 |
| 6.1 | 文書 |
