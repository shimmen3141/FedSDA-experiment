# 設定基盤の正式命名契約

- revision: 12（2026-10-03、正式契約のrevision 11を維持し、Lunaレビュー済みの境界検証用内部名を追加）
- 正本: この文書。旧名・候補は[naming-reconsideration.md](naming-reconsideration.md)、過去の契約は[history/](history/naming-revision-10.md)。
- 名前変更でアルゴリズム・値域・実行順序は変更しない。旧名alias・互換importは作らない。
- パッケージ: `federated_learning_experiments`。初回は設定部分だけを実装する。

## 基礎型・集約・検証

- `ExperimentRunConditions`: dataset・seed・client数・stream長・集約間隔。
- `ModelArchitectureSettings`: モデル構造・要求rank。
- `LocalTrainingSettings`: ローカル学習の固定条件。
- `ModelConsolidationSettings`: モデル対比較・クラスタ候補・統合方針。
- `RunSettingsValidationError`: `configuration_parameter_name`、`specified_parameter_value`、`validation_failure_reason`を保持するValueError。
- `validate_experiment_run_settings(unvalidated_run_settings)`: 順序付き正式キー→型のmappingと機能間条件を検証し、成功ならNone。入力更新なし。
- `ResolvedExperimentRunSettings`、方式カタログ、preset、保存APIはdesign.mdの後続契約。初回では作らない。

## 改名の確定範囲と意味

初回の損失監視は`LossChangeDetectionSettings`、予測は`PredictionCombinationSettings`、
学習データ帰属は`TrainingDataAssignmentSettings`、候補は`CandidateModelTrainingAndAcceptanceSettings`。
前3型はtask 9.1で既存実装を改名する。候補型はtask 2.7で作成する。
各型・モジュールは下表の配置と一致させ、再export窓口を作らない。

- Fixed-Shareの再配分時間尺度Hは毎標本の再配分率1/Hを定める。最終構成では保留FIFO容量と同値。
- `restart_adahedge_preserve_fixed_share_prediction_state`: 帰属先変更時に予測用AdaHedge、予測クラス別AdaHedge、
  比較用の診断AdaHedge、有効な推論対象集合の状態を再始動する。モデル直接のFixed-Share、上位Fixed-Share、
  真の概念別診断の状態は保持する。重みだけでなく累積分散も保持する。
  根拠は旧`clients/fedsda.py::_RestartingSoftRoutingFedSDAClientMixin._on_local_model_change`。
  サーバ集約後の再較正とモデル集合変更時の更新は別イベントである。今回は状態機械を実装しない。
- `recompute_buffer_losses_and_replay_weight_updates`: 集約後モデルで保留標本の損失を再計算し、
  重み更新の証拠を再構築する。空バッファ・単一モデル時等の旧動作は後続の移植契約で保全する。
- 候補設定は候補学習・既存モデル適合確認と再利用・比較・採否を説明する固定条件で、今回は計算を実装しない。
- `classwise_unique_correctness_lower_confidence_bound`: 利用可能クラス・左右方向の固有正解率の
  Bonferroni補正Wilson下限の最大。標本数条件を満たすクラスがゼロなら全体集計へfallbackする。
- 単位・必須入力・不変性・値域は従来契約を維持する。学習のmeanはサンプル数で加重する意味を正式値へ含める。

## 1. 型・配置・正式フィールド

初回の内部集約型は`ValidatedExperimentRunSettingsSubset`とする。
各値と組合せを検証した実験設定の一部を表し、完全な実行・保存用の`ResolvedExperimentRunSettings`と区別する。
型名を変えるだけのaliasや自動昇格は作らない。両型の直接構築時にも集約の組合せ検証を行う。

### ファイルと型

全て`src/federated_learning_experiments/`以下。各型の初回フィールドは次表に示す。

| 新しい配置名 | 型・役割 |
|---|---|
| `learning/models/model_architecture_settings.py` | 承認済み`ModelArchitectureSettings`の所属先 |
| `learning/training/local_training_settings.py` | 承認済み`LocalTrainingSettings`の所属先 |
| `methods/fedsda/loss_change_detection/loss_change_detection_settings.py` | 承認済み`LossChangeDetectionSettings`の所属先 |
| `methods/fedsda/prediction_combination/prediction_combination_settings.py` | 承認済み`PredictionCombinationSettings`の所属先 |
| `methods/fedsda/training_data_assignment/training_data_assignment_settings.py` | **新規`TrainingDataAssignmentSettings`**。帰属保留容量の固定条件。バッファ内容・割当先IDは持たない |
| `methods/fedsda/candidate_model_selection/candidate_model_training_and_acceptance_settings.py` | 承認済み`CandidateModelTrainingAndAcceptanceSettings`の所属先 |
| `methods/fedsda/consolidation/model_consolidation_settings.py` | 承認済み`ModelConsolidationSettings`の所属先 |
| `core/configuration_errors.py` | 承認済み例外型の所属先。機能固有の設定型や外側の検証処理をimportしない |

初回テストは`tests/refactoring/test_run_settings_validation.py`、後続は
`test_component_option_definitions.py`、`test_run_settings_resolution.py`、`test_run_settings_serialization.py`。
テスト関数・fixture名と内部変数は担当task開始前に追記する。

### 集約型の公開フィールド

以下は初回の`ValidatedExperimentRunSettingsSubset`が持つ固定条件。実行状態へ参照を持たない。
完全型は後続で必要な全run条件を追加して確定する。初回の部分型は実行・保存APIへ渡さない。

| 提案名 | 型 | 役割 |
|---|---|---|
| `method_name` | 正式名 | 実験プロトコルの選択。個別機能の方式やpreset名とは別 |
| `experiment_run_conditions` | `ExperimentRunConditions` | dataset・seed・規模・集約間隔 |
| `model_architecture_settings` | `ModelArchitectureSettings` | モデルの構造条件 |
| `loss_change_detection_settings` | `LossChangeDetectionSettings` | 損失監視の方式と判定条件 |
| `prediction_combination_settings` | `PredictionCombinationSettings` | 予測混合の方式と状態管理方針 |
| `local_training_settings` | `LocalTrainingSettings` | ローカル学習の方式 |
| `training_data_assignment_settings` | `TrainingDataAssignmentSettings` | 帰属保留に関する固定条件 |
| `candidate_model_training_and_acceptance_settings` | `CandidateModelTrainingAndAcceptanceSettings` | 候補の作成・採否の固定条件 |
| `model_consolidation_settings` | `ModelConsolidationSettings` | サーバ側のモデル統合条件 |

非適用の機能設定をどう表現するかは方式追加時に具体化する。初回は最終FedSDA構成の型だけを扱う。

### 機能固有の初回フィールド

既に承認した数値・方針名は維持し、所属先を明示する。

| 所属する型 | 新しい公開名 | 型・単位・役割 |
|---|---|---|
| `ModelArchitectureSettings` | `model_architecture_name` | 正式名。モデル構造の選択 |
| `LossChangeDetectionSettings` | `drift_detector_name` | 正式名。個々の検出器の計算方式 |
| 同上 | `loss_monitoring_scope` | 正式名。全体・正解クラス別等の監視対象。検出器の種類とは別 |
| `LocalTrainingSettings` | `local_model_parameter_update_strategy` | 正式名。共有部・Adapter・headのローカル更新方式 |
| 同上 | `shared_backbone_gradient_combination_strategy` | 正式名。概念別勾配の統合方法 |
| `CandidateModelTrainingAndAcceptanceSettings` | `candidate_model_acceptance_policy` | 正式名。候補と比較用モデルの学習・検証・採否方針 |
| `ModelConsolidationSettings` | `model_clustering_trigger_policy` | 正式名。クラスタリングを開始する条件 |
| 同上 | `model_pair_comparison_strategy` | 正式名。統合候補となるモデル対の比較方式 |
| 同上 | `model_clustering_linkage` | 正式名。対評価からクラスタを作るlinkage |
| 同上 | `model_consolidation_policy` | 正式名。クラスタ作成後の統合方針 |
| `PredictionCombinationSettings` | `fixed_share_weight_redistribution_time_scale_samples` | 2以上の整数・sample/client。Fixed-Shareの時間尺度。容量とは意味が異なるが、今回の最終構成ではFIFO容量から導出し同値を検証する |

既存名の所属は次のとおり。

- `ExperimentRunConditions`: `dataset_name`、`random_seed`、`client_count`、`per_client_sample_count`、`server_aggregation_interval_per_client_samples`。
- `ModelArchitectureSettings`: `residual_adapter_requested_rank`。実効rankはモデル構築後の記録へ置き、初回設定型に持たせない。
- `LossChangeDetectionSettings`: `e_sr_false_alarm_control_alpha`。
- `PredictionCombinationSettings`: `prediction_combination_strategy`、`prediction_mixture_activation_policy`、`prediction_weight_recalibration_after_aggregation_policy`、`prediction_state_reset_on_training_assignment_change_policy`。
- `TrainingDataAssignmentSettings`: `pending_assignment_buffer_capacity_samples`。
- `CandidateModelTrainingAndAcceptanceSettings`: `candidate_post_alarm_validation_sample_count`。

各型は不変で、上表の値を実行中に変更しない。型の標準初期化検査`__post_init__`は自身の値を検証し、不正なら`RunSettingsValidationError`を送出する。
集約型・組合せ検証と機能固有型の相互importは避け、値の検証は担当機能内、組合せの検証は外側で行う。
集約型の`__post_init__`は`validate_experiment_run_settings`を呼ぶ。検証モジュールは集約型をimport・生成せず、正式フィールド名から機能型へのmappingを検査する。
例外型の新しい公開属性`validation_failure_reason`は日本語の文字列で、不正な理由・許容条件を表す。
`configuration_parameter_name`と`specified_parameter_value`は承認済み名を用いる。

### 最終構成の正式値

以下は新形式だけの値。旧名を受理するaliasは作らない。旧名との対応は調査資料・テストだけで保持する。

| フィールド | 提案する正式値 | 意味 |
|---|---|---|
| `method_name` | `fedsda` | FedSDAプロトコル |
| `model_architecture_name` | `shared_backbone_residual_adapter` | 共有表現と概念固有の低rank残差補正 |
| `drift_detector_name` | `e_sr` | e-SR検出器。論文の既存表記に合わせる |
| `loss_monitoring_scope` | `overall_and_true_class_losses` | 全体と正解クラス別の損失を監視 |
| `prediction_combination_strategy` | `fixed_share_weighted_prediction` | Fixed-Shareで優勢モデルの変化を追跡する混合 |
| `prediction_mixture_activation_policy` | `always` | 常時混合 |
| `prediction_weight_recalibration_after_aggregation_policy` | `recompute_buffer_losses_and_replay_weight_updates` | 集約後モデルでFIFOを再評価して損失情報を再較正 |
| `prediction_state_reset_on_training_assignment_change_policy` | `restart_adahedge_preserve_fixed_share_prediction_state` | 予測側AdaHedge・文脈・shadow routerと有効なactive setをrestartし、Switching・Meta-switching・oracle概念別診断routerは保持。全routingをresetする意味ではない |
| `local_model_parameter_update_strategy` | `joint_backbone_adapter_and_head_training` | 概念別ストアの損失で共有部と概念固有のAdapter・headを共同学習。共有部だけの更新ではない |
| `shared_backbone_gradient_combination_strategy` | `sample_weighted_mean_per_concept_gradients` | 概念別勾配を平均 |
| `candidate_model_acceptance_policy` | `current_model_first_reuse_then_two_segment_candidate_validation` | 既存モデル再適合・現行優先・前半後半の優位確認。数値の判断式は変更しない |
| `model_clustering_trigger_policy` | `on_new_model_registration` | 新規モデルの登録を受けて統合候補を調べる |
| `model_pair_comparison_strategy` | `classwise_unique_correctness_lower_confidence_bound` | クラス別の機能的評価に基づく比較 |
| `model_clustering_linkage` | `average_linkage` | average linkage |
| `model_consolidation_policy` | `weighted_parameter_average_and_merge_ids` | クラスタ内モデルを統合 |

これらの値の承認はアルゴリズム変更の承認ではない。移植時には旧構成と判断・順序を照合する。
γ、候補のoptimizer、事前学習・データ生成・評価ストア等の詳細は初回に含めず、実行層の移植前に追加レビューする。

### 値域・単位の宣言

初回から各dataclassフィールドのmetadataに下記を宣言し、自身の検証と後続の項目カタログが同じ宣言を使う。
数値種別は型注釈、既定値はフィールド既定値を正本とする。必須入力は既定値なしとして説明する。

| metadataの提案名 | 型・役割 |
|---|---|
| `parameter_unit` | 文字列。sample/client等の単位 |
| `minimum_allowed_value` | 数値。省略なら下限なし |
| `maximum_allowed_value` | 数値。省略なら上限なし |
| `minimum_value_is_inclusive` | bool。指定した下限を含むか |
| `maximum_value_is_inclusive` | bool。指定した上限を含むか |

機能型は後続のcatalogへ依存しない。公開の説明取得APIはcatalog段階で提供する。

## 2. データセットの正式値

`dataset_name`で受理する初回の正式値を、最終goldenの3ケースに対応させる。
FedDriftのベンチマーク名との対応を優先し、既存識別子を新APIでも正式名として維持する。
末尾の数字は本実装では概念数を表す。概念数・クラス数は名前を伸ばさず、定義と説明で明記する。

| 正式値 | 図表・文書の表記 | 意味 |
|---|---|---|
| `sine2` | SINE-2 | SINE、2概念・2クラス |
| `sea2` | SEA-2 | SEA、2概念・2クラス |
| `mnist2` | MNIST-2 | MNIST、2概念・10クラス |
| `mnist4` | MNIST-4 | MNIST、4概念・10クラス。初回実装対象外 |

根拠は既存`data/specs.py`の`num_concepts`・`num_classes`と最終goldenのケース定義。
初回は正式値の検証だけを扱い、生成・読込み・スケジュール実装は次specが担当する。
初回はgoldenに対応する3件だけを受理する。`mnist4`を含む他の既存データセットも、後続で扱う際は識別子を維持する。
値は厳密一致で受理し、未知名・改名案の`sine_two_concepts`等・大小文字の自動変換を受理しない。
正式名の維持は旧aliasの追加ではない。原論文: https://www.microsoft.com/en-us/research/uploads/prod/2023/02/FedDrift_Camera_Ready-63feb83f92b63.pdf


## 3. 初回実装の内部関数・テスト名

以下は公開契約を実装・検証する内部名。レビューと採否の履歴は[luna-naming-review.md](luna-naming-review.md)に記録する。

### 値域検証の共通処理

各機能型の`__post_init__`から、標準dataclassの型注釈・metadataだけを読む共通検証を呼ぶ。
これによりbool拒否・有限性・境界の扱いを機能別に重複実装しない。
共通処理は機能型・集約型・方式カタログ・旧configをimportしない。組合せ検証も担当しない。
機能固有型は基礎例外とこの基礎層の値検証へ依存できる。

| 種類 | 名前・呼出し | 役割・入出力・状態 |
|---|---|---|
| ファイル | `core/settings_field_validation.py` | 型注釈・値域宣言に従う値検証。方式間の依存を知らない |
| 関数 | `validate_settings_field_values(settings_instance)` | 機能設定のdataclass → 成功時`None`、失敗時`RunSettingsValidationError`。入力更新なし |
| 引数 | `settings_instance` | 自分のフィールドを検証する不変設定の実体。全run条件とは限らない |
| metadata | `allowed_parameter_values` | 正式な文字列値のtuple。型の宣言に所属し、未知値・自動変換を拒否する |
| 局所変数 | `settings_field`, `settings_field_metadata`, `settings_field_type` | 現在検証するdataclassフィールド、その制約、その型注釈 |
| 局所変数 | `minimum_allowed_value`, `maximum_allowed_value` | metadataから取得した境界。未指定時は`None` |
| 局所変数 | `allowed_parameter_values` | metadataから取得した受理可能な正式値 |
| 局所変数 | `configuration_parameter_name`, `specified_parameter_value`, `validation_failure_reason` | 承認済みの例外属性と同じ意味。項目名・指定値・日本語の不正理由 |
| 検証宣言 | `initial_component_settings_types` | 初回の集約フィールド名と期待する機能型の順序付きtuple。初回の検証順を定める。実体生成・方式カタログではない |
| 局所変数 | `component_settings_name`, `component_settings_type`, `component_settings` | 機能設定の集約キー、期待型、入力された機能設定値 |
| 局所変数 | `unknown_settings_names` | 未登録の集約キー。入力を変更せず、再現可能な順序で報告する |

標準特殊メソッドの`self`・`__init__`・`__post_init__`はPythonの慣用名を使う。
例外の引数は承認済み公開属性と同名とする。新しい可変の実行状態は導入しない。

### テスト・fixture

全て`tests/refactoring/test_run_settings_validation.py`へ置く。新APIの契約を検証し、旧実装を呼んで新設定を検証しない。

| 名前 | 検証する契約・役割 |
|---|---|
| `valid_run_settings_mapping` | 初回の有効な機能型を揃えたmappingをテストごとに作るfixture。完全な実験条件・presetとは扱わない |
| `test_validation_error_preserves_parameter_details` | 共通例外の項目・値・日本語理由を保持する |
| `test_existing_dataset_identifiers_are_accepted` | 初回3件の既存正式名を受理する |
| `test_unknown_dataset_identifiers_are_rejected` | 未対応名・撤回した改名案・大小文字の違いを拒否する |
| `test_integer_settings_reject_invalid_types_and_ranges` | bool・文字列・実数・範囲外の整数を拒否する |
| `test_integer_settings_validate_boundary_values` | seed=0などの有効な境界を受理し、正整数の0や下限2の項目の1を拒否する |
| `test_aggregation_interval_can_exceed_stream_length` | 集約間隔がstream長を超える条件を受理する |
| `test_requested_adapter_rank_is_preserved` | 要求rankを設定段階で丸めない |
| `test_monitoring_alpha_rejects_nonfinite_and_out_of_range_values` | alphaの有限性と開区間を検証する |
| `test_component_options_reject_unknown_names` | 正式値以外の方式を拒否し、旧名aliasを持たない |
| `test_candidate_post_alarm_validation_accepts_odd_sample_counts` | 候補の将来標本による検証件数は2以上の奇数も許容する |
| `test_settings_instances_are_immutable` | 構築後の条件変更を拒否する |
| `test_field_annotations_and_metadata_declare_parameter_constraints` | 型注釈・単位・値域が同じフィールド宣言に存在する |
| `test_run_settings_validation_accepts_valid_component_mapping` | 有効な初回組合せで成功し、入力を変更しない |
| `test_run_settings_validation_rejects_invalid_component_mapping` | 未知キー・欠落・型違い・非対応手法を拒否する |
| `test_fixed_share_time_scale_must_match_assignment_buffer_capacity` | 個別値が有効でも容量と時間尺度の不一致を拒否する |
| `test_validation_reports_failures_in_declaration_order` | 複数の不正がある場合の最初のエラーを固定する |
| `test_direct_subset_construction_uses_the_same_validation` | 集約型の直接構築も共通の組合せ検証を通す |
| `test_subset_does_not_share_mutable_input_mapping` | 元mapping変更が構築済み部分型へ波及しない |
| `test_new_package_imports_only_allowed_dependencies` | 設定基盤が標準ライブラリと新設定型だけへ依存する |

テストの引数・一時値は`settings_type`（対象型）、`valid_settings_values`（有効なフィールド値）、
`configuration_parameter_name`（対象項目）、`specified_parameter_value`（試す入力）、
`expected_failure_reason`（期待する不正理由）、`validation_error`（捕捉した例外）、
`unvalidated_run_settings`（検証前mapping）、`validated_settings_subset`（構築した部分型）、
`settings_instance`（単独機能設定）を用いる。

## 4. 実験条件型の分離

| 配置 | 名前・役割 |
|---|---|
| `configuration/experiment_run_conditions.py` | 承認済み`ExperimentRunConditions`の所属先。dataset・seed・規模・集約間隔の固定条件だけを持つ |
| `configuration/run_settings.py` | 承認済み`ValidatedExperimentRunSettingsSubset`の所属先。実験条件型と機能型を束ね、外側の組合せ検証を呼ぶ |

実験条件型と集約型を同じモジュールへ置くと、検証モジュールが実験条件型を読む際に集約型までimportし、循環参照になる。
実験条件型を分け、検証モジュールは同型と機能型だけをimportする。公開型・項目・検証契約は変更しない。
gpt-6-lunaのレビューで配置名と分離を妥当と確認し、次の依存注記を反映して承認した。
`ExperimentRunConditions.__post_init__`はcoreの共通フィールド検証だけを呼び、外側の集約型・組合せ検証へ依存しない。
`run_settings.py`を実験条件型の再export窓口にはしない。

## 5. 共通値検証の直接テスト

| 名前 | 役割 |
|---|---|
| `test_settings_field_validation_accepts_supported_numeric_values` | 整数・実数項目の許容入力、有限性と開閉境界の受理を検証する |
| `test_settings_field_validation_rejects_invalid_values` | bool・型違い・非有限値・値域違反を拒否する |
| `test_settings_field_validation_rejects_unsupported_annotations` | 未対応の型注釈を黙って検証対象外にしないことを確認する |
| `NumericSettingsForValidation` | 数値宣言だけを持つテスト専用dataclass。実験用の正式型ではない |
| `UnsupportedSettingsForValidation` | 未対応の型注釈を検査するテスト専用dataclass |

gpt-6-lunaは名前と役割が一致し、変更必須の指摘なしと判断した。主担当も維持を採用し、ユーザー委任に従い承認した。

## 6. import境界の検証で使う内部名

task 3.3の`test_new_package_imports_only_allowed_dependencies`で用いる。公開API・設定値は変更しない。

| 名前 | 型・役割 |
|---|---|
| `package_source_directory` | Path。検査する新パッケージのsrcルート |
| `source_file_path` | Path。検査するPythonファイル |
| `source_module_path` | str。パッケージルートからの相対ファイルパス（POSIX表記） |
| `parsed_source_module` | ast.Module。構文解析したソース |
| `import_statement` | ast.Importまたはast.ImportFrom。検査するimport文 |
| `importing_package_name` | str。当該ファイルの所属先の完全修飾package名。__init__.pyは自身のpackage、それ以外は親package |
| `imported_module_alias` | ast.alias。ast.Importの一つの宣言alias |
| `imported_module_names` | tuple[str, ...]。一つのimport文が参照する、解決済みの完全修飾module名の一覧 |
| `imported_module_name` | str。上記一覧から検査する一つのmodule名 |

gpt-6-lunaの指摘を採用し、相対ImportFromはlevelとmoduleを用いて所属packageから解決する。
Importは各alias.nameを別々に検査する。相対importの禁止を新しい設計制約として追加しない。
追加名の再レビューはPASS。主担当も有用性を確認し、ユーザー委任に従って承認した（revision 12）。
