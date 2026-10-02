# 命名・役割レビュー: 設定・選択肢管理の基盤

- revision: 8
- パッケージ名: **`federated_learning_experiments`は2026-10-02に人間が承認済み。**
- revision 2の名前・役割: **2026-10-02に人間が承認済み。承認hashはspec.jsonの履歴へ保存。**
- revision 4: **2026-10-03に設計時の追加案とレビューによる修正を人間が承認済み。実装は未開始。**
- revision 5: **2026-10-03に既存データセット名を維持する修正方針と初回taskの実装開始を人間が承認済み。**
- revision 6: **2026-10-03にgpt-6-lunaのレビューを受け、有用な指摘を反映して承認済み。ユーザーの明示的な委任に従う。**
- revision 7: **実験条件型の配置分離をgpt-6-lunaがレビューし、依存方向の注記を反映して承認済み。**
- revision 8: **共通値検証の直接テスト3件とテスト専用型をgpt-6-lunaがレビューし、変更不要と判断して承認済み。**
- 方針の正本: [リファクタリング方針](../../../docs/research/refactoring-policy.md)
- 初回は最終構成に必要な設定と検証を扱う。下表の「後続」は、その機能の移植時に必要性を判断する候補である。

## 1. 名前を判断する基準

1. 短さより、名前から対象・役割を特定できることを優先する。必要なら4語・5語以上を使う。
2. 対象、単位、時点、状態の違いを名前へ含め、似た名前との混同を防ぐ。
3. `Settings`は個々の実験条件、`Definition`は使用可能な項目の説明・制約、`Catalog`は定義の集合を表す。
4. 入力の変更指定、未検証の設定、検証済み設定、実行中の可変状態を区別する。
5. 未確認の意味を名前で断定しない。既存コードの役割を確認してから正式名を決める。

## 2. パッケージと配置

`federated_learning_experiments`は「連合学習の実験」を表し、複数手法の比較・評価という用途を含める。
FedSDA等の手法名は`methods/`以下、構成名はpresetへ置く。研究手法名の改名は別の判断である。

以下はファイル名候補であり、全ファイルを最初に作る指示ではない。
機能固有の設定型は担当機能の近くへ置き、配置の確定は設計レビューで行う。
手法の計算部から汎用の選択肢一覧・設定解決・CLI・ファイルI/Oを参照しない。

| 配置案（`src/federated_learning_experiments/`以下） | 役割 | 着手範囲 |
|---|---|---|
| `configuration/run_settings.py` | 一実験の確定条件を束ねる型。各機能の詳細設定をここへ集中させない | 初回 |
| `configuration/run_settings_validation.py` | 最終構成に必要な値・組合せを検証 | 初回 |
| `configuration/component_option_definitions.py` | 機能ごとの方式と、使用するパラメータの定義 | 後続 |
| `configuration/experiment_presets.py` | 最終提案・比較構成の設定値の組合せ | 後続 |
| `configuration/run_settings_resolution.py` | presetと明示変更から実効設定を生成 | 後続 |
| `configuration/run_settings_serialization.py` | 実効設定とversion付き辞書の相互変換。ファイルI/Oなし | 後続 |

## 3. 型と役割

| revision 1 | 改善案 | 役割・区別 | 着手範囲 |
|---|---|---|---|
| `RunSettings` | `ResolvedExperimentRunSettings` | 一回の実験の全条件の解決・検証が完了した不変設定。生の入力や実行状態を持たない | 全run条件が揃った後 |
| `ExperimentSettings` | `ExperimentRunConditions` | dataset、seed、規模、同期間隔。手法の内部方式とは区別 | 初回 |
| `ModelSettings` | `ModelArchitectureSettings` | モデル構造と構造上の指定。学習済み重みやoptimizerは含まない | 初回 |
| `DetectionSettings` | `DriftMonitoringSettings` | ドリフト検出方式、損失の監視範囲、判定設定。検出器の実行状態とは別 | 初回 |
| `RoutingSettings` | `PredictionRoutingSettings` | 予測の混合方式、有効期間、再較正、割当変更時reset。データ割当とは別 | 初回 |
| `TrainingSettings` | `LocalTrainingSettings` | ローカル更新方式、勾配統合、学習率等。進行中のoptimizerやbatchを持たない | 初回 |
| `ModelCreationSettings` | `CandidateModelCreationSettings` | 候補の学習・比較・前向き検証・採否条件。単純なモデル生成とは別 | 初回 |
| `ConsolidationSettings` | `ModelConsolidationSettings` | モデル対評価、linkage、統合方針。クラスタやID対応表は実行結果 | 初回 |
| `ParameterDefinition` | `ConfigurationParameterDefinition` | 設定項目の名前・型・単位・範囲・既定値。個々のrunの指定値とは別 | 後続 |
| `ChoiceDefinition` | `ComponentOptionDefinition` | 一機能で選べる方式の名前・説明・能力・専用項目・制約。実装や生成関数を持たない | 後続 |
| `ChoiceCatalog` | `ComponentOptionCatalog` | 機能ごとの方式定義の読取り専用集合。実行インスタンスは含まない | 後続 |
| `PresetDefinition` | `ExperimentPresetDefinition` | 名前付きの実験設定の組合せ。一機能の方式を指すoptionとは別 | 後続 |
| `ConfigurationError` | `RunSettingsValidationError` | 実験設定の不正な項目・値・原因。未知のpreset等の入力誤りも含む | 初回 |
| 同上に混在 | `ConfigurationDefinitionError` | 定義側の重複・参照漏れ・既定値不整合。研究者の入力誤りと区別 | 後続 |

初回の型も、全方式向けのフィールドや階層を一度に作らない。最終構成で必要なものから具体化する。
方式固有の値は担当機能の設定型へ所属させる。詳細フィールドは実装前に追加してレビューする。

## 4. 関数・メソッド

| revision 1 | 改善案 | 主な入力 → 出力 | 役割・状態更新 | 着手範囲 |
|---|---|---|---|---|
| `validate_run_settings` | `validate_experiment_run_settings` | `unvalidated_run_settings` → 成功なら`None`、不正なら`RunSettingsValidationError` | 値・適用条件を検証。入力更新なし | 初回 |
| `list_choices` | `list_component_option_definitions` | `component_category`, `component_option_catalog` → 方式定義の一覧 | 指定機能の方式定義を列挙。更新なし | 後続 |
| `find_choice` | `get_component_option_definition` | 上記と`component_option_name` → 一つの方式定義 | 正式名で取得。未知なら入力エラー。任意値を返す検索とは区別 | 後続 |
| `expand_preset` | `get_experiment_preset_settings` | `experiment_preset_name`, `experiment_preset_definitions` → 設定値のコピー | 名前に対応する値を取得。再帰展開は要件にない | 後続 |
| `apply_explicit_settings` | `merge_explicit_run_settings_overrides` | `preset_run_settings`, `explicit_run_settings_overrides` → `unvalidated_run_settings` | 明示指定を優先して新しい値を返す。入力更新なし | 後続 |
| `resolve_run_settings` | `resolve_and_validate_experiment_run_settings` | preset名、明示変更、定義 → `ResolvedExperimentRunSettings` | 取得・反映・検証・確定。実装インスタンスは生成しない | 後続 |
| `validate_choice_catalog` | `validate_configuration_definitions` | 方式定義、preset定義 → 成功なら`None`、不正なら`ConfigurationDefinitionError` | preset参照・項目定義・既定値も検査 | 後続 |
| `run_settings_to_dict` | `serialize_resolved_run_settings_to_mapping` | `resolved_run_settings` → version付き辞書 | 保存用表現へ変換。JSON文字列の生成やファイル書込みなし | 後続 |
| `run_settings_from_dict` | `restore_and_validate_run_settings_from_mapping` | `serialized_run_settings_mapping`, 定義 → `ResolvedExperimentRunSettings` | versionと内容を検証して復元。旧形式の自動解釈なし | 後続 |

未検証の値から不変型を作る経路は設計で定める。型を作っただけで検証済みと見なさない。
後続の関数は必要になった段階の設計で追加・削除を判断する。
内部の一時変数も、担当task開始前に必要な範囲だけ追記する。

## 5. 引数・フィールド・変数

| revision 1 | 改善案 | 型・単位 | 役割・所有者・更新 |
|---|---|---|---|
| `preset_name` | `experiment_preset_name` | 文字列 | 選ぶ実験構成の名前。入力 |
| `component_name` | `component_category` | 正式名 | 検出・予測routing等の機能区分。方式名ではない |
| `choice_name` | `component_option_name` | 正式名 | その機能で選ぶ方式の名前 |
| `choice_catalog` | `component_option_catalog` | `ComponentOptionCatalog` | 使用可能な方式定義の集合。読取り専用 |
| `preset_definitions` | `experiment_preset_definitions` | preset名→定義 | 使用可能な実験presetの集合。読取り専用 |
| `preset_settings` | `preset_run_settings` | 項目→値 | presetから取得した値。解決処理内の一時値 |
| `explicit_settings` | `explicit_run_settings_overrides` | 項目→値 | 明示した変更。未指定と既定値の明示指定を区別 |
| 解決候補（名前なし） | `unvalidated_run_settings` | 項目→値 | 検証前の入力。検証済み型と混同しない |
| `resolved_settings` | `resolved_run_settings` | `ResolvedExperimentRunSettings` | 検証して確定した実効条件。不変 |
| `settings_document` | `serialized_run_settings_mapping` | 辞書 | 保存形式の設定。ファイルやJSON文字列ではない |
| `schema_version` | `run_settings_schema_version` | 正整数 | 設定保存形式のversion。コードの版とは別 |
| `dataset_name` | `dataset_name`（維持） | 正式名 | データセットの選択。既に対象が明確 |
| `random_seed` | `random_seed`（維持） | 整数 | 実験の乱数seed。実行中に変更しない |
| `client_count` | `client_count`（維持） | 正整数・client数 | 参加クライアント数。既に対象と量が明確 |
| `samples_per_client` | `per_client_sample_count` | 正整数・sample/client | 一クライアントのstream長。全client合計とは区別 |
| `aggregation_interval_samples` | `server_aggregation_interval_per_client_samples` | 正整数・sample/client | サーバ集約まで各clientが処理する標本数。旧Aに対応 |
| `adapter_rank` | `residual_adapter_requested_rank` | 正整数・rank | 指定する低rank残差補正のrank。旧実装では特徴次元を超えると丸められる |
| 上記と混同しやすい値 | `residual_adapter_effective_rank` | 正整数・rank | 構造決定後の実際のrank。指定値とは別の記録。決定時点は設計で定める |
| `detection_alpha` | `e_sr_false_alarm_control_alpha` | 実数・無次元 | e-SRの誤警報制御値。ADWINのdeltaや一回の判定の誤警報確率と同一視しない |
| `routing_strategy` | `prediction_routing_strategy` | 正式名 | 予測の組合せ方式。データ割当方式とは別 |
| `routing_activation_policy` | `prediction_mixture_activation_policy` | 正式名 | 混合予測を使う期間の方針 |
| `routing_recalibration_policy` | `post_aggregation_routing_recalibration_policy` | 正式名 | サーバ集約・配布後の過去の損失情報の扱い |
| `assignment_change_reset_policy` | `routing_reset_on_assignment_change_policy` | 正式名 | データ割当先が変わったときのrouter状態の扱い |
| `forward_validation_sample_count` | `candidate_future_validation_sample_count` | 正整数・sample/client | 候補提案後の標本の検証件数。前半・後半を合わせた総数 |
| `assignment_buffer_sample_count` | `pending_assignment_buffer_capacity_samples` | 正整数・sample/client | 帰属確定を保留するFIFOの設定容量。現在件数や検証件数とは別 |
| `parameter_name` | `configuration_parameter_name` | 文字列 | エラーや定義が指す設定項目 |
| `parameter_value` | `specified_parameter_value` | 定義に対応する値 | 検証対象の指定値。既定値・解決後の値とは区別 |

`residual_adapter_effective_rank`は実体構築時の値でもあり、設定解決時に必ず決まるとは限らない。
設定保存と実行結果のどちらで記録するかは設計レビューで確定する。
旧`NEW_MODEL_FORWARD_VALIDATION_SAMPLES`は候補検証以外の診断等にも使われている。
新実装では候補検証の設定を無関係な機能へ渡さず、必要なら担当機能の設定名を別途レビューする。
`gamma`等の旧記号も、用途を確認してから正式名を決める。

## 6. 改善した点と残るレビュー

- `Detection`へ`Drift`、`Routing`へ`Prediction`、`Consolidation`へ`Model`を加え、対象を明示した。
- 取得・検証・復元の役割と、未知の方式を指定した場合の挙動を区別した。
- 設定入力の誤りと定義側の不整合を別の例外名に分けた。
- クライアントごとの標本数、FIFOの容量、検証総数、Adapterの指定rankと実rankを区別した。
- 分かりやすい既存名は維持した。単語数を増やすこと自体を目的にしない。

上記revision 2の候補は承認済み。下記の配置・型・公開フィールド・正式値は設計時の追加案である。
未記載の方式名・詳細フィールド・後続機能は、その実装単位の開始前に追加してレビューする。

## 7. 設計時の追加案（revision 4で承認済み）

初回の内部集約型は`ValidatedExperimentRunSettingsSubset`とする。
各値と組合せを検証した実験設定の一部を表し、完全な実行・保存用の`ResolvedExperimentRunSettings`と区別する。
型名を変えるだけのaliasや自動昇格は作らない。両型の直接構築時にも集約の組合せ検証を行う。

### ファイルと型

全て`src/federated_learning_experiments/`以下。各型の初回フィールドは次表に示す。

| 新しい配置名 | 型・役割 |
|---|---|
| `learning/models/model_architecture_settings.py` | 承認済み`ModelArchitectureSettings`の所属先 |
| `learning/training/local_training_settings.py` | 承認済み`LocalTrainingSettings`の所属先 |
| `methods/fedsda/detection/drift_monitoring_settings.py` | 承認済み`DriftMonitoringSettings`の所属先 |
| `methods/fedsda/routing/prediction_routing_settings.py` | 承認済み`PredictionRoutingSettings`の所属先 |
| `methods/fedsda/assignment/data_assignment_settings.py` | **新規`DataAssignmentSettings`**。帰属保留容量の固定条件。バッファ内容・割当先IDは持たない |
| `methods/fedsda/model_creation/candidate_model_creation_settings.py` | 承認済み`CandidateModelCreationSettings`の所属先 |
| `methods/fedsda/consolidation/model_consolidation_settings.py` | 承認済み`ModelConsolidationSettings`の所属先 |
| `core/configuration_errors.py` | 承認済み例外型の所属先。機能固有の設定型や外側の検証処理をimportしない |

テストの配置候補は`tests/refactoring/test_run_settings_validation.py`、後続は
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
| `drift_monitoring_settings` | `DriftMonitoringSettings` | 損失監視の方式と判定条件 |
| `prediction_routing_settings` | `PredictionRoutingSettings` | 予測混合の方式と状態管理方針 |
| `local_training_settings` | `LocalTrainingSettings` | ローカル学習の方式 |
| `data_assignment_settings` | `DataAssignmentSettings` | 帰属保留に関する固定条件 |
| `candidate_model_creation_settings` | `CandidateModelCreationSettings` | 候補の作成・採否の固定条件 |
| `model_consolidation_settings` | `ModelConsolidationSettings` | サーバ側のモデル統合条件 |

非適用の機能設定をどう表現するかは方式追加時に具体化する。初回は最終FedSDA構成の型だけを扱う。

### 機能固有の初回フィールド

既に承認した数値・方針名は維持し、所属先を明示する。

| 所属する型 | 新しい公開名 | 型・単位・役割 |
|---|---|---|
| `ModelArchitectureSettings` | `model_architecture_name` | 正式名。モデル構造の選択 |
| `DriftMonitoringSettings` | `drift_detector_name` | 正式名。個々の検出器の計算方式 |
| 同上 | `loss_monitoring_scope` | 正式名。全体・正解クラス別等の監視対象。検出器の種類とは別 |
| `LocalTrainingSettings` | `shared_backbone_update_strategy` | 正式名。共有部の更新方法 |
| 同上 | `shared_backbone_gradient_combination_strategy` | 正式名。概念別勾配の統合方法 |
| `CandidateModelCreationSettings` | `candidate_model_creation_policy` | 正式名。候補と比較用モデルの学習・検証・採否方針 |
| `ModelConsolidationSettings` | `model_clustering_trigger_policy` | 正式名。クラスタリングを開始する条件 |
| 同上 | `model_pair_comparison_strategy` | 正式名。統合候補となるモデル対の比較方式 |
| 同上 | `model_clustering_linkage` | 正式名。対評価からクラスタを作るlinkage |
| 同上 | `model_consolidation_policy` | 正式名。クラスタ作成後の統合方針 |
| `PredictionRoutingSettings` | `switching_share_horizon_samples` | 2以上の整数・sample/client。Fixed-Shareの時間尺度。容量とは意味が異なるが、今回の最終構成ではFIFO容量から導出し同値を検証する |

既存名の所属は次のとおり。

- `ExperimentRunConditions`: `dataset_name`、`random_seed`、`client_count`、`per_client_sample_count`、`server_aggregation_interval_per_client_samples`。
- `ModelArchitectureSettings`: `residual_adapter_requested_rank`。実効rankはモデル構築後の記録へ置き、初回設定型に持たせない。
- `DriftMonitoringSettings`: `e_sr_false_alarm_control_alpha`。
- `PredictionRoutingSettings`: `prediction_routing_strategy`、`prediction_mixture_activation_policy`、`post_aggregation_routing_recalibration_policy`、`routing_reset_on_assignment_change_policy`。
- `DataAssignmentSettings`: `pending_assignment_buffer_capacity_samples`。
- `CandidateModelCreationSettings`: `candidate_future_validation_sample_count`。

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
| `prediction_routing_strategy` | `switching_fixed_share_mixture` | Fixed-Shareで優勢モデルの変化を追跡する混合 |
| `prediction_mixture_activation_policy` | `always` | 常時混合 |
| `post_aggregation_routing_recalibration_policy` | `fifo_loss_replay` | 集約後モデルでFIFOを再評価して損失情報を再較正 |
| `routing_reset_on_assignment_change_policy` | `restart_adahedge_preserve_switching` | 予測側AdaHedge・文脈・shadow routerと有効なactive setをrestartし、Switching・Meta-switching・oracle概念別診断routerは保持。全routingをresetする意味ではない |
| `shared_backbone_update_strategy` | `joint_shared_backbone_update` | 概念別ストアの損失で共有部と概念固有のAdapter・headを共同学習。共有部だけの更新ではない |
| `shared_backbone_gradient_combination_strategy` | `mean` | 概念別勾配を平均 |
| `candidate_model_creation_policy` | `future_split_validation_with_reference_refitting` | 既存モデル再適合・現行優先・前半後半の優位確認。数値の判断式は変更しない |
| `model_clustering_trigger_policy` | `on_new_model_registration` | 新規モデルの登録を受けて統合候補を調べる |
| `model_pair_comparison_strategy` | `class_functional_confidence` | クラス別の機能的評価に基づく比較 |
| `model_clustering_linkage` | `average` | average linkage |
| `model_consolidation_policy` | `merge` | クラスタ内モデルを統合 |

これらの値の承認はアルゴリズム変更の承認ではない。移植時には旧構成と判断・順序を照合する。
γ、候補のoptimizer、事前学習・データ生成・評価ストア等の詳細は初回に含めず、実行層の移植前に追加レビューする。

### 値域・単位の宣言（追加案）

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

## 8. データセットの正式値（revision 5で承認済み）

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

## 9. 初回実装の内部関数・テスト名（revision 6で承認済み）

revision 5までの公開型・フィールド・正式値は承認済み。以下は実装準備で具体化した追加分。
gpt-6-lunaのレビューと主担当による有用な指摘の反映を完了した。詳細は[luna-naming-review.md](luna-naming-review.md)に記録する。

### 値域検証の共通処理

各機能型の`__post_init__`から、標準dataclassの型注釈・metadataだけを読む共通検証を呼ぶ。
これによりbool拒否・有限性・境界の扱いを機能別に重複実装しない。
共通処理は機能型・集約型・方式カタログ・旧configをimportしない。組合せ検証も担当しない。
機能固有型の許可依存には、既存の基礎例外に加えてこの基礎層の値検証を追加する設計案である。

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
| `test_integer_settings_accept_boundary_values` | seed=0や正整数の下限など、許容する境界を受理する |
| `test_aggregation_interval_can_exceed_stream_length` | 集約間隔がstream長を超える条件を受理する |
| `test_requested_adapter_rank_is_preserved` | 要求rankを設定段階で丸めない |
| `test_monitoring_alpha_rejects_nonfinite_and_out_of_range_values` | alphaの有限性と開区間を検証する |
| `test_component_options_reject_unknown_names` | 正式値以外の方式を拒否し、旧名aliasを持たない |
| `test_candidate_future_validation_accepts_odd_sample_counts` | 候補の将来標本による検証件数は2以上の奇数も許容する |
| `test_settings_instances_are_immutable` | 構築後の条件変更を拒否する |
| `test_field_annotations_and_metadata_declare_parameter_constraints` | 型注釈・単位・値域が同じフィールド宣言に存在する |
| `test_run_settings_validation_accepts_valid_component_mapping` | 有効な初回組合せで成功し、入力を変更しない |
| `test_run_settings_validation_rejects_invalid_component_mapping` | 未知キー・欠落・型違い・非対応手法を拒否する |
| `test_switching_share_horizon_must_match_assignment_buffer_capacity` | 個別値が有効でも容量と時間尺度の不一致を拒否する |
| `test_validation_reports_failures_in_declaration_order` | 複数の不正がある場合の最初のエラーを固定する |
| `test_direct_subset_construction_uses_the_same_validation` | 集約型の直接構築も共通の組合せ検証を通す |
| `test_subset_does_not_share_mutable_input_mapping` | 元mapping変更が構築済み部分型へ波及しない |
| `test_new_package_imports_only_allowed_dependencies` | 設定基盤が標準ライブラリと新設定型だけへ依存する |

テストの引数・一時値は`settings_type`（対象型）、`valid_settings_values`（有効なフィールド値）、
`configuration_parameter_name`（対象項目）、`specified_parameter_value`（試す入力）、
`expected_failure_reason`（期待する不正理由）、`validation_error`（捕捉した例外）、
`unvalidated_run_settings`（検証前mapping）、`validated_settings_subset`（構築した部分型）、
`settings_instance`（単独機能設定）を用いる。

## 10. 実験条件型の分離（revision 7で承認済み）

| 配置 | 名前・役割 |
|---|---|
| `configuration/experiment_run_conditions.py` | 承認済み`ExperimentRunConditions`の所属先。dataset・seed・規模・集約間隔の固定条件だけを持つ |
| `configuration/run_settings.py` | 承認済み`ValidatedExperimentRunSettingsSubset`の所属先。実験条件型と機能型を束ね、外側の組合せ検証を呼ぶ |

実験条件型と集約型を同じモジュールへ置くと、検証モジュールが実験条件型を読む際に集約型までimportし、循環参照になる。
実験条件型を分け、検証モジュールは同型と機能型だけをimportする。公開型・項目・検証契約は変更しない。
gpt-6-lunaのレビューで配置名と分離を妥当と確認し、次の依存注記を反映して承認した。
`ExperimentRunConditions.__post_init__`はcoreの共通フィールド検証だけを呼び、外側の集約型・組合せ検証へ依存しない。
`run_settings.py`を実験条件型の再export窓口にはしない。

## 11. 共通値検証の直接テスト（revision 8で承認済み）

| 名前 | 役割 |
|---|---|
| `test_settings_field_validation_accepts_supported_numeric_values` | 整数・実数項目の許容入力、有限性と開閉境界の受理を検証する |
| `test_settings_field_validation_rejects_invalid_values` | bool・型違い・非有限値・値域違反を拒否する |
| `test_settings_field_validation_rejects_unsupported_annotations` | 未対応の型注釈を黙って検証対象外にしないことを確認する |
| `NumericSettingsForValidation` | 数値宣言だけを持つテスト専用dataclass。実験用の正式型ではない |
| `UnsupportedSettingsForValidation` | 未対応の型注釈を検査するテスト専用dataclass |

gpt-6-lunaは名前と役割が一致し、変更必須の指摘なしと判断した。主担当も維持を採用し、ユーザー委任に従い承認した。
