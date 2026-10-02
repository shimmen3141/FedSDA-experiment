# 命名・役割レビュー: 設定・選択肢管理の基盤

- revision: 1
- 状態: **人間レビュー待ち。以下は実装済みの名前ではない。**
- 対象: 最初の設定基盤。後続の予測・学習・サーバ等は、その単位の開始前に別表を作る。

## 1. 配置とファイル

新パッケージ名の第一案は`federated_drift`（連合学習のドリフトを扱う）とする。
研究手法名FedSDAの変更は含まない。まず次の設定部分をレビューする。

| 配置案 | 役割 |
|---|---|
| `src/federated_drift/configuration/settings.py` | 解決済みの実験条件・機能別設定の型 |
| `src/federated_drift/configuration/choice_catalog.py` | 選択肢・パラメータ定義と一覧取得 |
| `src/federated_drift/configuration/presets.py` | 最終提案・比較構成の設定値の組合せ |
| `src/federated_drift/configuration/resolution.py` | presetと明示指定から実効設定を生成 |
| `src/federated_drift/configuration/validation.py` | 値・組合せ・定義の整合性検査 |
| `src/federated_drift/configuration/representation.py` | 実効設定と保存用辞書の相互変換。ファイルI/Oなし |

これらは配置の案であり、設計レビュー時に小さすぎる分割を調整できる。

## 2. 型と役割

| 提案名 | 役割 | 状態・似た名前との違い |
|---|---|---|
| `RunSettings` | 一つのrunで使用する解決済み設定 | 不変。入力の生データではなく実効値 |
| `ExperimentSettings` | dataset、seed、規模、集約間隔 | 方法の中身と区別する実験条件 |
| `ModelSettings` | モデル構造とAdapter rank | モデルの重み・optimizerを持たない |
| `DetectionSettings` | 検出方式・監視範囲・判定値 | 検出器の実行時状態は持たない |
| `RoutingSettings` | 予測方式・activation・再較正・reset | 割当先やrouter重みとは別の固定条件 |
| `TrainingSettings` | 更新方式・勾配統合・学習率等 | optimizerやbatchの実行状態は持たない |
| `ModelCreationSettings` | 候補作成・比較・検証の条件 | 進行中の検証sessionとは別 |
| `ConsolidationSettings` | 対評価・linkage・統合方針 | 完成したクラスタやID対応ではない |
| `ParameterDefinition` | パラメータの名前・型・単位・範囲・既定値 | 個々のrunで指定された値とは別 |
| `ChoiceDefinition` | 選択肢の正式名・説明・能力・専用パラメータ・制約 | 実装インスタンスや生成関数を持たない |
| `ChoiceCatalog` | 機能ごとの選択肢定義を集約する | 実行中のモデル等を持たない |
| `PresetDefinition` | 名前付きの設定値の組合せ | choiceは一機能の方式、presetは複数機能の組合せ |
| `ConfigurationError` | 誤指定の項目・値・原因を示す | 不正を黙って既定値へ置換しない |

機能固有の追加項目は、選択方式ごとの設定型へ所属させる。
この一覧は全方式の詳細フィールドまで確定するものではない。担当taskの設計時に詳細表を追加する。

## 3. 関数・メソッド

| 提案名 | 主な入力 | 出力 | 役割・状態更新 |
|---|---|---|---|
| `list_choices` | `component_name`, `choice_catalog` | 選択肢定義の一覧 | その機能で選べる方式を取得。更新なし |
| `find_choice` | `component_name`, `choice_name`, `choice_catalog` | 一つの選択肢定義 | 正式名で取得。不明ならエラー。更新なし |
| `expand_preset` | `preset_name`, `preset_definitions` | 展開した設定値 | 元のpresetを変更しない |
| `apply_explicit_settings` | `preset_settings`, `explicit_settings` | 反映後の設定値 | 明示指定の優先を適用。入力を書き換えない |
| `resolve_run_settings` | `preset_name`, `explicit_settings`, `choice_catalog` | `RunSettings` | 展開・反映・検証・確定を順に実施 |
| `validate_run_settings` | 解決候補の設定、選択肢定義 | 成功または`ConfigurationError` | 値・能力・組合せ・専用値の適用を検証。更新なし |
| `validate_choice_catalog` | 選択肢定義、preset定義 | 成功または`ConfigurationError` | 重複・参照漏れ・既定値の不整合を検出 |
| `run_settings_to_dict` | `run_settings` | version付きの辞書 | 保存表現を作る。ファイル書込みなし |
| `run_settings_from_dict` | `settings_document`, `choice_catalog` | `RunSettings` | 新schemaのversion・内容を検証して復元。旧形式非対応 |

`validate_*`が「成功ならNone、不正なら例外」とする案。エラーを集めて返す方式への変更は設計で判断し、命名表も更新する。
具体的な実装を生成するruntimeの関数は、この単位に含めない。

## 4. 引数・フィールド・変数

| 提案名 | 型・単位の案 | 役割・所有者・更新 |
|---|---|---|
| `preset_name` | 文字列 | 選んだ構成の名前。入力 |
| `component_name` | 文字列 | detection/routing等の機能名。選択肢取得の入力 |
| `choice_name` | 文字列 | 一機能の正式な選択肢名。入力 |
| `choice_catalog` | `ChoiceCatalog` | 有効な定義の集合。読取り専用 |
| `preset_definitions` | preset名→定義 | 使用可能なpreset集合。読取り専用 |
| `preset_settings` | 項目→値 | 展開したpreset値。解決関数内の一時値 |
| `explicit_settings` | 項目→値 | ユーザーが明示した変更。未指定と既定値指定を区別する |
| `resolved_settings` | `RunSettings` | 検証済みで確定したrun条件。不変 |
| `settings_document` | 辞書 | 保存表現から読み込んだ入力。ファイルそのものではない |
| `schema_version` | 正整数 | 設定表現のversion。コードversion・方式名とは別 |
| `dataset_name` | 正式名の文字列 | データセットの選択 |
| `random_seed` | 整数 | runの乱数seed。実行中に変更しない |
| `client_count` | 正整数・client数 | 参加クライアント数 |
| `samples_per_client` | 正整数・sample/client | 一クライアントのstream長。全client合計とは区別 |
| `aggregation_interval_samples` | 正整数・sample/client | 同期間隔。旧Aが表す量 |
| `adapter_rank` | 正整数・rank | 概念固有の低rank補正のrank |
| `detection_alpha` | 実数・無次元 | e-SRの判定設定。ADWINのdeltaとは別 |
| `routing_strategy` | 正式名 | 混合予測の方式。文脈一般を意味しない |
| `routing_activation_policy` | 正式名 | 混合を使う期間の方針 |
| `routing_recalibration_policy` | 正式名 | 配布後の損失情報の扱い |
| `assignment_change_reset_policy` | 正式名 | 割当変更時のrouter状態の扱い |
| `forward_validation_sample_count` | 正整数・sample | 候補の将来標本による検証件数 |
| `assignment_buffer_sample_count` | 正整数・sample | 帰属保留FIFOの長さ。検証sessionの長さとは別 |
| `parameter_name` | 文字列 | エラーや定義が指す設定項目 |
| `parameter_value` | 定義に対応する値 | 検証する指定値。汎用処理でのみ使用 |

最終構成の具体的なchoice名と機能固有フィールドは、次の設計レビューで一覧に追加する。
`gamma`等の旧記号は役割別の意味を確認してから正式名を決め、ここで一括置換しない。

## 5. 人間による確認

特に確認する点は、パッケージ名`federated_drift`、`Settings`/`Definition`/`Catalog`の意味の区別、
`samples_per_client`と`aggregation_interval_samples`の単位、`assignment_buffer_sample_count`の分かりやすさ。
名前だけで役割を理解できるか、長すぎる名前・意味が重なる名前がないかを確認する。
レビュー対象はまず本revision。承認しても未記載の詳細名まで承認したことにはならない。
