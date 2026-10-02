# 機能・方式の命名再検討

2026-10-03のユーザー依頼による検討表。状態は**候補のレビュー段階**。
前回回答の案を保存し、今回の推奨と区別する。実装の正式名はまだrevision 9のままである。
本表は旧呼称の説明と移植候補の索引であり、全候補を新実装へ追加する計画ではない。
実装停止は継続し、task 2.6には着手しない。

## 1. 用語の意味と判断基準

- MoEのgating/routingと本実装は、複数expertの選択・重み付けという意味で関係がある。
  MoEにも出力を混合する方式があり、「MoEは振分け、本実装は混合」と一律に区別しない。
  典型的な学習可能な入力依存gateと、最終FedSDA構成の過去損失に基づくオンライン重み更新を区別する。
  本構成の重みは現在ラベルの観測前に確定し、ラベル観測後に次回用の重みを更新する。
- 本実装の機能名は「予測の選択・混合」／`prediction_combination`とする案を推奨する。
  学習データの帰属、推論計算対象の選択、サーバのパラメータ集約と区別する。
- Fixed-Shareは既存のアルゴリズム名なので残す。`SoftRouting`や`Meta-switching`は本プロジェクトの
  構成呼称として説明し、組合せ全体が既存の標準アルゴリズムだとは主張しない。
- `prediction`は予測出力、`weight`は予測を混合する係数、`parameter`はモデルの学習済みパラメータ。
  名前に対象を含める。`aggregation`はサーバ集約に使い、予測混合は`combination`で表す。
- `refit`は再学習と誤読されるため、適合性を再確認するだけなら`compatibility_check`とする。
  `shadow`は仮モデル、診断用予測、再学習した比較モデルのそれぞれを区別する。
- 既存手法名・データセットIDは維持し、初出に日本語の役割・出典を添える。長さだけで改善を判断しない。
- 方式名は主な動作を表す。全分岐を一つの長い文字列に詰め込まず、例外・判定順序は契約に記載する。
  名前を短くするために異なる判断へ変更したり、今回新しい設定軸へ分解したりしない。

参照:
- [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [Fixed-Shareを記述した後続研究](https://proceedings.mlr.press/v76/mourtada17a/mourtada17a.pdf)

## 2. 型・配置・項目（前回案と今回推奨）

以下の案は未採用。未着手の型も含む。

| revision 9の型 | 前回回答の案 | 今回推奨と役割 |
|---|---|---|
| `ExperimentRunConditions` | 維持 | 維持。データ・規模・seed・集約間隔 |
| `ModelArchitectureSettings` | 維持 | 維持。モデル構造 |
| `DriftMonitoringSettings` | `LossChangeDetectionSettings` | 同案。直接監視するのは損失変化。警報・変化点推定もこの機能に属する |
| `PredictionRoutingSettings` | `PredictionCombinationSettings` | 同案。予測の選択・重み付き混合とその状態管理 |
| `LocalTrainingSettings` | 維持 | 維持。ローカル学習 |
| `DataAssignmentSettings` | `TrainingDataAssignmentSettings` | 同案。学習データのモデル別ストアへの帰属 |
| `CandidateModelCreationSettings` | `CandidateModelEvaluationSettings` | `CandidateModelTrainingAndAcceptanceSettings`。候補学習、既存モデルの適合確認・再利用、比較・登録判断を含む。境界は設計照合で確定 |
| `ModelConsolidationSettings` | 維持 | 維持。モデル対評価・クラスタ候補・統合操作 |

配置案は`routing/`→`prediction_combination/`、`detection/`→`loss_change_detection/`、
`assignment/`→`training_data_assignment/`、`model_creation/`→`candidate_model_selection/`。
最後の配置は既存モデル再利用との境界を確認して確定する。型とファイルの対応も同時に更新する。

| 現在の項目 | 前回案 | 今回推奨と確認点 |
|---|---|---|
| `prediction_routing_strategy` | `prediction_combination_strategy` | 同案 |
| `prediction_mixture_activation_policy` | 維持 | 維持。混合を使う期間だけの方針 |
| `post_aggregation_routing_recalibration_policy` | `prediction_weight_recalibration_after_aggregation_policy` | 同案。サーバ集約後に使う重み更新の状態も対象 |
| `routing_reset_on_assignment_change_policy` | `prediction_state_reset_on_training_assignment_change_policy` | 同案。予測・診断・対象モデル集合の状態を棚卸しして契約を追記 |
| `switching_share_horizon_samples` | `fixed_share_weight_redistribution_interval_samples`、代案`fixed_share_inverse_redistribution_rate_samples` | `fixed_share_weight_redistribution_time_scale_samples`。毎標本再配分するのでintervalは撤回。Hに対し再配分率1/H。実際のモデル切替周期や実験総長ではない |
| `shared_backbone_update_strategy` | 未提示 | `local_model_parameter_update_strategy`。adapter/headも更新する実態を反映。Lunaレビューで対象をmodelとして明示する指摘を反映 |
| `shared_backbone_gradient_combination_strategy` | 未提示 | 維持。共有部の勾配を組み合わせる方式 |
| `candidate_model_creation_policy` | 未提示 | `candidate_model_acceptance_policy`。方式が比較・再利用・採否の順序まで定めることを明記 |
| `candidate_future_validation_sample_count` | 未提示 | `candidate_post_alarm_validation_sample_count`。警報後の新着標本で検証する時点を明示 |
| `model_pair_comparison_strategy` | 未提示 | 維持。比較する量は各方式で説明 |
| `pending_assignment_buffer_capacity_samples` | 維持 | 維持。平時の保留容量。警報時に一時的に容量+1標本となる契約も記載 |

dataset名、seed、client数、stream長、サーバ集約間隔、adapter指定rankと実効rankはrevision 9の案を維持する。
Fixed-Shareの時間尺度Hは最終構成では保留FIFO容量から導出し、同値を検証する設計を維持する。
名前の変更で独立の数値ハイパーパラメータを増やさない。

## 3. 最終構成の方式値（前回案を保存）

| 現在の正式値 | 前回回答の案 | 今回推奨・注意点 |
|---|---|---|
| `shared_backbone_residual_adapter` | 維持 | 維持 |
| `e_sr` | 維持 | 維持。文書では有界損失平均向け混合Shiryaev–Roberts型e-detectorと説明 |
| `overall_and_true_class_losses` | 維持 | 維持 |
| `switching_fixed_share_mixture` | `fixed_share_weighted_prediction` | 同案。適応的Hedge更新＋Fixed-Share型再配分の組合せと説明 |
| `always` | 維持 | 維持 |
| `fifo_loss_replay` | `recompute_buffer_losses_and_replay_weight_updates` | 同案。更新状態を再構築する。空バッファ・単一モデル等の例外は移植時の契約で明示 |
| `restart_adahedge_preserve_switching` | `restart_adahedge_preserve_fixed_share_weights` | `restart_adahedge_preserve_fixed_share_prediction_state`。重み以外に累積分散等も持つためstateに改善し、予測側状態と明示。resetする文脈・診断・active集合、保持する上位・oracle診断も要確認 |
| `joint_shared_backbone_update` | `joint_backbone_adapter_and_head_training` | 同案。概念別ストアから学習 |
| `mean` | `mean_per_concept_gradients` | `sample_weighted_mean_per_concept_gradients`。旧実装はサンプル数で加重。等数なら単純平均と一致 |
| `future_split_validation_with_reference_refitting` | `current_model_first_reuse_then_two_segment_candidate_validation` | 同案。既存モデルの適合再確認は再学習ではない |
| `on_new_model_registration` | 維持 | 維持 |
| `class_functional_confidence` | `classwise_unique_correctness_confidence_bound` | `classwise_unique_correctness_lower_confidence_bound`。下限を明示。対象は片方だけが正解する率であり予測確信度ではない |
| `average` | `average_linkage` | 同案 |
| `merge` | `weighted_parameter_average_and_merge_ids` | 同案。モデルのパラメータ平均とID統合 |

クラス別固有正解率の比較は、左右方向・クラスのWilson同時信頼下限の最大値を使う。
不足クラスの除外と全体へのfallbackもあるので、常に全クラスを保証する名前にはしない。
実装根拠: `federated_drift_experiment/clustering.py::FunctionalPairStats.class_conditional_confidence_distance`。

## 4. 過去の構成・比較候補の呼称

以下は将来の移植候補であり、今回の初回設定に選択肢を追加しない。
日本語の説明名と識別子を対応付ける。名称の正式採用は担当機能の契約レビュー後とする。

### 予測方式・診断

| 旧呼称 | 日本語説明名 | 識別子案・区別 |
|---|---|---|
| `SoftRouting` | モデル予測の重み付き混合 | `weighted_model_prediction`。特定更新則の名前にはしない |
| `hard` | 現行データ帰属モデルによる単独予測 | `assigned_model_prediction`。最大混合重みモデルの選択とは違う |
| Global / `global` | 全入力で共通のAdaHedge予測混合 | `adahedge_prediction_combination`。サーバglobal modelとの混同を避ける |
| Switching / `switching` | Fixed-Share型予測混合 | `fixed_share_weighted_prediction` |
| Context / `predicted_class` | 予測クラス別AdaHedge予測混合 | `predicted_class_adahedge_prediction_combination`。正解クラスは使わない |
| Context leader | 予測クラス別の最大重みモデル | `predicted_class_maximum_weight_model`。最低直近損失モデルとは区別 |
| Meta / `meta_predicted_class` | 共通混合予測とクラス別最大重みモデルの予測を再混合 | `global_mixture_and_class_leader_prediction_combination`。globalは上行の共通混合を指す。後続で用語統一 |
| Meta-switching / `meta_switching` | 2つの予測方式をFixed-Shareで追跡して選択 | `fixed_share_prediction_strategy_selection`。候補は上行と直接Fixed-Share混合 |
| 上位`leader` | 最大重みの予測方式を採用 | `maximum_weight_strategy_prediction`。モデル選択とは区別 |
| 上位`mixture` | 予測方式の出力も重み付き混合 | `weighted_strategy_prediction_combination` |
| `shadow`診断 | 実予測へ採用しない比較予測 | `diagnostic_prediction`。未来情報を使うという意味ではない |
| oracle router | 真の概念IDを使う診断用予測混合 | `true_concept_conditioned_diagnostic_prediction`。実方式へ接続しない |
| `drift_recovery` | 警報後の回復期間だけ予測混合 | `post_alarm_recovery_only`。終了条件は契約で定める |
| `restarting_soft` / `protected_soft` | 状態reset・保護方針を含む旧構成 | 新しい一枚岩の名前は付けず、予測方式と状態reset/保護方針を別に示す。保護対象を確認して正式値を決める |
| `bounded_score` | 正解クラスの確率に基づく有界損失 | `bounded_true_class_probability_loss` |
| `zero_one` | 正誤による0/1損失 | `zero_one_classification_loss` |

### 候補モデルの検証・採用

| 旧方針 | 識別子案 | 判定の違い |
|---|---|---|
| `immediate` | `immediate_candidate_registration` | 候補比較なしで登録 |
| `validated` | `detection_interval_tail_holdout_candidate_validation` | 検知区間末尾のholdout。全体と後半の優位。初稿の`pre_alarm_holdout_candidate_validation`は警報を起こす標本も含み得るため撤回 |
| `forward_validated` | `post_alarm_full_and_recent_candidate_validation` | 新着区間全体と後半の優位。区間は重複 |
| `forward_requalified` | `existing_model_compatibility_then_candidate_validation` | 既存モデルの適合確認後に候補比較 |
| `forward_requalified_current_first` | `current_model_first_reuse_then_full_and_recent_candidate_validation` | 現行モデル優先。重複区間を使う |
| `forward_persistent` | `current_model_first_reuse_then_two_segment_candidate_validation` | 現行モデル優先。重複しない前半・後半双方で候補が優位 |
| Shadow tournament | `candidate_and_retrained_reference_comparison` | 再学習する比較用コピーを含む方式。互換性の再確認と区別。詳細な採否条件は移植前に調査 |
| provisional / 仮モデル | `unregistered_candidate_model` | 正式登録前のモデル。診断用予測ではない |

### 集約後の状態・学習・統合

| 旧呼称 | 識別子案 | 確認点 |
|---|---|---|
| `none`（再較正） | `preserve_prediction_weight_state` | 何もしない対象を明示 |
| `aggregation_restart` | `reset_prediction_weight_state_after_aggregation` | reset対象を方式別に列挙 |
| `leader_change_replay` | `replay_weight_updates_on_leading_model_change` | 集合変更時のreplay等は契約へ |
| `persistent_leader_change_replay` | `replay_weight_updates_on_two_segment_challenger_advantage` | challengerが両非重複区間で旧leaderより優位 |
| `sequential`（学習） | `sequential_per_concept_parameter_updates` | 概念別に共有部も順次更新 |
| `frozen`（学習） | `train_concept_parameters_with_frozen_backbone` | 共有部を固定し概念固有部を学習 |
| `pcgrad` | `pcgrad` | 既存手法名を維持。射影と加重集約の実装差は説明に記す |
| `parameter_share` | `weighted_parameter_average_preserving_model_ids` | IDを残したパラメータ共有 |
| `noninferiority_merge` | `merge_with_paired_loss_noninferiority_check` | 採否・代表選択・部分統合を契約へ |
| `functional` | `maximum_unique_correctness_rate` | 片方だけ正解する率の最大値 |
| `oracle_concept`（統合診断） | `true_concept_id_equality` | 真の概念IDが同じかで候補生成 |
| active / archive | `prediction_evaluated_model_set` / `prediction_skipped_model_set` | 推論対象と休止対象。学習停止・配布停止・サーバ削除を意味しない |
| `periodic_forward_probe`（実予測） | `periodic_full_model_evaluation_then_selective_prediction` | 全評価と絞り込みを交互に行う。誤り駆動probeと更新凍結は契約へ |
| `Cached` / `NoCached` | 移植時に`cross_evaluation_parameter_cache_policy`等を検討 | 何をどこにキャッシュするか確認して方式化。旧mode名へ他機能と連結しない |
| `gamma` / `distance_threshold` | 用途別に`existing_model_loss_increase_tolerance`等を検討 | 既存モデル適合と統合判定で意味が違う。名前の分離だけで値・判定を変えない |

## 5. 文書・承認・完了済みtaskの扱い

1. `naming.md`は正式な実装契約の入口とする。本表へリンクし、レビュー中と正式採用を区別する。
   本表の前回案は議論履歴として保存し、正式名を二重定義しない。
2. 今回Lunaに候補の意味・誤読・欠落をレビューさせる。採否を本表に記録する。
   reset範囲等の未確定案を一括で実装承認にはしない。
3. 次の実装前に、変更する名前・配置・正式値・契約・対象ファイルを小さな改名単位で確定する。
   その差分だけLunaに最終レビューさせ、有用な指摘を反映してnaming revisionを承認する。
   要求・設計・taskの変更が必要なら通常のSDD承認も維持する。
4. 完了済みtask 1.1〜2.5は完了履歴・テスト証拠を残す。checkboxを戻して作り直さない。
   追加の改名task案を作り、既存の機能taskの後続変更として扱う。task 2.6再開より前に必要な改名を揃える。
5. 改名taskは予測、損失監視、学習データ帰属などの責務で区切り、名前とimport・宣言・テスト・設計を同時更新する。
   移行用aliasや旧値の互換読込みを新経路へ残さない。未実装の型は改名taskで実装せず、その担当taskに新名を反映する。
6. 検証は設定schema・対象機能・旧golden・最終構成goldenと、影響範囲のimport/旧名残存確認。
   名前変更のためにgolden数値を更新しない。新実装へまだ実行経路が接続されていないことも報告する。
7. 旧実験のmanifest・raw・CSVは元名のまま保存する。旧名→新名対応は資料と比較テストで管理する。
   新APIのaliasにはしない。既存研究結果の名称変更は表示・説明と来歴を対応付けて行う。

### 改名taskの候補（未実行・追加task承認前）

- 予測の型・配置・項目・Fixed-Share正式値を一単位で変更し、reset範囲を明文化する。
- 損失変化検出と学習データ帰属の型・配置・集約型の参照案を揃える。
- 未着手task 2.6〜3.3の命名と参照を改訂する。型・アルゴリズムの先取り実装はしない。
- 比較方式・診断・成果物の呼称は担当specの開始時に最終レビューする。

## 6. 未確定事項

- `PredictionCombinationSettings`が単独予測を含むことは型の説明で明示する。
- 状態resetの正式値は実際の対象・保持状態の一覧と照合する。略称だけで対象を断定しない。
- 候補学習・採否の型と既存モデル再利用の責務境界、配置名を設計に照合する。
- 監視機能の改名は検出器の入出力・警報時の処理を変えない。研究上の「ドリフト」は概観の説明で維持する。
- 前回の`mean_per_concept_gradients`は加重条件を省略していたため改善した。
- 前回の再配分`interval`案は周期実行と誤読されるため改善した。

## 7. Lunaの候補レビューと対応

2026-10-03、gpt-6-lunaが旧実装と候補表を照合した。主担当は以下を反映した。
レビューは候補段階のものであり、未確定の契約・配置を含む一覧全体の実装承認ではない。

| 指摘 | 判断・反映 |
|---|---|
| `pre_alarm`は警報を起こす標本を含む検知区間と矛盾 | 採用。`detection_interval_tail_holdout_candidate_validation`へ改善。旧文書の「警報以前のみ」表現も移植時に訂正する |
| classwise下限名は全体fallbackを誤読させ得る | 現案維持。利用可能クラスなしなら全体集計、左右・クラスへのBonferroni補正を契約に明記する。全条件を識別子へ連結しない |
| Fixed-Share保持状態はweightsだけではなく予測側状態と明示すべき | 採用。`restart_adahedge_preserve_fixed_share_prediction_state`を候補に。全reset範囲のコード照合前には確定しない |
| 再配分time scaleは妥当、FIFO由来を維持すること | 採用。毎標本適用・率1/H・容量から導出を明記し、新しい設定自由度を増やさない |
| 候補TrainingAndAcceptance型の説明へ既存モデル再利用を含める | 採用。役割を明記し、型名自体のさらなる長文化は避ける。責務境界は後続設計で確認 |
| `local_parameter_update_strategy`は更新する対象が曖昧 | 一部採用。候補を`local_model_parameter_update_strategy`に改善し、localとmodelの両方を明示 |
| SoftRouting、Fixed-Share予測、Meta-switching案は動作と一致 | 維持。実行方式・診断・上位選択を分けた説明を採用 |

具体的な改名単位を確定した際に、そのファイル・公開名・フィールド・値・テスト名・状態の一覧を
再度Lunaへ提示する。そこで採否を反映し、naming.mdの正式契約とspec.jsonの承認を更新する。
