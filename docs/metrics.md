# 実験指標の構成

実験指標の正規ID、分類、適用範囲、CSV列順は
[`experiment_spec/metrics.py`](../federated_drift_experiment/experiment_spec/metrics.py)を正本とする。
指標を追加するときは、実験スクリプトへ列名を直接追加せず、このスキーマへ登録する。

## 1. 三つの利用階層

| 階層 | 用途 | 例 |
|---|---|---|
| `primary` | 手法の主要な比較・結論 | accuracy、stable accuracy、precision、recall、F1、モデル通信量、最終モデル数 |
| `secondary` | コスト、規模、実行環境を含む補助評価 | 軽量メッセージ、計算量、実行時間、平均・最大モデル数 |
| `diagnostic` | 特定機能の原因分析 | 変化点誤差、FP種別、適応操作、仮モデルの採否理由 |

主要な結論は`primary`を中心に述べ、`secondary`でコストとの交換関係を確認する。
`diagnostic`は該当機能を使用した実験だけで解釈し、全手法の優劣を直接表す値としては扱わない。

## 2. 適用範囲

| スキーマ値 | 意味 |
|---|---|
| `all_methods` | Obliviousを含む全手法で定義できる |
| `adaptive_methods` | ドリフトに応じてモデルを切り替える手法で意味を持つ |
| `change_point_estimators` | 変化開始点を明示的に推定する検出器で意味を持つ |
| `fedsda_methods` | FedSDA固有の警報・適応・サーバ割当で意味を持つ |
| `forward_creation_policies` | forward検証を使う仮モデル作成方式でのみ意味を持つ |

現行CSVは既存結果との互換性のため、適用外の列も保持し、多くの場合`0`を格納する。
したがって、`0`を「測定済みで事象がなかった」と解釈する前に、スキーマの`applicability`を確認する。
将来この表現を変更する場合は、CSVのschema versionを上げて欠損値と実測ゼロを区別する。

## 3. 指標群

| group | 内容 | 主な指標 |
|---|---|---|
| `predictive_performance` | 予測性能 | `accuracy`, `stable_accuracy` |
| `detection` | 真のドリフトとの照合 | `precision`, `recall`, `f1`, `avg_delay`, `total_detect` |
| `communication` / `communication_volume` | モデル転送回数、軽量メッセージ、実パラメータ量 | `comm_models_*`, `comm_messages_*`, `comm_parameter_values_*`, `comm_bytes_*` |
| `model_population` | モデル数の終値・時系列要約 | `final_model_count`, `mean_model_count`, `max_model_count`, `model_count_auc` |
| `runtime` / `compute` | 時間とモデル計算回数 | `runtime_seconds`, `client_compute_seconds_*`, `compute_*` |
| `change_point` | 推定変化点の誤差 | `change_point_mae`, `change_point_bias`, `change_point_estimate_count` |
| `alarm_episode` / `false_positive` | 警報エピソードと誤検出の内訳 | `alarm_*`, `switch_fp_*` |
| `adaptation_action` | 再利用・新規作成・維持の判断 | `adaptation_*`, `model_reuse_*` |
| `provisional_model` | 仮モデル方式の採否・標本数・棄却理由 | `provisional_*` |
| `server_mapping` | サーバ割当変更 | `server_mapping_change_count` |
| `model_learning` | モデル別の割当データ・学習サンプル・optimizer stepの分布 | `model_assigned_samples_*`, `model_training_examples_*`, `model_optimizer_steps_*` |
| `model_complementarity` | 同一標本上のモデル対正誤表とoracle選択余地 | `model_pair_*` |
| `clustering_oracle_diagnostic` | 真の概念一致に対するクラスタ判定と距離の診断 | `clustering_oracle_*` |
| `soft_routing` | SoftRoutingが全モデル中の正解可能性を回収できた割合 | `routing_*` |
| `routing_contribution` | 各保持モデルを除外したときの反実仮想損失差 | `routing_loo_*` |

コードから用途別に選ぶ場合は`metrics_in_profile("core")`、`"detection"`、
`"adaptation"`、`"resource"`、`"all"`を使用できる。profileはCSV列を削るものではなく、
表や分析で表示する指標集合を揃えるためのものである。

## 4. CSVとNPZの役割

- CSVには、seed間で平均・標準偏差を計算するスカラー要約を保存する。
- NPZには、回復曲線、クライアント別・ラウンド別計算量、モデル数時系列、SoftRoutingの重みなど、
  後から再集計するための系列を保存する。
- 実行時間は環境依存なので、アルゴリズム比較では`compute_model_examples_total`や
  `compute_optimizer_steps_total`も併記する。
- `comm_models_total`はモデル転送回数であり、モデル構造が同じ場合の比較を前提とする。
- `comm_parameter_values_total`と`comm_bytes_total`は共有部の重複送信を除いた実転送量であり、
  異なるモデル構造の比較ではこちらを主要な通信量とする。
- `compute_backbone_examples_total`と`compute_head_examples_total`は、共有表現による特徴抽出の
  再利用を既存の論理モデル処理数と分けて示す。
- `compute_backbone_optimizer_steps_total`と`compute_head_optimizer_steps_total`は、共有表現方式で
  共有部と概念別ヘッドの実更新回数を分ける。非共有方式では適用外のため0となる。
- `final_parameter_values`と`final_parameter_bytes`は、共有部を1個として数えた最終保持容量である。

## 5. モデル多様性と学習断片化の診断

モデル別学習量は、最終時点のモデルIDごとにクライアント間で集約する。CSVには総量・平均・最小値・
変動係数（CV）だけを保存し、クライアント×モデルの詳細はNPZの`model_diagnostic_*`へ保存する。
CVが大きい、または最小値が小さい場合は、一部モデルへ十分な学習量が届いていない可能性がある。
`model_training_examples_total`は最終的に保持されたモデルへ帰属できる学習量であり、棄却された仮モデルや
一時的shadowの学習は含めない。それらを含む全計算量は`compute_training_examples_total`で確認する。

モデル対予測相補性はFedSDAの既存クロス評価と同じ標本で計測する。追加のモデル通信は行わないが、
対象モデル自身の予測を得るforward計算は追加される。NPZの`model_pair_*`には、候補のみ正解、対象のみ
正解、両方正解、両方不正解の件数を保存する。CSVの主な要約は次のとおり。

- `model_pair_correctness_disagreement_rate`: 片方だけが正解した割合。
- `model_pair_oracle_gain_rate`: 良い方の単一モデルに対し、標本ごとにoracle選択した場合の改善上限。
- `model_pair_both_correct_rate`: 両モデルが正解した割合。

新しいrawでは、集約値に加えて次をラウンド単位で保存する。

- `cross_evaluation_*`: クライアント・候補モデル・対象モデルごとの損失十分統計と正誤表。
- `clustering_pair_*`: モデル対の距離、判定スコア、最終的に同じクラスタへ入ったか。

新しいrawでは、クラスタ判定へ真値を入力せず、事後診断専用として次も保存する。

- `clustering_pair_oracle_same_concept`: 両モデルへ割り当てられた標本の一意な多数概念が同じか。`-1`は未観測または同数首位で判定不能。
- `clustering_pair_personalized_parameter_distances`: 共有backboneを除いたadapter・head間の相対L2距離。通常モデルでは全パラメータを使う。
- `clustering_oracle_merge_*`: 実際の同一クラスタ判定を真の概念一致と照合したTP・FP・FN・TN、precision・recall・F1。
- `clustering_oracle_loss_distance_auc`: クロス評価損失距離の小ささが同一概念を識別するROC-AUC。
- `clustering_oracle_parameter_distance_auc`: 個別パラメータ距離の小ささが同一概念を識別するROC-AUC。

真の概念IDは診断値の算出だけに使い、通常のクラスタリング、学習、routingには渡さない。個別パラメータ距離はサーバへ既にアップロード済みのモデルから計算するため、追加通信や追加forwardを発生させない。

これにより、固定閾値を変更せずに、実際に統合されたモデル対と残されたモデル対の機能的相補性を
事後比較できる。集計には
`python -m tools.experiments.clustering_functional_diagnostics <result-root> --output <csv>`を使う。
旧rawはラウンドとモデル対の対応を保持していないため、この集計の対象外である。

SoftRoutingでは、予測時にすでに計算している全保持モデルの出力から次を記録する。

- `routing_oracle_accuracy`: 実混合または少なくとも一つの保持モデルが正解した割合。
- `routing_mixture_accuracy`: 実際の重み付き混合が正解した割合。
- `routing_leader_accuracy`: 最大重みモデル単体が正解した割合。
- `routing_oracle_gain_rate`: oracle accuracyと実混合accuracyの差。
- `routing_oracle_recovery_rate`: oracleが正解可能だった標本のうち実混合も正解した割合。
- `routing_confidence_leader_accuracy`: 各標本で予測確信度が最大の保持モデルを選んだ場合の
  影評価accuracy。正解ラベルはモデル選択に使わない。
- `routing_confidence_leader_oracle_recovery_rate`: oracleが正解可能だった標本のうち、
  最大確信度モデルでも正解できた割合。
- `routing_confidence_leader_missed_oracle_count`: 最大確信度モデルが取り逃したoracle正解数。
- `routing_missed_oracle_count`: 正解モデルが存在したのに実混合が誤答した件数。
- `routing_class_macro_*`: 正解クラスごとに求めたoracle・mixture・leader精度のマクロ平均。
  `routing_class_macro_confidence_leader_accuracy`は最大確信度モデルのクラス別精度を表す。
- `routing_meta_accuracy`: 文脈方式で、global mixtureとcontextual leaderを文脈別AdaHedgeで
  再混合したmeta-routerの精度。`predicted_class`ではshadow、`meta_predicted_class`では実予測である。
- `routing_meta_gain_rate`: meta-router精度と実予測精度との差。shadow診断では昇格余地を表し、
  `meta_predicted_class`では定義上0になる。
- `routing_meta_global_accuracy` / `routing_meta_context_leader_accuracy`: 同一標本上で測る
  2候補それぞれの精度。
- `routing_meta_context_mixture_accuracy`: 同一標本上で測る予測クラス別mixtureの精度。
- `routing_meta_best_candidate_gain_rate`: shadow meta-router精度と、実験全体で良かった方の
  単一候補精度との差。正なら文脈別オンライン選択自体に利益がある。
- `routing_meta_context_leader_weight_mean`: contextual leaderへ与えた平均重み。
- `routing_meta_context_leader_preferred_rate`: contextual leaderの重みが0.5を超えた標本割合。
- `routing_switching_accuracy`: 現行予測を変えずに計算する、Fixed-Share型switching-expertのshadow精度。
- `routing_switching_gain_rate` / `routing_switching_global_gain_rate`: switching-expert精度と、実予測または
  Global mixture精度との差。
- `routing_switching_stable_accuracy` / `routing_switching_recovery_accuracy`: 真のドリフト直後の回復窓を
  除いた区間と、回復窓内に分けたswitching-expertのshadow精度。
- `routing_switching_stable_gain_rate` / `routing_switching_recovery_gain_rate`: 上記2区間における
  switching-expertと実予測の精度差。切替追従の利益と定常時の損失を分離する。
- `routing_switching_effective_experts_mean`: switching-expert重みの逆Simpson指数の平均。
- `routing_switching_leader_switch_count`: 最大重みexpertが切り替わった回数。
- `routing_switching_pool_reset_count`: 保持モデル集合の変化でswitching-expert状態を初期化した回数。
- `routing_switching_recalibration_sample_count`: 共有表現集約後の再較正に使ったFIFOサンプル総数。
- `routing_meta_switching_accuracy`: 現行Meta mixtureとswitching mixtureを上位Fixed-Shareで選んだ精度。
- `routing_meta_switching_meta_gain_rate` / `routing_meta_switching_switching_gain_rate`: 上位選択精度と
  各候補単独精度との差。
- `routing_meta_switching_selected_switching_rate`: 上位選択がswitching mixtureを採用した標本割合。
- `routing_meta_switching_leader_switch_count`: 上位選択候補が切り替わった回数。
- `routing_class_macro_meta_accuracy`: shadow meta-router精度の正解クラス別マクロ平均。
- `routing_class_macro_meta_global_accuracy` / `routing_class_macro_meta_context_mixture_accuracy` /
  `routing_class_macro_meta_context_leader_accuracy`: 各候補の正解クラス別マクロ平均。

shadow meta-routerは既存のモデル出力だけを再利用するため、モデルforward、通信、モデル管理を
増やさない。ラベルは予測後のAdaHedge更新にだけ使う。時系列はNPZの
`history_routing_meta_correct`と`history_routing_meta_context_leader_weight`へ保存する。
- `routing_class_oracle_gap_mean` / `routing_class_oracle_gap_std`: クラス別の
  `oracle accuracy - mixture accuracy`の平均と標準偏差。標準偏差が大きければ、未回収余地が
  特定クラスへ偏っており、クラス文脈ルーティングを検討する根拠になる。
- `routing_class_oracle_recovery_rate_mean` / `routing_class_oracle_recovery_rate_min`:
  クラス別oracle正解可能標本の回収率について、マクロ平均と最悪クラス値を示す。

NPZには`history_routing_oracle_correct`と`history_routing_leader_correct`に加え、クライアント・
正解クラス別の件数を`routing_class_*`へ保存する。
これらの指標は追加のモデルforwardや通信を発生させない。

SoftRoutingのleave-one-out診断は、実際に使用した混合からモデル`m`を一つ除き、残った実効重みを
正規化した反実仮想予測を作る。`loss(without m) - loss(actual)`を寄与と定義するため、正ならモデルを
残す利益、0以下ならその標本では除いても悪化しなかったことを表す。CSVには次を保存する。

- `routing_loo_bounded_delta_mean`: `[0, 1]`有界予測損失に対する平均寄与。
- `routing_loo_zero_one_delta_mean`: 0/1誤分類損失に対する平均寄与。
- `routing_loo_positive_rate`: モデル除外で有界損失が増えたモデル・標本対の割合。
- `routing_loo_active_unassigned_*`: 最終active集合のうち、どのクライアントにも最終割当されていない
  モデルと、その中で平均寄与が非正だったモデルの数・割合。
- `routing_loo_active_*_joint_nonpositive_*`: 有界損失と0/1損失の寄与がともに非正のモデル数・割合。
  有界損失だけを改善してもaccuracyを下げる場合を、archive候補から区別するために使う。
- `routing_archive_shadow_*`: 選択した因果的な方針で二つの損失寄与がともに非正だったグローバルモデルを、
  そのクライアントの予測集合から外した反実仮想結果。`previous_block`は直前区間を使い、`forward_probe`は
  現在区間先頭`N_forward`件を全モデルで評価してから同一区間の残りを絞る。
  `periodic_forward_probe`は`N_forward`件のprobeと適用を交互に繰り返す。現行hard割当モデルと
  ローカル仮モデルは常に残し、accuracy差・有界損失差・保持モデル割合を記録する。

保持モデル集合が変わると同じモデルIDでも実体が変わり得るため、NPZは`routing_loo_pool_epochs`、
`routing_loo_block_indices`、`routing_loo_model_ids`をキーとして十分統計を保存する。通信区間ごとの
`sample_counts`、`probability_sums`、損失差の和・二乗和、正負件数、hard割当件数から、後でarchive候補の
連続性や分散を再集計できる。実効重みが一点に集中して除外後の正規化が不能な場合だけ、同じ標本で
計算済みのglobal提案重みへ戻し、件数を`routing_loo_fallback_count`へ記録する。この診断も追加forward、
学習、通信を発生させず、現時点ではactive/archive判断を変更しない。NPZには最終グローバル状態を
誤読しないよう、`routing_loo_final_active_model_ids`と`routing_loo_final_assigned_model_ids`も保存する。

shadow archiveは`--routing-archive-shadow-diagnostics`で明示的に有効化し、方針を
`--routing-archive-shadow-policy`で選ぶ。`previous_block`は前区間だけから次区間を決める。
`forward_probe`は現在区間の先頭標本を予測して正解を受け取った後に、未来の標本へだけ保持判断を適用する。
`periodic_forward_probe`も同じ因果的な順序でprobeと適用を周期的に繰り返す。いずれも評価対象標本の
正解による先読みは行わない。モデル集合が変わった場合は全モデル保持へ戻す。
この段階では実配布・学習・通信量を変えず、クライアント別の可逆なactive/archiveへ進む価値だけを評価する。

## 検証付き蒸留

`clustering_distillation_*`は、クラスタ候補をSoftRouting teacherから一つの
adapter/headへ蒸留したときの候補数、採択率、ローカル更新数、学習・検証標本数を表す。
`extra_parameter_values`と`extra_bytes`はteacher転送、student upload、集約後student検証の
追加通信を数える。`break_even_rounds_mean`は採択候補について、追加パラメータ通信量を
以後の概念別adapter/head配布削減量で回収するまでの推定ラウンド数である。
raw NPZには候補クラスタ、集約後studentの最大非劣性上限、採否、候補別追加通信量を保存する。

これらは診断専用であり、現時点ではクラスタリング判定を変更しない。精度改善を主張する指標ではなく、
モデルを残す利益と学習量断片化の原因を切り分けるために使用する。
