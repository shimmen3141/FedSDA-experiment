# 旧SINE goldenの条件と実行順序の棚卸し

## 根拠と調査範囲

- 旧ソース基準: `748c3aa`。観測時HEAD: `a27b1fd`。旧ソース・goldenに差分なし。
- 正本: `tests/test_proposed_regression.py`のdefinition/run_caseと`tests/proposed_regression_golden.json`のSINEケース。
- 固定Windows CPU環境でSINEケースを1回実行し、33指標・離散列・件数を既存goldenと比較して一致した。
- 観測件数: 設定module属性65種類。設定適用・保存・診断の参照も含む。読まれたことだけを、手法への適用の証拠にしない。
- 登録・復元時の旧既定値を実効値と混同しない。下表の値はgoldenの保存済み定義（既定値の展開を含む）を優先する。
- 単一seed・経路の観測である。未通過の分岐、裸のmodule内global、constructor既定引数、数値定数は別途ソースで確認する。
- 実験結果: 新規登録4、候補採用5、候補棄却3、吸収モデル2、切替32。これは旧実行の証拠である。
- 調査用記録は共有venv内。再開に必要な結論は本書へ保存し、一時ファイルを正本にしない。

## 条件の照合表

旧名は移植元の参照位置に限る。新形式では既存正式名か、追加レビュー済みの名前だけを使う。

| 旧参照名 | SINEでの値・由来 | 実処理の参照例 | 追加・移植時の扱い |
|---|---|---|---|
| `ADWIN_MAX_WINDOW` | `1000`（既定/派生） | `federated_drift_experiment/clients/fedsda.py:1140:__init__` | 検出。e-SR候補変化点の保持上限へ改名・追加が必要 |
| `AGGREGATION_INTERVAL` | `50`（golden定義） | `federated_drift_experiment/experiment.py:2251:run_random_drift_experiment` | ExperimentRunConditions.server_aggregation_interval_per_client_samples |
| `AMSGRAD` | `true`（golden定義） | `federated_drift_experiment/models.py:215:_build_component_optimizer` | 学習。Adamの方式条件 |
| `BASE_LR` | `0.01`（golden定義） | `federated_drift_experiment/models.py:316:__init__` | 学習。通常・初期学習の学習率 |
| `CLIENT_BATCH_SIZE` | `32`（golden定義） | `federated_drift_experiment/clients/base.py:76:__init__` | 学習。ローカル・候補のミニバッチ上限 |
| `CLUSTER_MIN_EVAL_N` | `5`（既定/派生） | `federated_drift_experiment/servers/clustering.py:345:perform_hierarchical_clustering` | 統合。評価に必要な最小標本件数 |
| `CONCEPT_SCHEDULE` | `"random"`（golden定義） | `federated_drift_experiment/data/schedules.py:98:make_concept_schedules` | データ供給。系列方式の追加契約 |
| `CONCEPT_SCHEDULES` | `["random", "feddrift_fixed"]`（既定/派生） | `federated_drift_experiment/data/schedules.py:99:make_concept_schedules` | 登録一覧。runの入力条件ではない |
| `CROSS_EVAL_MAX_CLIENTS` | `3`（golden定義） | `federated_drift_experiment/servers/clustering.py:84:_cross_evaluate` | クロス評価。モデルあたりclient上限 |
| `DATASET` | `"sine2"`（golden定義） | `federated_drift_experiment/experiment.py:2223:run_random_drift_experiment` | ExperimentRunConditions.dataset_name |
| `DELAY_TOLERANCE` | `100`（golden定義） | `federated_drift_experiment/metrics.py:93:compute_metrics` | 評価。検出と真のドリフトの対応許容遅延 |
| `DRIFT_PROB` | `0.015`（golden定義） | `federated_drift_experiment/data/schedules.py:97:make_concept_schedules` | データ供給。試行可能な各標本での変更確率 |
| `EVAL_MAX_SAMPLES` | `50`（golden定義） | `federated_drift_experiment/clients/base.py:541:_cross_evaluation_data` | クロス評価。使用標本上限 |
| `EVAL_STORE_SAMPLE_SIZE` | `20`（golden定義） | `federated_drift_experiment/clients/base.py:154:_store_evaluation_data` | 評価データ管理。一度に追加する標本上限 |
| `E_DETECTOR_ALPHA` | `0.001`（golden定義） | `federated_drift_experiment/clients/fedsda.py:1139:__init__` | LossChangeDetectionSettings.e_sr_false_alarm_control_alpha |
| `FEDDRIFT_CLUSTER_LINKAGE` | `"average"`（golden定義） | `設定適用時だけ観測` | このSINE経路では実処理参照なし。適用対象を再確認 |
| `FEDSDA_CLUSTERING_CONFIDENCE` | `0.95`（golden定義） | `federated_drift_experiment/servers/fedsda.py:28:__init__` | 統合。信頼水準 |
| `FEDSDA_CLUSTERING_CONSOLIDATION` | `"merge"`（golden定義） | `federated_drift_experiment/servers/fedsda.py:32:__init__` | パラメータ平均とID統合へ改名済み |
| `FEDSDA_CLUSTERING_CONSOLIDATIONS` | `["merge", "parameter_share", "noninferiority_merge"]`（既定/派生） | `federated_drift_experiment/servers/fedsda.py:41:__init__` | 登録一覧。runの入力条件ではない |
| `FEDSDA_CLUSTERING_DECISION` | `"class_functional_confidence"`（golden定義） | `federated_drift_experiment/servers/fedsda.py:25:__init__` | クラス別固有正解率の信頼下限へ改名済み |
| `FEDSDA_CLUSTERING_POLICY` | `"on_new_model"`（golden定義） | `federated_drift_experiment/experiment.py:106:_run_per_sample_timestep` | 新規登録時の統合開始へ改名済み |
| `FEDSDA_CLUSTER_LINKAGE` | `"average"`（golden定義） | `federated_drift_experiment/servers/fedsda.py:26:__init__` | average linkageへ改名済み |
| `FEDSDA_DETECTION_EPISODES_ENABLED` | `false`（golden定義） | `federated_drift_experiment/clients/fedsda.py:71:__init__` | 検出。falseを固定しエピソード統合は移植しない |
| `FEDSDA_DISTANCE_THRESHOLD` | `0.1`（golden定義） | `federated_drift_experiment/experiment.py:2233:run_random_drift_experiment` | 候補・再利用・統合。共用値0.1の用途別契約が必要 |
| `FEDSDA_MERGE_NONINFERIORITY_MARGIN` | `0.0`（golden定義） | `federated_drift_experiment/servers/fedsda.py:34:__init__` | 統合。構築時に読むがmerge方針では非劣性判定なし |
| `FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS` | `1`（golden定義） | `federated_drift_experiment/clients/fedsda.py:60:__init__` | 同期。候補のアップロードまでの区間数 |
| `FIFO_BUFFER_SIZE` | `30`（golden定義） | `federated_drift_experiment/clients/fedsda.py:48:__init__` | TrainingDataAssignmentSettings.pending_assignment_buffer_capacity_samples（Hも同値） |
| `LOCAL_UPDATE_INTERVAL` | `1`（golden定義） | `federated_drift_experiment/clients/base.py:205:train_step` | 学習。更新間隔（標本/client） |
| `MIN_DRIFT_DATA` | `5`（golden定義） | `federated_drift_experiment/clients/fedsda.py:772:_resolve_drift` | 候補・帰属。ドリフト解決に必要な件数 |
| `MIN_STABLE_PERIOD` | `100`（golden定義） | `federated_drift_experiment/data/schedules.py:95:make_concept_schedules` | データ供給。変更試行前の標本位置差（厳密な > 判定） |
| `NEW_MODEL_CREATION_POLICY` | `"forward_persistent"`（golden定義） | `federated_drift_experiment/clients/fedsda.py:849:_resolve_drift` | 候補の採否方針へ改名済み |
| `NEW_MODEL_EARLY_STOPPING_MIN_DELTA` | `0.0001`（golden定義） | `federated_drift_experiment/clients/base.py:387:_train_new_model_early_stopping` | 候補学習と採否。改善判定の量を用途別に確認 |
| `NEW_MODEL_EARLY_STOPPING_PATIENCE` | `3`（golden定義） | `federated_drift_experiment/clients/base.py:386:_train_new_model_early_stopping` | 候補学習。改善なしの継続許容回数 |
| `NEW_MODEL_EPOCHS` | `30`（golden定義） | `federated_drift_experiment/clients/base.py:328:new_model_initial_epochs` | 候補学習。上限エポック数 |
| `NEW_MODEL_FORWARD_VALIDATION_SAMPLES` | `10`（golden定義） | `federated_drift_experiment/clients/fedsda.py:54:__init__` | CandidateModelTrainingAndAcceptanceSettings.candidate_post_alarm_validation_sample_count |
| `NEW_MODEL_INITIALIZATION` | `"best_candidate"`（golden定義） | `federated_drift_experiment/clients/fedsda.py:590:_select_initialization_params` | 候補学習。既存候補からの初期値選択 |
| `NEW_MODEL_LR` | `0.01`（golden定義） | `federated_drift_experiment/models.py:250:reset_optimizer` | 候補学習。学習率 |
| `NEW_MODEL_TRAINING` | `"early_stopping"`（golden定義） | `federated_drift_experiment/clients/base.py:332:_train_new_model` | 候補学習。early stopping方式 |
| `NEW_MODEL_VALIDATION_FRACTION` | `0.2`（golden定義） | `federated_drift_experiment/clients/base.py:367:_train_new_model_early_stopping` | 候補学習。早期終了の分割比。警報後の10件とは別 |
| `N_CLIENTS` | `3`（golden定義） | `federated_drift_experiment/experiment.py:345:_setup_server_and_clients` | ExperimentRunConditions.client_count |
| `OPTIMIZER` | `"adam"`（golden定義） | `federated_drift_experiment/models.py:210:_build_component_optimizer` | 学習。optimizer選択 |
| `PRETRAIN_BATCH_SIZE` | `32`（golden定義） | `federated_drift_experiment/experiment.py:298:_pretrain_initial_model` | 初期学習。ミニバッチの標本数 |
| `PRETRAIN_EPOCHS` | `10`（golden定義） | `federated_drift_experiment/experiment.py:297:_pretrain_initial_model` | 初期学習。エポック数 |
| `PRETRAIN_SAMPLES` | `100`（golden定義） | `federated_drift_experiment/experiment.py:296:_pretrain_initial_model` | 初期学習。初期標本件数 |
| `ROUTING_ACTIVE_SET_POLICY` | `"all"`（golden定義） | `federated_drift_experiment/clients/fedsda.py:1387:__init__` | 予測。全モデル評価を維持。追加方式は移植しない |
| `ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS` | `false`（golden定義） | `federated_drift_experiment/clients/fedsda.py:1804:_record_prediction` | 診断。falseを固定 |
| `ROUTING_ARCHIVE_SHADOW_POLICY` | `"previous_block"`（golden定義） | `federated_drift_experiment/clients/fedsda.py:1806:_record_prediction` | 診断。falseでも読まれる。条件のコピーは保留 |
| `SEA_LABEL_NOISE` | `0.1`（golden定義） | `設定適用時だけ観測` | このSINE経路では実処理参照なし。適用対象を再確認 |
| `SEA_THRESHOLDS` | `{"0": 9.0, "1": 8.0, "2": 7.0, "3": 9.5}`（golden定義） | `設定適用時だけ観測` | このSINE経路では実処理参照なし。適用対象を再確認 |
| `SHARED_ADAPTER_RANK` | `8`（golden定義） | `federated_drift_experiment/models.py:309:__init__` | ModelArchitectureSettings.residual_adapter_requested_rank |
| `SHARED_BACKBONE_GRADIENT_STRATEGY` | `"mean"`（golden定義） | `federated_drift_experiment/clients/shared_backbone.py:276:_train_heads_together` | 概念別勾配の標本数加重平均へ改名済み |
| `SHARED_BACKBONE_ROUTING_RECALIBRATION` | `"fifo_replay"`（golden定義） | `federated_drift_experiment/servers/shared_backbone.py:28:run_round` | 集約後のバッファ再評価・重み更新再生へ改名済み |
| `SHARED_BACKBONE_TRAINING` | `"joint"`（golden定義） | `federated_drift_experiment/clients/shared_backbone.py:189:train_all_held_models` | 共同学習方針へ改名済み |
| `SOFT_ROUTING_ACTIVATION_POLICY` | `"always"`（golden定義） | `federated_drift_experiment/clients/fedsda.py:1327:__init__` | 常時混合へ改名済み |
| `SOFT_ROUTING_CONTEXT` | `"switching"`（golden定義） | `federated_drift_experiment/clients/fedsda.py:1687:_record_prediction` | Fixed-Share予測混合へ改名済み |
| `SOFT_ROUTING_META_LOSS` | `"zero_one"`（golden定義） | `設定適用時だけ観測` | このSINE経路では実処理参照なし。適用対象を再確認 |
| `SOFT_ROUTING_TOP_COMBINATION` | `"leader"`（golden定義） | `設定適用時だけ観測` | このSINE経路では実処理参照なし。適用対象を再確認 |
| `STABLE_WINDOW` | `50`（golden定義） | `federated_drift_experiment/metrics.py:95:compute_metrics` | 評価。各ドリフト後の除外窓 |
| `STORED_DATA_LIMIT` | `50`（golden定義） | `federated_drift_experiment/clients/base.py:43:__init__` | 評価データ管理。モデル別ストア上限 |
| `TOTAL_DATA_POINTS` | `1500`（golden定義） | `federated_drift_experiment/experiment.py:2252:run_random_drift_experiment` | ExperimentRunConditions.per_client_sample_count |
| `UPDATES_PER_SAMPLE` | `1`（golden定義） | `federated_drift_experiment/clients/base.py:77:__init__` | 学習。標本あたり更新回数 |
| `WEIGHT_DECAY` | `0.001`（golden定義） | `federated_drift_experiment/models.py:214:_build_component_optimizer` | 学習。減衰係数 |
| `dataset_spec` | `{"derived_function": "dataset_spec"}`（既定/派生） | `federated_drift_experiment/models.py:299:__init__` | 派生参照。構造・学習率等を明示化するため静的定義も照合 |
| `num_classes` | `{"derived_function": "num_classes"}`（既定/派生） | `federated_drift_experiment/clients/fedsda.py:1185:__init__` | 派生参照。SINEは2クラス |
| `num_concepts` | `{"derived_function": "num_concepts"}`（既定/派生） | `federated_drift_experiment/data/streams.py:31:generate_data` | 派生参照。SINEは2概念 |

## module属性以外の条件と処理順

| 項目 | 観測元・維持する意味 | 所属候補 |
|---|---|---|
| seed=0 | run_caseの引数。Python・NumPy・torchへ同じseed。乱数方式・消費順も維持 | 実験条件と実行時の乱数所有 |
| 手法とモデル | MODE_SPECSの逐次処理・ResidualAdapterモデル・共有部対応server/client | 実行時の具体方式組立 |
| SINE構造 | data/specs.py: 入力2、概念2、クラス2、隠れ層(32,32)、専用学習率なし | データ定義/モデル構造 |
| SINEラベル | uniform(0,1)^2、x2 <= sin(x1)、concept1で反転。生成後float32へ変換 | データ供給/モデル入力変換 |
| 概念変更 | data/schedules.py: index-last_drift > 100 の時だけ確率0.015で試行。最初の試行可能位置101。各clientは0から開始 | データ供給 |
| 初期準備 | モデルを初期化→concept0の100件を生成→10epochsで毎回Python shuffle→初期損失統計→server/client構築 | 初期学習と実行順序 |
| stream準備 | 初期準備後、全clientの概念系列→client順に全標本生成。事前学習と同じ乱数源を継続 | データ供給と実行順序 |
| 区間処理 | floor(1500/50)=30区間。各標本位置でclient0,1,2の順。区間末は保留更新→状態要約→同期→pending昇格 | 実行順序/同期 |
| server内部 | 新規ID登録→FedAvg→任意のクロス評価・統合→一度の配布。実行窓口が統合計算を所有しない | 同期/統合 |
| 終端 | 未完了候補検証の確定→開始済み通信の確定。残余の標本を追加処理しない | 実行順序/各プロトコル状態 |
| e-SRの賭け率 | e_detector.py DEFAULT_LAMBDAS=(.05,.1,.2,.4,.8)、等重み。ADWIN_MAX_WINDOW=1000が候補上限 | 検出（定数と選択可能項目を設計で区別） |
| 数値安全策 | e-SR baseline clamp、AdaHedge/Fixed-Shareの数値式等はソースの計算契約。全定数をオプション化しない | 担当アルゴリズムの移植仕様 |
| モデルの所有 | 共有backboneと概念固有adapter/head、optimizer状態の共有/再設定と初期化順 | モデル/学習 |

## 次段階への接続と保留

1. このspecはSINE供給と実行順序だけを実装する。初期準備・学習・判断処理は接続先の契約として分離する。
2. 初回の不足は系列方式・最小安定期間・変更確率と、乱数状態の受渡し・処理範囲。設定宣言は担当機能へ置く。
3. モデル・学習・候補・同期・統合・評価の追加条件は、その移植単位の命名・設計・taskへ送る。部分型を完全型と呼ばない。
4. 閾値0.1は候補の再利用/採否とモデル対統合で意味が異なる。用途を確認してから別フィールドにするかを決め、値を勝手に変えない。
5. メタ混合の選択肢、SEA設定、FedDrift設定、非劣性統合、archive診断は読まれるだけで新必須入力にしない。
6. 全指標・離散列の新経路golden照合は、処理部の移植と不足条件の充足後に行う。この棚卸しだけで網羅性を確定しない。
