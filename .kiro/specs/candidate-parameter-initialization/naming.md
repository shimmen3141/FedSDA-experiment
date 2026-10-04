# 命名: 候補パラメータ初期化
revision: 1

| 名前 | 役割・型・単位・似た名前との違い |
|---|---|
| candidate_parameter_initialization_settings.py | 初期化元の固定選択肢宣言。学習・採否設定とは別 |
| CandidateParameterInitializationSettings | frozen kw-onlyの初期化固定条件。状態を持たない |
| candidate_parameter_initialization_source | str正式choice。候補の採否や再利用適合性ではない |
| assigned_training_model | 現在の学習割当先からコピー。予測のleaderとは別 |
| lowest_evaluated_mean_loss_model | 評価済み平均loss最小。同率先着、未評価モデル除外 |
| equal_mean_of_available_models | 全保有parameter等重み平均。sample数重みではない |
| candidate_parameter_initialization.py | 完成した初期parameter snapshotの選択・コピー |
| select_candidate_initial_parameter_snapshot | keyword-only入力→dict[str,Tensor]または空averageのNone、モデルを生成しない |
| settings | exact専用設定。publicconstructorで再検査 |
| available_parameter_snapshots_by_model_id | 登録順dict[int,dict[str,Tensor]]、モデル実体でなく入力snapshot |
| current_training_model_id | int、現在の学習割当先。既存候補評価の承認名を維持 |
| evaluated_mean_losses_by_model_id | 評価順tuple[(int,float)]。loss最小同率先着の順序を持つ。適合済みだけの列ではない |
| _validate_initialization_inputs | 全入力検査。選択・副作用を持たないprivate helper |
| _validate_parameter_snapshot | name→Tensorの型/CPU/shape以外基本検査 |
| _copy_parameter_snapshot | name順でdetach cloneするprivate helper |
| _average_parameter_snapshots | 登録順stack/meanか先頭cloneを選ぶprivate helper |
| selected_model_id / model_id / evaluated_model_id | 保有/評価側のint ID（負ID可） |
| parameter_name / parameter_value | str key / Tensor state要素。optimizer値ではない |
| parameter_snapshot / first_parameter_snapshot / selected_parameter_snapshot | 一モデル全state / 比較と順序の先頭 / コピー元 |
| mean_loss / lowest_loss_model_id | 有界平均loss / 最小lossの選択ID |
| initialization_source / validated_settings | 正式choice / 再検査した設定copy |
| parameter_snapshots / parameter_values / averaged_parameter_snapshot | 順序付き全state列 / 同key Tensor列 / 構築中の出力dict |
| copied_parameter_snapshot / averaged_parameter_value | detach copy dict / 一keyの平均値 |
| allowed_parameter_dtypes / first_parameter_value | 許可dtype列 / 構造比較・非浮動copyの先頭Tensor |
| seen_evaluated_model_ids / evaluated_model_loss | 重複検査set / (ID,loss)pair |
| input_name | 検査対象field/pathのエラー文脈文字列。state keyであるparameter_nameとは別 |
| validated_parameter_snapshots_by_model_id | 全検査済みsnapshot入力の順序付きdict。登録先モデル状態ではない |
| parameter_names | productionで同一key集合を検査する名前列、先頭snapshotの順序も表す。testでも同じ意味 |
| legacy_client / legacy_model / legacy_parameters / legacy_evaluated_candidates | test-only旧oracle入力。新productionでは使わない |
| initialization_settings / actual / expected / original / snapshots / losses | test-only専用設定 / 実測 / 期待 / 変更前copy / 明示入力 / loss列 |
| model_ids / source / case / invalid_value / model_parameters | test-only ID列/choice/ケース/不正値/明示parameterdict |

## Test-only追加表
| 名前 | 役割 |
|---|---|
| run_legacy_candidate_parameter_initialization | monkeypatchで旧方式へ明示対応してunbound実旧helperを呼ぶ |
| build_candidate_parameter_initialization_inputs | CPU正常parameter fixture、モデルは生成しない |
| assert_candidate_initialization_parameters_match_legacy | key順/shape/dtype/device/equalを比較 |
| test_candidate_parameter_initialization_matches_legacy | 三方式の直接数値照合 |
| test_candidate_parameter_initialization_uses_evaluated_loss_order | 同率先着/評価外の除外 |
| test_candidate_parameter_initialization_preserves_independent_snapshots | 双方向コピー/別呼出非共有 |
| _validate_model_id | exact整数ID、bool拒否のprivate検査 |
| validation_error | 捕捉した検査例外、状態なし |

永続可変状態は持たない。標準test fixture/局所assert変数は状態所有者ではない。
新しいprivate helperや局所名が必要なら実装前に追加表をレビューする。

