# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `feddrift-fedem-window` | 構成 | FedEM、全成分を乱数で初期化、学習の標本はその時刻だけ | `feddrift-ifca-window`は固い選択 |
| `feddrift-fedem-window-pretrained-component0` | 構成 | 上と同じで、ID 0だけ共通の事前学習の初期モデル | IFCAの`-pretrained-cluster0`と同じ初期化 |
| `feddrift-fedem-history` | 構成 | FedEM、全成分を乱数で初期化、学習の標本はrunの初めからの全部 | `-window`は時刻の初めに空にする |
| `feddrift-fedem-history-pretrained-component0` | 構成 | 上と同じで、ID 0だけ共通の事前学習の初期モデル | |
| `fedem_responsibility_assignment` | 割り当ての方式の値 | 標本ごとの事後の重み $q$ で全モデルを学習し、$\pi_c$ で予測を混ぜる | `minimum_loss_cluster_assignment`は損失が最小の1モデル |
| `mixture_weight` | 記録の列 | clientの成分の重み $\pi_c$ の1成分 | FedHASPのFixed-Shareの重みとは別の量 |
| `max_responsibility_model_id` | 記録の列 | 標本の $q$ が最大の成分のID | 学習の割り当て先の記録ではない（FedEMは全成分を学習する） |

実装した設定の項目・表・関数:

| 実装名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDEM_CONFIGURATION_NAMES` と `FEDDRIFT_FEDEM_{WINDOW,HISTORY}[_PRETRAINED_COMPONENT0]_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 末尾へ足した4構成とその集合 | 既存の構成順・保存設定を変えない |
| `model_assignment_policy="fedem_responsibility_assignment"` | `FeddriftRunClientSettings`・`feddrift_model_assignment_policy`宣言 | qによる全成分の学習 | 1モデルへの固い帰属ではない |
| `training_sample_window="all_observed_history"` | 同設定・`feddrift_training_sample_window`宣言 | FedEMの全履歴を全成分へ渡す | 既存の`all_timesteps`は手法ごとの帰属履歴。windowは既存の`current_timestep` |
| `cluster_model_initialization` | 同設定・`feddrift_cluster_model_initialization`宣言 | 既存の`all_random`・`pretrained_cluster0`を再利用 | component0の公開構成名に対応 |
| `fedem_mixture.FedemClientMixture` | runtime | π・qと最後のroundの記録を所有 | モデル・optimizer・学習標本のownerは借用する |
| `FedemClientMixture.predict` | 同module | ラベルを参照せずπで予測確率を混合 | 単一モデルの予測と区別 |
| `FedemClientMixture.append_timestep_samples` | 同module | 観測済み標本を全成分へ追加 | windowだけ以前の標本を空にする |
| `FedemClientMixture.expectation_step` | 同module | 全学習標本のqを計算し、今の時刻のqの平均でπを更新 | M-stepの直前。集約中はπを変えない |
| `_fedem_responsibilities`・`_fedem_per_sample_loss`・`_fedem_weighted_loss` | 同module、内部関数 | 公式のE-step・交差エントロピー・M-stepの式 | 有界損失・分類誤り率は使わない |
| `sample_responsibilities_by_model_id` | `train_feddrift_held_models_for_round` | 学習populationの位置に対応した成分ごとのq | 集約の件数重みではない |
| `fedem_client_mixture` | 記録表 | `timestep_index, client_id, model_id, mixture_weight` | 最後のroundのπ |
| `fedem_sample_responsibility` | 記録表 | `timestep_index, client_id, sample_index, max_responsibility_model_id, max_responsibility` | 最後のroundのqの診断。historyでもその時刻の標本だけ |
