# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `feddrift-fedem-window` | 構成 | FedEM、全成分を乱数で初期化、学習の標本はその時刻だけ | `feddrift-ifca-window`は固い選択 |
| `feddrift-fedem-window-pretrained-component0` | 構成 | 上と同じで、ID 0だけ共通の事前学習の初期モデル | IFCAの`-pretrained-cluster0`と同じ初期化 |
| `feddrift-fedem-history` | 構成 | FedEM、全成分を乱数で初期化、学習の標本はrunの初めからの全部 | `-window`は時刻の初めに空にする |
| `feddrift-fedem-history-pretrained-component0` | 構成 | 上と同じで、ID 0だけ共通の事前学習の初期モデル | |
| `fedem_responsibility_assignment` | 割り当ての方式の値（案） | 標本ごとの事後の重み $q$ で全モデルを学習し、$\pi_c$ で予測を混ぜる | `minimum_loss_cluster_assignment`は損失が最小の1モデル |
| `mixture_weight` | 記録の列 | clientの成分の重み $\pi_c$ の1成分 | FedHASPのFixed-Shareの重みとは別の量 |
| `max_responsibility_model_id` | 記録の列 | 標本の $q$ が最大の成分のID | 学習の割り当て先の記録ではない（FedEMは全成分を学習する） |

（実装担当が、実際に使った設定の項目・表・内部の関数の名前を追記する）
