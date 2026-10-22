# LEGACY-022: クロス評価で、対象のモデルの標本を持たないclientを評価に選び、登録の後の統合が起きにくい

- 発見日: 2026-10-23。対象: `src/federated_learning_experiments/runtime/model_cross_evaluation.py`（対象のモデルを保有するclientから最大3つを無作為に選ぶ。`_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL = 3`、`runtime/fedsda_final_configuration_run_settings.py`）、`runtime/fedsda_run_client.py`の`FedsdaRunClient.get_cross_evaluation_held_model_ids`。発見の経緯: SEA4の概念2の取り込みの調査（[NEW-006](new-006-minimum-description-length-keeps-partially-changed-concept-in-current-model.md)）と、統合を遅らせる変種の結果（[IMPROVE-028](../improvement-candidates/improve-028-evidence-accumulating-model-consolidation.md)の1）。
- 種別: 旧から引き継いだ挙動（旧の`federated_drift_experiment/servers/clustering.py`の`_cross_evaluate`が、`clients/base.py`の`get_held_model_ids`——評価標本・学習データ・保有モデル・現在のモデルのどれかのID——で保有するclientを集め、`config.CROSS_EVAL_MAX_CLIENTS = 3`で無作為に選ぶ。新実装は同じ規則を移植した）。状態: 未対応（2026-10-23に記録）。

## 内容

クロス評価では、対象のモデルjについて、jを「保有する」clientから最大3つを無作為に選び、候補のモデルiを、そのclientのjの標本で評価させる。「保有する」には、配布されたパラメータだけを持ち、jの評価標本も学習データも持たないclientが含まれる。そうしたclientは、jの標本を0件しか返さない（評価標本が6件未満で、jが現在の学習帰属でもないとき）。

登録の同期では、jを保有するのは作ったclientだけなので、標本が必ずある。登録の後は、配布で多くのclientがjを保有するので、選ばれた3つの多くが標本を持たず、対の4つの損失の統計のどれかが下限（5件）に届かない。判定の値の無い対は、average linkageで統合の候補にならない。結果として、統合は、構造上ほぼ登録の同期でしか起きない。

## 頻度（2026-10-23、`method-comparison-v10`の`fedsda`の間隔100の各40 runのraw）

| データ | 評価したclientのうち、対象の標本が0件の割合（対象の年齢0／1〜4／5以上） | 判定の値を持てない（候補、対象）の割合（同） |
|---|---|---|
| MNIST2 | 0.00／0.41／0.07 | 0.00／0.17／0.00 |
| CIRCLE2 | 0.00／0.89／0.67 | 0.00／0.70／0.53 |
| SEA4 | 0.00／0.43／0.37 | 0.00／0.39／0.26 |
| SINE2 | 0.00／0.82／0.29 | 0.00／0.58／0.17 |

年齢は、対象のモデルの登録からの同期の回数。同じ概念の対で判定の値を持てたのは、ほとんどが年齢0（MNIST2 42/43、SEA4 24/24、CIRCLE2 51/55、SINE2 350/371）。

## 影響

- 登録の同期で統合されなかった同じ概念の重複は、残り続ける。CIRCLE2では、版10の既定でサーバが持つモデルの数が多く（同期の後の平均5.65）、通信が増えていた（[IMPROVE-026](../improvement-candidates/improve-026-server-clustering-by-description-length.md)の1）。
- 統合の判定を「登録の直後の1回」より後へ延ばす案（IMPROVE-028のA・遅らせる変種）は、この挙動のままでは働かない。
- 評価の通信（依頼と返信、配布）の一部は、標本0件の返信に使われている。

## 直す方法の案と、手元の確認（2026-10-23）

評価させるclientの候補を、対象の標本を返せるclient（評価標本が6件以上、または対象が現在の学習帰属で学習データが11件以上。clientの標本の選び方と同じ条件）に限る。研究ツールの`default-sample-holding-cross-evaluation`（基準版10の既定の上）で、SEA4・MNIST2・CIRCLE2・SINE2、間隔100、seed 0〜3（各5000標本）を既定と比べた（IMPROVE-028の2）。

- 精度は4データとも既定と同じ（平均−0.03〜+0.02pp）。
- CIRCLE2の最終のモデル数が8.75→2.25、下りが0.53倍（統合33→57回）。SINE2の下りは0.94倍。SEA4・MNIST2はほぼ変わらない。
- 4 seedだけなので暫定。本規模の確認のため、基準版11の既定の上の変種`product-default-sample-holding-cross-evaluation`を足した（比べる相手は`method-comparison-v11`の既定）。
