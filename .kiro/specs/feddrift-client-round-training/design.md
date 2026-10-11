# Design Document

## Overview
一モデルの数値更新をlearning、一ラウンドのモデル順/抽出/計数をruntimeへ分ける。外側が既存registry・training store・counts・Randomを渡す。新たな学習要求counterは作らない。

## Boundary Commitments
- `learning/training/independent_model_parameter_update.py`: 独立モデルの平均loss一回更新と更新前検査。
- `runtime/feddrift_client_round_training.py`: 標本store順のモデル別連続更新。completed lossを順に返す。
- 既存samplingを再利用する。R回のserver round、run participant factoryは後続。
- 保持標本は観測入口で検査済みだが、ラウンド開始時にも参加モデルの全populationをcatして検査する。O(保持件数)の検査費用と一時連結を受け入れ、後方不正による先行モデルだけの更新を防ぐ。非参加payloadは見ない。
- 予算0はowner/数値型検査後のno-op。参加がないときはgrad無効でもno-op。
- 同期的な単一呼出し。runtime内部の実更新例外の全round rollbackは保証しない。

## Components and Interfaces
`validate_independent_model_parameter_update_inputs`は独立architecture、CPU float32/有限parameter、requires_grad、exact Adam/SGDと全parameter参照、標本・ラベル/範囲、grad有効をforward/zero_grad前に検査する。既存bounded loss入力検査・snapshot・共同更新のparameter検査を使う。

`update_independent_model_parameters`は検査後、optimizer.zero_grad→classifier forward→BCEまたはCE→backward→step→float lossの順。共同損失のbatch件数による乗除を経由しない。modeを変更しない。

構造検査は構成設定だけに留めず、空feature extractor、ModelOwnedHiddenLayersのLinear/ReLU列の順序/shapeと宣言幅、分類層shape、クラス数に対応したSigmoid/Identityまで確認する。保持後に形状やクラス数が変わっても、勾配を消去する前に拒否する。

`train_feddrift_held_models_for_round`はexact registry/store/counts/Random、正batch件数、非負intのupdates_per_sampleとk_stepsを検査。store snapshotと保有stateから参加列を作る。全参加標本の型/一標本shape、state/optimizer対応と連結batchを先に検査し、その後モデル別に反復。既存sample_training_batches_for_held_modelsへ一モデルだけ渡し、得たbatchを単一モデル更新へ渡す。完了ごとにcountsへ加える。registryの順とstoreの順が異なってもstore順。学習中に標本列/registryを変更しない。

## Requirements Traceability
| 要求 | 部品・証拠 |
| --- | --- |
| 1.1–1.3 | 学習数値部品、実旧updateのloss/params/grads/Adam state対照、事前拒否 |
| 2.1–2.3 | runtime、実旧local_train対照、順序・抽出・RNG・計数・no-op |
| 3.1–3.2 | 全参加populationの事前検査、後方不正/非参加skip対照 |
| 3.3 | exact依存とfresh smoke |

## Testing Strategy
旧FedDriftClientの実__init__/local_trainと実SimpleMLP.updateをoracleにする。SGD/Adam×二値/多クラス×batch1/2/3、複数round、正/負ID、逆store順、欠落/不足、ゼロ予算、既存状態継続を照合。検査は全owner/パラメータ・勾配・optimizer・RNGを保持してforward/zero_grad前に拒否するtest。全pytestには旧/提案goldenも含める。
