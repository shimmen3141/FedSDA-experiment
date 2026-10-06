# 設計: 準備済み分類器の標本別有界損失

## Overview
準備済み一分類器を一回呼び出し、観測済みラベルへの標本別有界誤差を生成する。学習のBCE/CE・予測済み確率の平均損失とは入力と出力の契約が異なる。
Goals: 旧per_sample_errorの演算順・標本順・数値を維持し、既存batch初期統計へ渡せる結果を返す。
Non-Goals: 登録全体、学習/optimizer、採否、ID/通信、新client/全体run。

## Boundary Commitments
### This Spec Owns
分類器とbatchの契約検証、勾配記録なしの一forward、結果検証、標本別有界損失生成。
### Out of Boundary
モデル生成/clone/prepare/登録、optimizer生成/reset/step、統計計算/保存、ID、候補判断、送信snapshot/counter、mixing、runtime/I/O。出力から初期統計を作る接続はtest-onlyで明示する。
通常のResidualAdapterClassifier構造が前提。外側で仕込まれた副作用hook・構造改ざん・並行変更のrollbackを保証しない。モデル/入力が不適合なら通常の検証例外を返す。任意forward内部例外は握りつぶさず伝播する。
### Allowed Dependencies
productionのfrom importはexact symbolだけを許可する: torch.Tensor/abs/float32/isfinite/no_grad/softmax/strided/trunc、learning.models.residual_adapter_classifier.ResidualAdapterClassifier。bare module import・re-export・private helperは拒否。
既存prediction計算、loss_statistics、registry、training、optimizer、methods、runtime、configuration、旧package、NumPy、stdlib/I/O/randomへ依存しない。
### Revalidation Triggers
loss式・演算順・shape/dtype/device・受理モデル・副作用契約が変われば、初期統計と今後の登録/候補比較/損失監視接続を再照合する。

## Architecture
上位→learning/predictionの評価関数→既存モデルとPyTorch公開演算。状態や上位のIDを持たず、分類器を学習損失計算のため改変しない。新依存なし、既存固定Windows CPU環境を用いる。
既存class_probability_calculationsはラベル観測前のモデル出力を保持してから損失平均を作る用途である。架空ID付き辞書へ変換して接続せず、直接の旧batch評価を独立に移植する。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/prediction/classifier_bounded_loss_evaluation.py|新規|一分類器のbatch損失生成と検証|
|tests/refactoring/test_classifier_bounded_loss_evaluation.py|新規|旧実モデルの損失/登録統計対照、境界と副作用|
|tests/refactoring/test_single_run_dependency_boundaries.py|変更|exact symbol guardと注入検証|
|対象spec、.kiro/steering/roadmap.md、resume.md|新規/変更|承認、証拠、現在地|

## Components & Interfaces
### 評価関数
evaluate_classifier_per_sample_bounded_losses(*, classifier:ResidualAdapterClassifier, input_features:Tensor, observed_class_labels:Tensor)->Tensor。
_validate_classifier_bounded_loss_inputsは同じkeyword入力→None。
_validate_classifier_outputs_for_bounded_loss(*,classifier_outputs:Tensor,batch_sample_count:int,class_count:int)->None。

前提: exact ResidualAdapterClassifier、class_countはexact builtin int>=2。全parameterはCPU float32 strided/notnested。任意Tensor派生のinput/labelを受理するがCPU float32 strided/notnestedが必要。
input_featuresはshape[N,F]、N>0、F=classifier.feature_extractor.input_feature_count、全特徴有限。labelsはshape[N,1]、同じN、有限の整数値、0<=label<class_count。1D自動展開/型変換/空batchは受理しない。非連続strided入力は受理。
全入力検査がforward前。type不正はTypeError、環境/shape/value不正はValueErrorで項目名と理由の日本語message。class_count検査もforward前。
@no_grad()の関数境界でclassifier(input_features)を一回。eval()/train()/zero_grad()/detach入力/cloneモデルは行わない。通常モデルのRNG/default dtype/device/training flags/値/既存grad/入力を保持し、例外時にも呼出元grad設定を戻す。
forward戻り値classifier_outputsはTensor、CPU float32 strided/notnested、shape[N,1]（二値）/shape[N,K]（多クラス）、有限。二値のclassifier_outputsのみ0〜1を確認する。多クラスlogitへ範囲制約なし。公開評価関数の戻り値はこの分類出力ではなく、後述の標本別損失shape[N]である。
二値はabs(classifier_outputs-observed_class_labels).reshape(-1)。多クラスはsoftmax(dim=1)→labels.reshape(-1).long().unsqueeze(1)でgather→squeeze(1)→1.0−correct_class_probabilities。clamp/reduce/sortはしない。
新しい演算結果shape[N]のCPU float32を返す。入力/モデルのstorageと独立し、requires_grad=Falseでgrad_fnなし。@no_gradは呼出元gradmodeを復元する。通常のinference_mode内呼出しも状態を変更しない。

## Requirements Traceability / Testing Strategy
|要件|検証・対応|
|---|---|
|1.1,1.2,3.1|同一parameterの実旧ResidualAdapterMLP.per_sample_errorとのexact対照、class2/4/10、adapter展開非ゼロ、クラス欠落/混合・正誤ラベル|
|1.3|一標本・非連続入力/ラベル・逆標本順、結果shapeとexact対照|
|1.4|forward hookで実行1回、検査時0回|
|2.1|型、meta/device/dtype/sparse/nested、shape、空/件数/特徴数、非有限feature、非整数/範囲label、モデルparameter不正の先行拒否|
|2.2|出力のみ置換するhookで型/device/dtype/layout/shape/有限/二値範囲不正を拒否|
|2.3,2.4|成功/拒否の前後input/全parameter/既存gradient/flags/RNG/default/gradmode保持、勾配なし、結果変更の非波及|
|3.2|exact AST先行RED/許可拒否・通常経路の責務点検|
|3.3|固定旧/golden/test差分空、既存全pytest旧11/最終3golden成功。Ruff/format/Pyright/pip、fresh新CPU、hash/独立レビューを完了gateで記録|

明示統合test: class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件で、既存旧/新の共同更新→旧準備処理/新採用共有反映→損失評価→既存batch初期統計を接続する。旧BaseClient._register_trained_new_modelを準備済みモデルへidentity prepareで直接呼び、モデル値/損失/全統計field一致と評価がoptimizer stateを変更しないことを確認する。singleton fallbackも比較する。学習/prepare/統計はtest側の対応付けであり評価部へ持ち込まない。

## Error Handling / Risks
不正入力を値の補正で隠さない。極大の有限parameter/特徴から非有限logitが生じた場合は出力検証で明示拒否する。旧正常経路を変えず、旧異常動作の修正を同時に行わない。学習後評価は先に準備済み共有部を接続するという呼出順をtest-onlyで確認する。
