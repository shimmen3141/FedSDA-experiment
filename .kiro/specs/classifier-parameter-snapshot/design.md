# 設計: 分類器parameter snapshot

revision: 1

## Overview / Boundary Commitments
準備済みResidualAdapterClassifierから全parameterの現在値を独立コピーする。学習・共有部準備・候補選択・平均・適用・登録・統計・送信保留は呼出側の責務で、本部品は状態を持たない。
通常の既存モデル構造（bufferなし）を対象にする。外部hook、構造改ざん、並行変更は対象外。旧OrderedDict metadataや旧key読込みを導入しない。

## Architecture / Allowed Dependencies
上位→learning/modelsのsnapshot関数→既存ResidualAdapterClassifierとPyTorch公開演算。新依存なし。exact from importのみ: torch.Tensor/float32/isfinite/no_grad/strided、learning.models.residual_adapter_classifier.ResidualAdapterClassifier。bare import、private、re-export、stdlib、旧package、methods、training、optimizer、runtime、I/Oは拒否する。

## Components & Interfaces
`snapshot_classifier_parameters(*, classifier: ResidualAdapterClassifier) -> dict[str, Tensor]`。
`_validate_classifier_parameter_snapshot_inputs(*, classifier: ResidualAdapterClassifier) -> None`。

exact ResidualAdapterClassifierを受理する。全named_parametersについてCPU float32 strided/notnestedかつ有限を要求する。型不正はTypeError、parameter不適合はValueError。項目名と理由を日本語で返す。通常構造が前提なので構造診断器や代替モデル適応を追加しない。
全検証後、state_dict()を一回呼び、native key順で各値のdetach().clone()をplain dictへ格納する。通常構造にはbufferがなくstate_dictは全parameterだけを列挙する。値・shape・dtype/deviceを保持し、requires_grad=False/grad_fnなし。辞書と各storageはモデル/他呼出結果と独立する。非連続strided parameterも受理し、strideの同一性は要求しない。
@no_grad()で成功/拒否時の呼出元gradmodeを復元する。forward/eval/train/reset/step/zero_gradを呼ばず、既存gradとparameter参照・共有部参照・training flags・RNG/default環境を保持する。通常inference_modeも受理する。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/models/classifier_parameter_snapshot.py|新規|snapshot生成と入力検証|
|tests/refactoring/test_classifier_parameter_snapshot.py|新規|独立性・状態・異常契約・実旧snapshot対照・既存initializerへのtest-only接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|変更|exact import guardと注入検証|
|対象specとsteering resume/roadmap|新規/更新|承認と再開案内・証拠|

## Requirements Traceability / Testing Strategy
|要件|受け入れ証拠|
|---|---|
|1.1,3.1|実旧ResidualAdapterMLP.get_paramsとのnative prefix対応、class2/4/10、全key順/shape/値一致|
|1.2,1.3|出力storage独立、両方向変更と複数snapshot・モデル更新後の再取得|
|2.1|型/subclass/dtype/device/layout/nonfinite拒否、既存grad不変|
|2.2|forward実行0、全parameter/grad参照/値、共有参照、混合training flags、Python/NumPy/Torch乱数、default環境/gradmode保持|
|3.2|exact AST許可/拒否のRED→GREEN、新CPU fresh smokeで旧importなし|
|3.3|固定旧/golden/test差分空、全pytest（旧11/最終3golden含む）、Ruff/format/Pyright/pip/hash|

test-only統合: class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件で実旧/新共同更新後の全parameterを照合し、snapshotを既存候補initializerへ渡し、新分類器のnative load_state_dictで同一値を復元する。optimizer stateはsnapshot前後で保持する。登録/送信保留を本部品へ追加しない。

## Risks / Revalidation Triggers
buffer付き構造、モデル型、環境契約、key順や所有権を変更する場合は別途承認とinitializer/将来送信接続の再検証が必要。旧正常経路の値を維持し、異常経路の厳格検証でgoldenを書き換えない。
