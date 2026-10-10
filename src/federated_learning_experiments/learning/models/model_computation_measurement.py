"""モデルの計算を、計測の区間の間、PyTorchのhookで外側から数える。

数えるもの:

- 共有部（`SharedFeatureExtractor`）と概念固有部（`NonlinearResidualAdapter`）の順伝播へ入力された標本数。
- 全結合層（`torch.nn.Linear`）の積和演算の数（標本数×入力の次元×出力の次元）。共有部の順伝播の中の層は
  共有部、それ以外の層は概念固有部。バイアスの加算、活性化関数、損失、optimizerの更新は数えない。
- optimizerの更新回数。

順伝播のときに勾配が有効なら学習、無効なら推論として数える。逆伝播は実測せず、勾配が有効な順伝播の
全結合層ごとに、層の形から見積って、順伝播とは別の項目に持つ。

hookはprocessの全体に効くので、区間の中で行われた、すべてのモデルの計算を数える。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from weakref import ReferenceType, ref

from torch import Tensor, is_grad_enabled
from torch.nn import Linear, Module
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.optim import Optimizer
from torch.optim.optimizer import register_optimizer_step_post_hook

from federated_learning_experiments.learning.models.nonlinear_residual_adapter import (
    NonlinearResidualAdapter,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)


@dataclass(frozen=True, kw_only=True)
class ModelComputationCounts:
    """モデルの計算の計数。

    `example_count`は、部品の順伝播へ入力された標本数。`multiply_accumulate_count`は、全結合層の
    積和演算の数で、`forward`は実測、`estimated_backward`は、勾配が有効な順伝播ごとの見積り
    （重みが勾配を求める層は、順伝播と同じ数。入力が勾配を求める層は、さらに同じ数）。
    """

    shared_part_training_example_count: int
    shared_part_inference_example_count: int
    concept_specific_part_training_example_count: int
    concept_specific_part_inference_example_count: int
    shared_parameter_optimizer_step_count: int
    concept_specific_parameter_optimizer_step_count: int
    shared_part_training_forward_multiply_accumulate_count: int
    shared_part_inference_forward_multiply_accumulate_count: int
    concept_specific_part_training_forward_multiply_accumulate_count: int
    concept_specific_part_inference_forward_multiply_accumulate_count: int
    shared_part_estimated_backward_multiply_accumulate_count: int
    concept_specific_part_estimated_backward_multiply_accumulate_count: int


def subtract_model_computation_counts(
    *, later_counts: ModelComputationCounts, earlier_counts: ModelComputationCounts
) -> ModelComputationCounts:
    """同じ計測の、2つの時点の計数の差（後の時点−前の時点）。"""
    if type(later_counts) is not ModelComputationCounts:
        raise TypeError("later_counts must be exact ModelComputationCounts")
    if type(earlier_counts) is not ModelComputationCounts:
        raise TypeError("earlier_counts must be exact ModelComputationCounts")
    count_differences = {
        count_field.name: getattr(later_counts, count_field.name)
        - getattr(earlier_counts, count_field.name)
        for count_field in fields(ModelComputationCounts)
    }
    if any(count_difference < 0 for count_difference in count_differences.values()):
        raise ValueError("later_counts must not be smaller than earlier_counts")
    return ModelComputationCounts(**count_differences)


def _count_input_examples(*, forward_input: Tensor) -> int:
    """最初の次元を標本数とする。1次元の入力は、標本1件。"""
    return 1 if forward_input.dim() == 1 else int(forward_input.shape[0])


class ModelComputationMeter:
    """計測の区間の間の計数を持つ。生成は`measure_model_computation`が行う。"""

    def __init__(self) -> None:
        self._counts = {count_field.name: 0 for count_field in fields(ModelComputationCounts)}
        # 順伝播の途中の共有部の数（共有部の中の全結合層を見分ける）。
        self._shared_part_forward_depth = 0
        # これまでに順伝播した共有部のパラメータ（idから弱参照。計測がモデルを生かし続けない）。
        # tensorは、等値の比較が要素ごとなので、集合の要素にせず、同一性で照合する。
        self._observed_shared_parameter_references: dict[int, ReferenceType[Tensor]] = {}

    def get_model_computation_counts(self) -> ModelComputationCounts:
        """その時点までの計数（後からの計算で変わらない値）。"""
        return ModelComputationCounts(**self._counts)

    def _observe_forward_start(self, module: Module, forward_inputs: tuple[object, ...]) -> None:
        if type(module) is SharedFeatureExtractor:
            self._shared_part_forward_depth += 1
            for shared_parameter in module.parameters():
                if not self._is_observed_shared_parameter(shared_parameter):
                    self._observed_shared_parameter_references[id(shared_parameter)] = ref(
                        shared_parameter
                    )

    def _is_observed_shared_parameter(self, parameter: Tensor) -> bool:
        """解放されたtensorと同じidの、別のtensorを、共有部のパラメータと取り違えない。"""
        parameter_reference = self._observed_shared_parameter_references.get(id(parameter))
        return parameter_reference is not None and parameter_reference() is parameter

    def _observe_forward_end(
        self, module: Module, forward_inputs: tuple[object, ...], forward_output: object
    ) -> None:
        if type(module) is SharedFeatureExtractor:
            self._shared_part_forward_depth -= 1
        # 例外で終わった順伝播（出力がない）は、数えない。
        if forward_output is None or not forward_inputs:
            return
        forward_input = forward_inputs[0]
        if not isinstance(forward_input, Tensor):
            return
        mode_name = "training" if is_grad_enabled() else "inference"
        if type(module) is SharedFeatureExtractor:
            self._counts[f"shared_part_{mode_name}_example_count"] += _count_input_examples(
                forward_input=forward_input
            )
            return
        if type(module) is NonlinearResidualAdapter:
            self._counts[f"concept_specific_part_{mode_name}_example_count"] += (
                _count_input_examples(forward_input=forward_input)
            )
            return
        if type(module) is not Linear:
            return
        # 全結合層: 標本数×入力の次元×出力の次元（入力の要素数×出力の次元）。
        part_name = (
            "shared_part" if self._shared_part_forward_depth > 0 else "concept_specific_part"
        )
        forward_multiply_accumulate_count = forward_input.numel() * module.out_features
        self._counts[f"{part_name}_{mode_name}_forward_multiply_accumulate_count"] += (
            forward_multiply_accumulate_count
        )
        if mode_name == "training":
            # 逆伝播の見積り: 重みの勾配と、入力へ戻す勾配。
            self._counts[f"{part_name}_estimated_backward_multiply_accumulate_count"] += (
                forward_multiply_accumulate_count
                * (int(module.weight.requires_grad) + int(forward_input.requires_grad))
            )

    def _observe_optimizer_step(
        self,
        parameter_optimizer: Optimizer,
        step_arguments: tuple[object, ...],
        step_keyword_arguments: dict[str, object],
    ) -> None:
        updates_shared_parameters = any(
            self._is_observed_shared_parameter(optimized_parameter)
            for parameter_group in parameter_optimizer.param_groups
            for optimized_parameter in parameter_group["params"]
        )
        part_name = "shared" if updates_shared_parameters else "concept_specific"
        self._counts[f"{part_name}_parameter_optimizer_step_count"] += 1


@contextmanager
def measure_model_computation() -> Iterator[ModelComputationMeter]:
    """計測の区間。入るときにhookを登録し、出るとき（例外を含む）に外す。

    渡すmeterは、区間の途中でも、区間を出た後でも、計数を読める。
    """
    model_computation_meter = ModelComputationMeter()
    hook_handles = (
        register_module_forward_pre_hook(model_computation_meter._observe_forward_start),
        # 順伝播が例外で終わっても呼ばれるようにして、共有部の深さを戻す。
        register_module_forward_hook(
            model_computation_meter._observe_forward_end, always_call=True
        ),
        register_optimizer_step_post_hook(model_computation_meter._observe_optimizer_step),
    )
    try:
        yield model_computation_meter
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()
