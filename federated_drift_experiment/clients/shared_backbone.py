"""共有バックボーンと概念別ヘッドを使うFedSDAクライアント。"""

from dataclasses import dataclass
import random
import time

import torch

from .. import config
from ..gradient_surgery import (
    assign_flat_gradient,
    compare_gradient_updates,
    flatten_parameter_gradients,
    project_conflicting_gradients,
    select_gradient_by_validation,
    summarize_gradient_conflicts,
)
from .fedsda import (
    ClassConditionalESRFedSDAClient,
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)


@dataclass(frozen=True)
class _JointTrainingBatch:
    """共同更新用の学習標本と、重複しない直近検証標本。"""

    model_id: int
    x: torch.Tensor
    y: torch.Tensor
    validation_x: torch.Tensor | None = None
    validation_y: torch.Tensor | None = None


class _SharedRepresentationFedSDAClientMixin:
    """ClassESRクライアントへ共有表現の管理と共同学習を追加するmixin。

    正式採用済みモデルは一つの特徴抽出部を共有し、概念別ヘッドだけを独立して
    保持する。仮モデルと比較用shadowは独立したバックボーンで学習し、棄却時に
    既存モデルへ副作用を残さない。採用時だけ学習済み共有部を反映する。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.model_cls, "is_shared_backbone_model", False):
            raise TypeError("共有表現modeには共有部を持つモデルが必要です")
        self.backbone_gradient_diagnostics = {
            "pair_count": 0,
            "conflict_count": 0,
            "cosine_sum": 0.0,
            "negative_cosine_sum": 0.0,
            "applied_pair_count": 0,
            "applied_conflict_count": 0,
            "applied_cosine_sum": 0.0,
            "applied_negative_cosine_sum": 0.0,
            "update_comparison_count": 0,
            "update_cosine_sum": 0.0,
            "update_norm_ratio_sum": 0.0,
            "update_delta_ratio_sum": 0.0,
            "validation_selection_count": 0,
            "validation_pcgrad_selection_count": 0,
            "validation_mean_selection_count": 0,
            "validation_fallback_count": 0,
            "validation_mean_alignment_sum": 0.0,
            "validation_pcgrad_alignment_sum": 0.0,
            "validation_selected_margin_sum": 0.0,
        }
        self._share_model_backbones()

    def _shared_backbone(self):
        if self.current_model_id in self.models:
            return self.models[self.current_model_id].backbone
        return next(iter(self.models.values())).backbone

    def _share_model_backbones(self):
        """配布後に全保持モデルを一つの共有部へ再接続する。"""
        if not self.models:
            return
        global_ids = sorted(model_id for model_id in self.models if model_id >= 0)
        source_id = global_ids[0] if global_ids else next(iter(self.models))
        shared = self.models[source_id].backbone
        for model_id, model in self.models.items():
            if model_id == source_id:
                continue
            model.attach_backbone(shared)

    def _prepare_model_for_registration(self, model):
        """採用された候補の表現学習を共有部へ反映し、ヘッドを接続する。"""
        shared = self._shared_backbone()
        shared.load_state_dict(model.backbone.state_dict())
        model.attach_backbone(shared)
        return model

    def _after_models_rebuilt(self):
        self._share_model_backbones()

    def recalibrate_routing_after_aggregation(self):
        """集約による共有表現の変化後にSoftRoutingを再較正する。"""
        # hard routingには累積ルーティング状態がないため再較正は不要。
        if not hasattr(self, "expert_router"):
            return
        strategy = config.SHARED_BACKBONE_ROUTING_RECALIBRATION
        if strategy == "aggregation_restart":
            self.expert_router.restart_after_aggregation()
        elif strategy == "fifo_replay":
            self.expert_router.replay_after_aggregation(
                self._fifo_routing_loss_sequence()
            )
        elif strategy == "leader_change_replay":
            self.expert_router.replay_after_aggregation_if_leader_changed(
                self._fifo_routing_loss_sequence(),
                preferred_id=self.current_model_id,
            )
        elif strategy == "persistent_leader_change_replay":
            self.expert_router.replay_after_aggregation_if_leader_persists(
                self._fifo_routing_loss_sequence(),
                preferred_id=self.current_model_id,
            )
        elif strategy != "none":
            raise ValueError(f"未知のルーティング再較正方式です: {strategy!r}")

    def _fifo_routing_loss_sequence(self):
        """集約後の全保持モデルをFIFO上で再評価し、時系列損失を返す。"""
        samples = tuple(self.buffer)
        model_ids = tuple(sorted(self.models))
        if not samples or len(model_ids) <= 1:
            return ()

        x = torch.cat([sample_x for sample_x, _ in samples])
        y = torch.cat([sample_y for _, sample_y in samples])
        losses_by_model = {}
        with torch.no_grad():
            features = self.models[model_ids[0]].extract_features(x)
            for model_id in model_ids:
                model = self.models[model_id]
                scores = model.forward_from_features(features)
                if model.num_classes > 2:
                    probabilities = torch.softmax(scores, dim=1)
                    labels = y.view(-1).long()
                    losses = 1.0 - probabilities.gather(
                        1, labels.unsqueeze(1)
                    ).squeeze(1)
                else:
                    losses = torch.abs(
                        scores.view(-1) - y.view(-1).float()
                    )
                losses_by_model[model_id] = losses

        sample_count = len(samples)
        self._record_model_compute(
            "routing_recalibration",
            sample_count * len(model_ids),
            calls=len(model_ids),
            backbone_examples=sample_count,
            head_examples=sample_count * len(model_ids),
        )
        return tuple(
            {
                model_id: float(losses_by_model[model_id][index].item())
                for model_id in model_ids
            }
            for index in range(sample_count)
        )

    def _routing_scores(self, x, model_ids):
        """特徴抽出を1回だけ行い、全概念別ヘッドを評価する。"""
        first = self.models[model_ids[0]]
        features = first.extract_features(x)
        scores = {
            model_id: self.models[model_id].forward_from_features(features)
            for model_id in model_ids
        }
        self._record_model_compute(
            "prediction",
            len(x) * len(model_ids),
            calls=len(model_ids),
            backbone_examples=len(x),
            head_examples=len(x) * len(model_ids),
        )
        return scores

    def train_all_held_models(self, count_multiplier=1):
        """設定された共有表現の更新方式で、データを持つ全ヘッドを学習する。"""
        strategy = config.SHARED_BACKBONE_TRAINING
        if strategy == "sequential":
            before_steps = self.compute_counters["optimizer_steps"]
            super().train_all_held_models(count_multiplier=count_multiplier)
            logical_steps = self.compute_counters["optimizer_steps"] - before_steps
            self.compute_counters["backbone_optimizer_steps"] += logical_steps
            self.compute_counters["head_optimizer_steps"] += logical_steps
            return
        if strategy not in {"joint", "frozen"}:
            raise ValueError(f"未知の共有バックボーン学習方式です: {strategy!r}")
        self._train_heads_together(
            count_multiplier=count_multiplier,
            update_backbone=(strategy == "joint"),
        )

    def _record_independent_shared_model_steps(self, steps_before):
        """仮モデル等の独立した共有構造モデルについて、構成要素別更新数を補う。"""
        steps = self.compute_counters["optimizer_steps"] - steps_before
        self.compute_counters["backbone_optimizer_steps"] += steps
        self.compute_counters["head_optimizer_steps"] += steps

    def _update_new_model_epochs(self, model, dataset, epochs):
        steps_before = self.compute_counters["optimizer_steps"]
        super()._update_new_model_epochs(model, dataset, epochs)
        self._record_independent_shared_model_steps(steps_before)

    def _update_tournament_shadows(self, session, x, y):
        steps_before = self.compute_counters["optimizer_steps"]
        super()._update_tournament_shadows(session, x, y)
        self._record_independent_shared_model_steps(steps_before)

    def _sample_training_batches(self, include_validation=False):
        """共同更新用バッチと、必要なら重複しない直近検証バッチを作る。

        検証側は概念ストアの末尾から取得し、学習側の乱数列を変えない。学習と
        同数の検証標本を確保できないヘッドが一つでもあれば、呼出側でmeanへ
        フォールバックする。
        """
        batches = []
        for model_id, data_list in self.train_data_store.items():
            if model_id not in self.models or len(data_list) < self.batch_size:
                continue
            training_samples = random.sample(data_list, self.batch_size)
            validation_samples = None
            if include_validation and len(data_list) >= 2 * self.batch_size:
                training_ids = {id(sample) for sample in training_samples}
                held_out = [
                    sample for sample in reversed(data_list)
                    if id(sample) not in training_ids
                ][:self.batch_size]
                if len(held_out) == self.batch_size:
                    validation_samples = held_out
            batches.append(_JointTrainingBatch(
                model_id=model_id,
                x=torch.cat([sample[0] for sample in training_samples]),
                y=torch.cat([sample[1] for sample in training_samples]),
                validation_x=(
                    torch.cat([sample[0] for sample in validation_samples])
                    if validation_samples is not None else None
                ),
                validation_y=(
                    torch.cat([sample[1] for sample in validation_samples])
                    if validation_samples is not None else None
                ),
            ))
        return batches

    @staticmethod
    def _combine_gradients(vectors, example_counts):
        total_examples = sum(example_counts)
        return sum(
            vector * (count / total_examples)
            for vector, count in zip(vectors, example_counts)
        )

    def _validation_gradient(self, batches, backbone, backbone_parameters):
        """各概念の重複しない直近バッチから共有部の検証勾配を求める。"""
        if any(batch.validation_x is None for batch in batches):
            return None

        ordered = sorted(batches, key=lambda batch: batch.model_id)
        all_x = torch.cat([batch.validation_x for batch in ordered])
        features = backbone(all_x)
        weighted_losses = []
        total_examples = 0
        offset = 0
        for batch in ordered:
            count = len(batch.validation_x)
            model_features = features[offset:offset + count]
            loss = self.models[batch.model_id].loss_from_features(
                model_features, batch.validation_y
            )
            weighted_losses.append(loss * count)
            total_examples += count
            offset += count
        validation_loss = sum(weighted_losses) / total_examples
        gradient = flatten_parameter_gradients(
            torch.autograd.grad(
                validation_loss,
                backbone_parameters,
                allow_unused=True,
            ),
            backbone_parameters,
        )
        self._record_model_compute(
            "gradient_validation",
            total_examples,
            calls=len(ordered),
            backbone_examples=total_examples,
            head_examples=total_examples,
        )
        return gradient

    def _select_applied_gradients(
        self,
        strategy,
        gradient_vectors,
        example_counts,
        batches,
        backbone,
        backbone_parameters,
    ):
        """設定方式に従い、optimizerへ渡す概念別・統合勾配を返す。"""
        mean_combined = self._combine_gradients(
            gradient_vectors, example_counts
        )
        if strategy == "mean":
            return gradient_vectors, mean_combined

        projected_vectors = project_conflicting_gradients(gradient_vectors)
        pcgrad_combined = self._combine_gradients(
            projected_vectors, example_counts
        )
        if strategy == "pcgrad":
            return projected_vectors, pcgrad_combined

        validation_gradient = self._validation_gradient(
            batches, backbone, backbone_parameters
        )
        if validation_gradient is None:
            self.backbone_gradient_diagnostics[
                "validation_fallback_count"
            ] += 1
            return gradient_vectors, mean_combined

        selection = select_gradient_by_validation(
            mean_combined, pcgrad_combined, validation_gradient
        )
        if selection is None:
            self.backbone_gradient_diagnostics[
                "validation_fallback_count"
            ] += 1
            return gradient_vectors, mean_combined

        diagnostics = self.backbone_gradient_diagnostics
        diagnostics["validation_selection_count"] += 1
        diagnostics[f"validation_{selection.strategy}_selection_count"] += 1
        diagnostics["validation_mean_alignment_sum"] += (
            selection.mean_alignment
        )
        diagnostics["validation_pcgrad_alignment_sum"] += (
            selection.pcgrad_alignment
        )
        diagnostics["validation_selected_margin_sum"] += abs(
            selection.pcgrad_alignment - selection.mean_alignment
        )
        return (
            (projected_vectors, pcgrad_combined)
            if selection.strategy == "pcgrad"
            else (gradient_vectors, mean_combined)
        )

    def _train_heads_together(self, count_multiplier, update_backbone):
        """全参加ヘッドの損失を平均し、共有部の更新を最大1回にまとめる。"""
        start_time = time.perf_counter()
        updates_needed = self.updates_per_sample * count_multiplier
        for _ in range(updates_needed):
            gradient_strategy = config.SHARED_BACKBONE_GRADIENT_STRATEGY
            if (
                update_backbone
                and gradient_strategy not in {
                    "mean", "pcgrad", "heldout_selected",
                }
            ):
                raise ValueError(
                    "未知の共有バックボーン勾配統合方式です: "
                    f"{gradient_strategy!r}"
                )
            batches = self._sample_training_batches(
                include_validation=(
                    update_backbone
                    and gradient_strategy == "heldout_selected"
                )
            )
            if not batches:
                continue

            backbone = self._shared_backbone()
            backbone.optimizer.zero_grad()
            for batch in batches:
                self.models[batch.model_id].head_optimizer.zero_grad()

            all_x = torch.cat([batch.x for batch in batches])
            if update_backbone:
                all_features = backbone(all_x)
            else:
                # 凍結診断では共有表現を変更せず、ヘッドだけを適応させる。
                with torch.no_grad():
                    all_features = backbone(all_x)

            losses = []
            weighted_losses = []
            example_counts = []
            offset = 0
            total_examples = 0
            for batch in batches:
                batch_examples = len(batch.x)
                features = all_features[offset:offset + batch_examples]
                loss = self.models[batch.model_id].loss_from_features(
                    features, batch.y
                )
                losses.append(loss)
                weighted_losses.append(loss * batch_examples)
                example_counts.append(batch_examples)
                offset += batch_examples
                total_examples += batch_examples

            joint_loss = sum(weighted_losses) / total_examples
            backbone_parameters = tuple(backbone.parameters())
            gradient_vectors = []
            gradient_example_counts = []
            applied_combined_gradient = None
            if update_backbone and len(losses) > 1:
                gradient_records = sorted(
                    (
                        (batch.model_id, loss, count)
                        for batch, loss, count in zip(
                            batches, losses, example_counts
                        )
                    ),
                    key=lambda record: record[0],
                )
                gradient_vectors = [
                    flatten_parameter_gradients(
                        torch.autograd.grad(
                            loss,
                            backbone_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        ),
                        backbone_parameters,
                    )
                    for _, loss, _ in gradient_records
                ]
                gradient_example_counts = [
                    count for _, _, count in gradient_records
                ]
                summary = summarize_gradient_conflicts(gradient_vectors)
                self.backbone_gradient_diagnostics["pair_count"] += (
                    summary.pair_count
                )
                self.backbone_gradient_diagnostics["conflict_count"] += (
                    summary.conflict_count
                )
                self.backbone_gradient_diagnostics["cosine_sum"] += (
                    summary.cosine_sum
                )
                self.backbone_gradient_diagnostics["negative_cosine_sum"] += (
                    summary.negative_cosine_sum
                )
                applied_vectors, applied_combined_gradient = (
                    self._select_applied_gradients(
                        gradient_strategy,
                        gradient_vectors,
                        gradient_example_counts,
                        batches,
                        backbone,
                        backbone_parameters,
                    )
                )
                applied_summary = summarize_gradient_conflicts(applied_vectors)
                self.backbone_gradient_diagnostics["applied_pair_count"] += (
                    applied_summary.pair_count
                )
                self.backbone_gradient_diagnostics[
                    "applied_conflict_count"
                ] += applied_summary.conflict_count
                self.backbone_gradient_diagnostics["applied_cosine_sum"] += (
                    applied_summary.cosine_sum
                )
                self.backbone_gradient_diagnostics[
                    "applied_negative_cosine_sum"
                ] += applied_summary.negative_cosine_sum

                reference_combined_gradient = self._combine_gradients(
                    gradient_vectors, gradient_example_counts
                )
                comparison = compare_gradient_updates(
                    reference_combined_gradient,
                    applied_combined_gradient,
                )
                if comparison is not None:
                    self.backbone_gradient_diagnostics[
                        "update_comparison_count"
                    ] += 1
                    self.backbone_gradient_diagnostics["update_cosine_sum"] += (
                        comparison.cosine
                    )
                    self.backbone_gradient_diagnostics[
                        "update_norm_ratio_sum"
                    ] += comparison.norm_ratio
                    self.backbone_gradient_diagnostics[
                        "update_delta_ratio_sum"
                    ] += comparison.delta_ratio
            joint_loss.backward()
            if update_backbone:
                if gradient_strategy in {"pcgrad", "heldout_selected"}:
                    if applied_combined_gradient is not None:
                        assign_flat_gradient(
                            backbone_parameters, applied_combined_gradient
                        )
                backbone.optimizer.step()
                self.compute_counters["backbone_optimizer_steps"] += 1
            for batch in batches:
                self.models[batch.model_id].head_optimizer.step()
                self.model_training_examples[batch.model_id] += len(batch.x)
                self.model_optimizer_steps[batch.model_id] += 1

            head_steps = len(batches)
            self.compute_counters["head_optimizer_steps"] += head_steps
            # 従来指標は論理的な概念モデル更新回数として維持する。
            self.compute_counters["optimizer_steps"] += head_steps
            self._record_model_compute(
                "training",
                total_examples,
                calls=head_steps,
                backbone_examples=total_examples,
                head_examples=total_examples,
            )
        self.phase_seconds["training"] += time.perf_counter() - start_time


class SharedBackboneClassConditionalESRFedSDAClient(
    _SharedRepresentationFedSDAClientMixin,
    ClassConditionalESRFedSDAClient,
):
    """共有バックボーンとhard routingを使うClassESRクライアント。"""


class SharedBackboneRestartingSoftRoutingFedSDAClient(
    _SharedRepresentationFedSDAClientMixin,
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
):
    """共有バックボーンとRestarting SoftRoutingを使うClassESRクライアント。"""


class ResidualAdapterClassConditionalESRFedSDAClient(
    SharedBackboneClassConditionalESRFedSDAClient
):
    """概念別低ランク残差adapterとhard routingを使うClassESRクライアント。"""


class ResidualAdapterRestartingSoftRoutingFedSDAClient(
    SharedBackboneRestartingSoftRoutingFedSDAClient
):
    """ゼロ初期化の概念別低ランク残差adapterを使うClassESRクライアント。"""
