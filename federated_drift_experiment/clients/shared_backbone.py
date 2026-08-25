"""共有バックボーンと概念別ヘッドを使うFedSDAクライアント。"""

import copy
import random
import time

import torch
import torch.nn.functional as F

from .. import config
from ..gradient_surgery import (
    assign_flat_gradient,
    compare_gradient_updates,
    flatten_parameter_gradients,
    project_conflicting_gradients,
    summarize_gradient_conflicts,
)
from .fedsda import (
    ClassConditionalESRFedSDAClient,
    RestartingSoftRoutingClassConditionalADWINFedSDAClient,
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from ..model_distillation import (
    DistillationDifferenceStats,
    DistillationUpdate,
)


class _SharedRepresentationFedSDAClientMixin:
    """FedSDAクライアントへ共有表現の管理と共同学習を追加するmixin。

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
        }
        # 蒸留の学習用標本と集約後検証用標本を混ぜないため、候補ごとに保持する。
        self._distillation_validation_cache = {}
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
        loss_sequence = ()
        if strategy in {
            "fifo_replay", "leader_change_replay",
            "persistent_leader_change_replay",
        }:
            loss_sequence = self._fifo_routing_loss_sequence()
        if strategy == "aggregation_restart":
            self.expert_router.restart_after_aggregation()
        elif strategy == "fifo_replay":
            self.expert_router.replay_after_aggregation(loss_sequence)
        elif strategy == "leader_change_replay":
            self.expert_router.replay_after_aggregation_if_leader_changed(
                loss_sequence,
                preferred_id=self.current_model_id,
            )
        elif strategy == "persistent_leader_change_replay":
            self.expert_router.replay_after_aggregation_if_leader_persists(
                loss_sequence,
                preferred_id=self.current_model_id,
            )
        elif strategy != "none":
            raise ValueError(f"未知のルーティング再較正方式です: {strategy!r}")

        # 予測クラス別の証拠は集約後の表現に対して再評価できないため破棄する。
        # 大域ルータは従来どおりFIFO等で再較正し、次の文脈を決める役割を保つ。
        for router in getattr(self, "context_expert_routers", {}).values():
            router.restart_after_aggregation()
        for router in getattr(self, "shadow_meta_routers", {}).values():
            router.restart_after_aggregation()
        switching_router = getattr(self, "switching_expert_router", None)
        if switching_router is not None:
            if strategy == "fifo_replay":
                switching_router.replay_after_aggregation(loss_sequence)
            elif strategy != "none":
                switching_router.restart_after_aggregation()

    def _fifo_routing_loss_sequence(self):
        """集約後の全保持モデルをFIFO上で再評価し、時系列損失を返す。"""
        samples = tuple(self.buffer)
        model_ids = tuple(sorted(self.models))
        if not samples or len(model_ids) <= 1:
            return ()

        x = torch.cat([sample[0] for sample in samples])
        y = torch.cat([sample[1] for sample in samples])
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

    def distillation_sample_count(self, model_ids):
        """候補クラスタについてローカル蒸留へ利用可能な標本数を返す。"""
        return sum(
            min(len(self.train_data_store.get(model_id, ())), config.EVAL_MAX_SAMPLES)
            for model_id in model_ids
        )

    def _distillation_teacher_probabilities(self, model_ids):
        """現在のglobal AdaHedge重みを候補クラスタ内で再正規化する。"""
        model_ids = tuple(sorted(model_ids))
        if not hasattr(self, "expert_router"):
            weight = 1.0 / len(model_ids)
            return {model_id: weight for model_id in model_ids}

        held_ids = tuple(sorted(self.models))
        if not set(model_ids).issubset(held_ids):
            weight = 1.0 / len(model_ids)
            return {model_id: weight for model_id in model_ids}

        # 診断・蒸留がオンラインルータの状態を変更しないよう、複製上で同期する。
        router = copy.deepcopy(self.expert_router)
        all_probabilities = router.probabilities(held_ids)
        total = sum(all_probabilities[model_id] for model_id in model_ids)
        if total <= 0.0:
            weight = 1.0 / len(model_ids)
            return {model_id: weight for model_id in model_ids}
        return {
            model_id: all_probabilities[model_id] / total
            for model_id in model_ids
        }

    def _split_distillation_data(self, model_ids):
        """各概念の標本を等分し、student学習と最終検証を分離する。"""
        training = []
        validation = {}
        for model_id in sorted(model_ids):
            data = list(self.train_data_store.get(model_id, ()))
            if len(data) < 4:
                continue
            if len(data) > config.EVAL_MAX_SAMPLES:
                data = random.sample(data, config.EVAL_MAX_SAMPLES)
            else:
                random.shuffle(data)
            midpoint = len(data) // 2
            if midpoint < 2 or len(data) - midpoint < 2:
                continue
            training.extend(data[:midpoint])
            validation[model_id] = data[midpoint:]
        return training, validation

    def _distillation_models(self, params_by_model):
        models = {}
        for model_id, params in params_by_model.items():
            model = self._new_model()
            model.set_params(params)
            models[model_id] = model
        return models

    @staticmethod
    def _teacher_mixture_from_features(
        teacher_models, probabilities, features, num_classes,
    ):
        scores = []
        for model_id, model in teacher_models.items():
            prediction = model.forward_from_features(features)
            if num_classes > 2:
                prediction = torch.softmax(prediction, dim=1)
            scores.append(prediction * probabilities[model_id])
        return sum(scores)

    @staticmethod
    def _distillation_loss(student_scores, teacher_scores, num_classes):
        if num_classes == 2:
            return F.binary_cross_entropy(student_scores, teacher_scores)
        return -(
            teacher_scores * F.log_softmax(student_scores, dim=1)
        ).sum(dim=1).mean()

    @staticmethod
    def _bounded_prediction_loss(scores, labels, num_classes):
        if num_classes == 2:
            return torch.abs(scores.view(-1) - labels.view(-1).float())
        probabilities = torch.softmax(scores, dim=1)
        class_ids = labels.view(-1).long()
        return 1.0 - probabilities.gather(
            1, class_ids.unsqueeze(1)
        ).squeeze(1)

    @staticmethod
    def _bounded_teacher_loss(probabilities, labels, num_classes):
        if num_classes == 2:
            return torch.abs(
                probabilities.view(-1) - labels.view(-1).float()
            )
        class_ids = labels.view(-1).long()
        return 1.0 - probabilities.gather(
            1, class_ids.unsqueeze(1)
        ).squeeze(1)

    def prepare_distillation_update(
        self, job_id, teacher_params_by_model, representative_model_id,
    ):
        """候補モデル混合をteacherとして概念別adapter/headをローカル学習する。"""
        model_ids = tuple(sorted(teacher_params_by_model))
        training_data, validation_data = self._split_distillation_data(model_ids)
        if len(training_data) < 2 or not validation_data:
            return None

        teacher_models = self._distillation_models(teacher_params_by_model)
        probabilities = self._distillation_teacher_probabilities(model_ids)
        student = self._new_model()
        student.set_params(teacher_params_by_model[representative_model_id])
        student.reset_optimizer()

        training_x = torch.cat([item[0] for item in training_data])
        with torch.no_grad():
            features = student.extract_features(training_x)
            teacher_scores = self._teacher_mixture_from_features(
                teacher_models, probabilities, features, student.num_classes,
            ).detach()
        self._record_model_compute(
            "training", len(training_x) * len(model_ids), calls=len(model_ids),
            backbone_examples=len(training_x),
            head_examples=len(training_x) * len(model_ids),
        )

        dataset = torch.utils.data.TensorDataset(training_x, teacher_scores)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(config.CLIENT_BATCH_SIZE, len(dataset)),
            shuffle=True,
        )
        start_time = time.perf_counter()
        for _ in range(self.new_model_initial_epochs()):
            for batch_x, batch_teacher in loader:
                with torch.no_grad():
                    batch_features = student.extract_features(batch_x)
                student.head_optimizer.zero_grad()
                student_scores = student.forward_from_features(batch_features)
                loss = self._distillation_loss(
                    student_scores, batch_teacher, student.num_classes
                )
                loss.backward()
                student.head_optimizer.step()
                self._record_model_compute(
                    "training", len(batch_x),
                    backbone_examples=len(batch_x), head_examples=len(batch_x),
                )
                self.compute_counters["optimizer_steps"] += 1
                self.compute_counters["head_optimizer_steps"] += 1
        self.phase_seconds["training"] += time.perf_counter() - start_time

        _, personalized = student.split_params(student.get_params())
        self._distillation_validation_cache[job_id] = (
            validation_data, probabilities
        )
        return DistillationUpdate(
            personalized_params=personalized,
            sample_count=len(training_x),
        )

    def evaluate_distilled_student(
        self, job_id, teacher_params_by_model, student_params,
    ):
        """集約後studentと元teacher混合を未使用のローカル標本で比較する。"""
        cached = self._distillation_validation_cache.pop(job_id, None)
        if cached is None:
            return DistillationDifferenceStats(by_target_model={})
        validation_data, probabilities = cached
        teacher_models = self._distillation_models(teacher_params_by_model)
        student = self._new_model()
        student.set_params(student_params)
        num_classes = student.num_classes
        by_target = {}
        start_time = time.perf_counter()
        for target_model_id, samples in validation_data.items():
            x = torch.cat([item[0] for item in samples])
            y = torch.cat([item[1] for item in samples])
            with torch.no_grad():
                features = student.extract_features(x)
                teacher_scores = self._teacher_mixture_from_features(
                    teacher_models, probabilities, features, num_classes,
                )
                student_scores = student.forward_from_features(features)
                teacher_losses = self._bounded_teacher_loss(
                    teacher_scores, y, num_classes
                )
                student_losses = self._bounded_prediction_loss(
                    student_scores, y, num_classes
                )
                differences = student_losses - teacher_losses
            self._record_model_compute(
                "cross_evaluation", len(x) * (len(teacher_models) + 1),
                calls=len(teacher_models) + 1,
                backbone_examples=len(x),
                head_examples=len(x) * (len(teacher_models) + 1),
            )
            by_target[target_model_id] = (
                len(differences),
                float(differences.sum().item()),
                float((differences ** 2).sum().item()),
            )
        self.phase_seconds["cross_evaluation"] += time.perf_counter() - start_time
        return DistillationDifferenceStats(by_target_model=by_target)

    def discard_distillation_job(self, job_id):
        """通信・学習に失敗した候補のローカル検証状態を破棄する。"""
        self._distillation_validation_cache.pop(job_id, None)

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

    def _sample_training_batches(self):
        """一回の共同更新へ参加できるヘッドとミニバッチを抽出する。"""
        batches = []
        for model_id, data_list in self.train_data_store.items():
            if model_id not in self.models or len(data_list) < self.batch_size:
                continue
            batch = random.sample(data_list, self.batch_size)
            batches.append((
                model_id,
                torch.cat([sample[0] for sample in batch]),
                torch.cat([sample[1] for sample in batch]),
            ))
        return batches

    def _train_heads_together(self, count_multiplier, update_backbone):
        """全参加ヘッドの損失を平均し、共有部の更新を最大1回にまとめる。"""
        start_time = time.perf_counter()
        updates_needed = self.updates_per_sample * count_multiplier
        for _ in range(updates_needed):
            batches = self._sample_training_batches()
            if not batches:
                continue

            backbone = self._shared_backbone()
            backbone.optimizer.zero_grad()
            for model_id, _, _ in batches:
                self.models[model_id].head_optimizer.zero_grad()

            all_x = torch.cat([bx for _, bx, _ in batches])
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
            for model_id, bx, by in batches:
                batch_examples = len(bx)
                features = all_features[offset:offset + batch_examples]
                loss = self.models[model_id].loss_from_features(features, by)
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
            gradient_strategy = config.SHARED_BACKBONE_GRADIENT_STRATEGY
            if (
                update_backbone
                and gradient_strategy not in {"mean", "pcgrad"}
            ):
                raise ValueError(
                    "未知の共有バックボーン勾配統合方式です: "
                    f"{gradient_strategy!r}"
                )
            if update_backbone and len(losses) > 1:
                gradient_records = sorted(
                    (
                        (model_id, loss, count)
                        for (model_id, _, _), loss, count in zip(
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
                applied_vectors = (
                    project_conflicting_gradients(gradient_vectors)
                    if gradient_strategy == "pcgrad"
                    else gradient_vectors
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

                reference_combined_gradient = sum(
                    vector * (count / total_examples)
                    for vector, count in zip(
                        gradient_vectors, gradient_example_counts
                    )
                )
                applied_combined_gradient = sum(
                    vector * (count / total_examples)
                    for vector, count in zip(
                        applied_vectors, gradient_example_counts
                    )
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
                if gradient_strategy == "pcgrad":
                    if applied_combined_gradient is not None:
                        assign_flat_gradient(
                            backbone_parameters, applied_combined_gradient
                        )
                backbone.optimizer.step()
                self.compute_counters["backbone_optimizer_steps"] += 1
            for model_id, bx, _ in batches:
                self.models[model_id].head_optimizer.step()
                self.model_training_examples[model_id] += len(bx)
                self.model_optimizer_steps[model_id] += 1

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


class ResidualAdapterClassADWINRestartingSoftRoutingFedSDAClient(
    _SharedRepresentationFedSDAClientMixin,
    RestartingSoftRoutingClassConditionalADWINFedSDAClient,
):
    """低ランク残差adapter、ClassADWIN、Restarting SoftRoutingを使う。"""
