"""共有バックボーンと概念別ヘッドを使うFedSDAクライアント。"""

import random
import time

import torch

from .. import config
from .fedsda import RestartingSoftRoutingClassConditionalESRFedSDAClient


class SharedBackboneRestartingSoftRoutingFedSDAClient(
    RestartingSoftRoutingClassConditionalESRFedSDAClient
):
    """ClassESR + RestartingSoftRoutingへ共有表現を追加したクライアント。

    正式採用済みモデルは一つの特徴抽出部を共有し、概念別ヘッドだけを独立して
    保持する。仮モデルと比較用shadowは独立したバックボーンで学習し、棄却時に
    既存モデルへ副作用を残さない。採用時だけ学習済み共有部を反映する。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.model_cls, "is_shared_backbone_model", False):
            raise TypeError("共有表現modeには共有部を持つモデルが必要です")
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

            weighted_losses = []
            offset = 0
            total_examples = 0
            for model_id, bx, by in batches:
                batch_examples = len(bx)
                features = all_features[offset:offset + batch_examples]
                loss = self.models[model_id].loss_from_features(features, by)
                weighted_losses.append(loss * batch_examples)
                offset += batch_examples
                total_examples += batch_examples

            joint_loss = sum(weighted_losses) / total_examples
            joint_loss.backward()
            if update_backbone:
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


class PartialSharedAdapterRestartingSoftRoutingFedSDAClient(
    SharedBackboneRestartingSoftRoutingFedSDAClient
):
    """部分共有表現と概念別adapterを使うClassESRクライアント。

    通信、集約、仮モデル、SoftRoutingの処理は共有表現クライアントと共通で、
    共有範囲と概念固有範囲の境界だけをモデル構造へ委譲する。
    """


class ResidualAdapterRestartingSoftRoutingFedSDAClient(
    SharedBackboneRestartingSoftRoutingFedSDAClient
):
    """ゼロ初期化の概念別低ランク残差adapterを使うClassESRクライアント。"""
