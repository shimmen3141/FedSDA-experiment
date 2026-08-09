"""共有バックボーンと概念別ヘッドを使うFedSDAクライアント。"""

import random
import time

import torch

from .. import config
from ..models import SharedBackboneMLP
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
        if self.model_cls is not SharedBackboneMLP:
            raise TypeError("共有バックボーンmodeにはSharedBackboneMLPが必要です")
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
