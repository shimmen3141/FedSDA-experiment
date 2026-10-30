"""IMPROVE-033の13候補を経過の外で学習・評価する。製品は変更しない。"""

import random
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import torch

from federated_learning_experiments.data.observed_sample_generation import (
    create_observed_sample_generator,
)
from federated_learning_experiments.learning.models import (
    model_computation_measurement as computation,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
    evaluate_classifier_per_sample_log_losses,
)
from federated_learning_experiments.learning.training import (
    candidate_epoch_training as epochs,
)
from federated_learning_experiments.runtime import (
    fedsda_run_participant_factory as factory,
)
from federated_learning_experiments.runtime import single_run_execution as streams
from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
    _get_unique_majority_concept_ids,
)
from tools.research_provisional_candidate_expert import (
    candidate_initialization_follows_learner as initialization,
)

VARIANT = "product-default-candidate-start-diagnostics"
training = initialization.training
processing = initialization.processing
response = initialization.response
resolution = initialization.resolution
mdl = initialization.mdl
selection = initialization.selection


@contextmanager
def _model_computation_gate():
    """製品の計測開始前にhookを包み、診断中は計数も共有部参照も更新しない。"""
    paused = [False]

    def gated(original):
        def observe(self, *args, **kwargs):
            if not paused[0]:
                return original(self, *args, **kwargs)
            return None

        return observe

    with (
        initialization.patch.object(
            computation.ModelComputationMeter,
            "_observe_forward_start",
            gated(computation.ModelComputationMeter._observe_forward_start),
        ),
        initialization.patch.object(
            computation.ModelComputationMeter,
            "_observe_forward_end",
            gated(computation.ModelComputationMeter._observe_forward_end),
        ),
        initialization.patch.object(
            computation.ModelComputationMeter,
            "_observe_optimizer_step",
            gated(computation.ModelComputationMeter._observe_optimizer_step),
        ),
    ):
        yield paused


def _random_states():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def _restore(states):
    torch.set_rng_state(states[0])
    random.setstate(states[1])
    np.random.set_state(states[2])


def _seed(run_seed, client_id, alarm_sample_index, purpose):
    return int.from_bytes(
        sha256(
            f"{run_seed}:{client_id}:{alarm_sample_index}:{purpose}".encode()
        ).digest()[:4],
        "big",
    )


def _generate_samples(
    *, dataset, run_seed, client_id, alarm_sample_index, purpose, concepts
):
    """用途ごとの専用RandomState。グローバル乱数も例外時を含め復元する。"""
    before = _random_states()
    seed = _seed(run_seed, client_id, alarm_sample_index, purpose)
    try:
        generator = create_observed_sample_generator(
            dataset_name=dataset, numpy_random_generator=np.random.RandomState(seed)
        )
        samples = [generator.generate_sample(concept_id=int(c)) for c in concepts]
        return (
            torch.tensor([s.feature_values for s in samples], dtype=torch.float32),
            torch.tensor([[s.class_label] for s in samples], dtype=torch.float32),
            seed,
        )
    finally:
        _restore(before)


def _changed_classes(dataset, previous, current, class_count):
    if previous is None:
        return None
    if not dataset.startswith(("mnist", "fmnist")):
        return list(range(class_count))

    def mapping(concept):
        labels = list(range(class_count))
        if concept:
            a, b = 2 * concept - 1, 2 * concept
            labels[a], labels[b] = labels[b], labels[a]
        return labels

    return [
        i
        for i, (p, q) in enumerate(
            zip(mapping(previous), mapping(current), strict=True)
        )
        if p != q
    ]


def _parameter_digest(classifier):
    return initialization._digest(
        initialization.snapshot_classifier_parameters(classifier=classifier)
    )


def _require_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def _held_concept_evidence(clients, model_ids, alarm_concept, local_client_id=None):
    """既存の診断と同じく全clientの一意な最多割当概念を使う。同数は不明。"""
    counts, concepts = {}, {}
    for mid in model_ids:
        # 負の一時IDはclientをまたいで衝突する。正式IDだけ全clientを合算する。
        scope = (
            tuple(c for c in clients if c.client_id == local_client_id)
            if mid < 0
            else tuple(clients)
        )
        counts[mid] = dict(
            sum(
                (
                    Counter(c.get_model_assigned_sample_concept_counts(model_id=mid))
                    for c in scope
                ),
                Counter(),
            )
        )
        concepts.update(
            _get_unique_majority_concept_ids(run_clients=scope, model_ids=(mid,))
        )
    if factory._INITIAL_MODEL_ID in model_ids and not counts[factory._INITIAL_MODEL_ID]:
        concepts[factory._INITIAL_MODEL_ID] = 0
    unknown = [mid for mid in model_ids if mid not in concepts]
    present = True if alarm_concept in concepts.values() else None if unknown else False
    return counts, concepts, unknown, present


def _state_digest(value):
    """借用ownerの全fieldとtensorを比較する。objectの番地は含めない。"""
    digest = sha256()

    def visit(item):
        digest.update(str(type(item)).encode())
        if isinstance(item, torch.Tensor):
            digest.update(initialization._digest({"value": item}).encode())
            digest.update(str((item.requires_grad, item.is_leaf)).encode())
            if item.is_leaf and item.grad is not None:
                digest.update(initialization._digest({"gradient": item.grad}).encode())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode() + item.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, (set, frozenset)):
            for child in sorted(item, key=repr):
                visit(child)
        elif hasattr(item, "__dict__"):
            visit(vars(item))
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _correct(classifier, features, labels):
    with torch.no_grad():
        outputs = classifier(features)
        predicted = (
            predict_class_labels_from_prediction_scores(
                prediction_scores=outputs, class_count=classifier.class_count
            )
            .reshape(-1)
            .long()
        )
        return predicted.eq(labels.reshape(-1).long()).tolist()


def _evaluate(classifier, static, sequential, optimizer_settings):
    features, labels, _ = static
    correct = _correct(classifier, features, labels)
    losses = evaluate_classifier_per_sample_log_losses(
        classifier=classifier, input_features=features, observed_class_labels=labels
    ).tolist()
    classes = labels.reshape(-1).long().tolist()
    static_result = dict(
        accuracy=sum(correct) / len(correct),
        mean_log_loss=sum(losses) / len(losses),
        class_counts={
            str(c): dict(
                correct=sum(
                    ok for ok, label in zip(correct, classes, strict=True) if label == c
                ),
                count=classes.count(c),
            )
            for c in range(classifier.class_count)
        },
    )
    features, labels, _ = sequential
    learner = deepcopy(classifier)
    sequential_correct = []
    original_loss = training.evaluate_classifier_per_sample_log_losses

    def observe(**arguments):
        sequential_correct.extend(
            _correct(
                arguments["classifier"],
                arguments["input_features"],
                arguments["observed_class_labels"],
            )
        )
        return original_loss(**arguments)

    prefixes = []
    with initialization.patch.object(
        training, "evaluate_classifier_per_sample_log_losses", observe
    ):
        training._accumulate_prequential_code_length(
            learner=learner,
            input_features=features,
            observed_class_labels=labels,
            parameter_optimizer_settings=optimizer_settings,
            prefixes=prefixes,
        )
    return dict(
        static=static_result,
        sequential=dict(
            cumulative_accuracy=(
                np.cumsum(sequential_correct) / np.arange(1, len(prefixes) + 1)
            ).tolist(),
            cumulative_log_loss=prefixes,
            labels=labels.reshape(-1).long().tolist(),
            correct=sequential_correct,
        ),
    )


def _train_diagnostic(
    original_train, arguments, initial, mode, sample_arguments, concepts
):
    """製品の分割・反復・最良値復元を使い、S4では訓練datasetだけ拡張する。"""
    copied = deepcopy(arguments | {"initial_candidate_parameter_snapshot": initial})
    maximum = 100 if mode == "E100" else 300 if "E300" in mode else None
    if maximum is not None:
        copied["candidate_epoch_training_settings"] = replace(
            copied["candidate_epoch_training_settings"],
            maximum_epoch_count=maximum,
            consecutive_non_improving_epoch_limit=maximum,
        )
    original_dataset = epochs.TensorDataset
    original_randperm = epochs.randperm
    original_epoch = epochs._train_candidate_dataset_epochs
    original_update = epochs.perform_joint_model_parameter_update
    original_validation = epochs.evaluate_classifier_per_sample_bounded_losses
    partition = []
    validation_losses = []
    last_training_loss = []
    extra_counts = Counter()
    epoch_losses = []

    def randperm(count):
        indices = original_randperm(count)
        partition.extend(indices.tolist())
        return indices

    def dataset(features, labels):
        if mode.startswith("S4"):
            if not partition:
                raise ValueError("S4には製品の検証分割が必要です")
            training_indices = partition[len(partition) - len(features) :]
            extra_concepts = [concepts[i] for i in training_indices for _ in range(3)]
            extra_features, extra_labels, _ = _generate_samples(
                **sample_arguments, purpose="s4-training", concepts=extra_concepts
            )
            extra_counts.update(extra_concepts)
            features = torch.cat((features, extra_features))
            labels = torch.cat((labels, extra_labels))
        return original_dataset(features, labels)

    def epoch(**kwargs):
        epoch_losses.clear()
        result = original_epoch(**kwargs)
        last_training_loss[:] = [
            sum(loss * count for loss, count in epoch_losses)
            / sum(count for _, count in epoch_losses)
        ]
        return result

    def update(**kwargs):
        loss = original_update(**kwargs)
        count = sum(
            len(batch.input_features)
            for batch in kwargs["participating_training_batches"]
        )
        epoch_losses.append((loss, count))
        return loss

    def validation(**kwargs):
        result = original_validation(**kwargs)
        validation_losses.append(float(result.mean()))
        return result

    with (
        initialization.patch.object(epochs, "randperm", randperm),
        initialization.patch.object(epochs, "TensorDataset", dataset),
        initialization.patch.object(epochs, "_train_candidate_dataset_epochs", epoch),
        initialization.patch.object(
            epochs, "perform_joint_model_parameter_update", update
        ),
        initialization.patch.object(
            epochs, "evaluate_classifier_per_sample_bounded_losses", validation
        ),
    ):
        result = original_train(**copied)
    classifier = result.candidate_training_state.candidate_classifier
    settings = copied["candidate_epoch_training_settings"]
    accepted_validation_loss = float("inf")
    for loss in validation_losses:
        if loss < accepted_validation_loss - settings.minimum_validation_loss_decrease:
            accepted_validation_loss = loss
    return classifier, dict(
        completed_epoch_count=result.candidate_epoch_training_result.completed_epoch_count,
        best_validation_loss=accepted_validation_loss if validation_losses else None,
        last_training_loss=last_training_loss[-1] if last_training_loss else None,
        training_loss_kind="sample_weighted_mean_BCE_or_cross_entropy_during_last_epoch",
        validation_loss_kind="mean_bounded_loss",
        maximum_epoch_count=settings.maximum_epoch_count,
        consecutive_non_improving_epoch_limit=settings.consecutive_non_improving_epoch_limit,
        training_sample_indices=partition[
            max(1, round(len(concepts) * settings.validation_sample_fraction)) :
        ],
        added_sample_count=sum(extra_counts.values()),
        added_sample_counts_by_concept=dict(extra_counts),
        parameter_digest=_parameter_digest(classifier),
    )


@contextmanager
def candidate_start_diagnostics(*, dataset, run_seed, diagnostics):
    """選択の実体に紐づけ、候補学習直前の乱数を13候補それぞれへ戻す。"""
    original_select = processing.select_alarm_interval_assignment
    original_resolve = response.resolve_alarm_change_interval
    original_initialize = resolution.select_candidate_initial_parameter_snapshot
    original_train = resolution.train_candidate_on_alarm_change_interval
    original_streams = streams.build_client_observed_streams
    original_assemble = factory.assemble_fedsda_run_client
    original_accumulate = training._accumulate_prequential_code_length
    pending = {}
    active = []
    traces = {}
    clients = []
    diagnostics.update(schema_version=1, alarms=[])

    def assemble(**arguments):
        client = original_assemble(**arguments)
        clients.append(client)
        return client

    def observe_streams(**arguments):
        traces.update(
            {
                t.client_id: t.concept_ids_by_sample_index
                for t in arguments["evaluation_concept_traces"]
            }
        )
        return original_streams(**arguments)

    def select(**arguments):
        learners = []
        learner_choices = []
        assignment_inputs = []
        original_choose = mdl.select_new_model_learner
        original_assign = mdl.select_assignment_by_minimum_description_length

        def accumulate(**samples):
            initial = initialization.snapshot_classifier_parameters(
                classifier=samples["learner"]
            )
            losses = []
            original_loss = training.evaluate_classifier_per_sample_log_losses

            def observe_loss(**inputs):
                result = original_loss(**inputs)
                losses.extend(result.tolist())
                return result

            with initialization.patch.object(
                training, "evaluate_classifier_per_sample_log_losses", observe_loss
            ):
                value = original_accumulate(**samples)
            _require_equal(sum(losses), value, "学習者の損失の和が製品と一致しません")
            learners.append(
                dict(
                    initial=initial,
                    losses=losses,
                    labels=samples["observed_class_labels"].reshape(-1).long().tolist(),
                )
            )
            return value

        def choose(**kwargs):
            result = original_choose(**kwargs)
            learner_choices.append((kwargs, result))
            return result

        def assign(**kwargs):
            assignment_inputs.append(kwargs)
            return original_assign(**kwargs)

        with (
            initialization.patch.object(
                training, "_accumulate_prequential_code_length", accumulate
            ),
            initialization.patch.object(mdl, "select_new_model_learner", choose),
            initialization.patch.object(
                mdl, "select_assignment_by_minimum_description_length", assign
            ),
        ):
            chosen = original_select(**arguments)
        if chosen.selected_hypothesis == "new_model":
            if (
                arguments["change_interval_search"]
                != initialization.EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME
            ):
                raise ValueError("診断は製品既定の逆向き1本の探索だけに対応します")
            count = chosen.change_interval_sample_count
            if len(learners) != 2 or id(chosen) in pending:
                raise ValueError("2学習者または選択の対応が不正です")
            inputs = assignment_inputs[0]
            new_length = inputs[
                "new_model_code_lengths_by_change_interval_sample_count"
            ][count]
            total = (
                sum(inputs["current_model_log_losses"][:-count])
                + inputs["switching_code_length"]
            ) + new_length
            _require_equal(
                total, chosen.description_length, "選択の符号長が一致しません"
            )
            choice_arguments, choice = learner_choices[
                count - arguments["change_interval_grid_sample_count"]
            ]
            _require_equal(
                choice.description_length,
                new_length,
                "学習者の選択の符号長が一致しません",
            )
            names = [
                selection.REINITIALIZED_LEARNER_NAME,
                selection.WARM_STARTED_LEARNER_NAME,
            ]
            for learner, (name, length) in zip(
                learners,
                choice_arguments["prequential_code_lengths_by_learner_name"],
                strict=True,
            ):
                learner.update(
                    name=name,
                    losses=learner["losses"][:count],
                    labels=learner["labels"][:count],
                    product_code_length=length,
                )
                _require_equal(
                    sum(learner["losses"]),
                    length,
                    "選択した区間の学習者の符号長が一致しません",
                )
                expected_labels = [
                    int(o.training_sample.observed_class_labels.item())
                    for o in reversed(arguments["pending_sample_observations"][-count:])
                ]
                _require_equal(
                    learner["labels"],
                    expected_labels,
                    "学習者の標本の処理順が一致しません",
                )
            _require_equal(
                [x["name"] for x in learners], names, "学習者の処理順が不正です"
            )
            _require_equal(
                choice.learner_name,
                chosen.new_model_learner_name,
                "短かった学習者が一致しません",
            )
            pending[id(chosen)] = dict(
                selection=chosen,
                arguments=arguments,
                learners=learners,
                new_model_description_length=new_length,
                preceding_current_model_code_length=sum(
                    inputs["current_model_log_losses"][:-count]
                ),
                switching_code_length=inputs["switching_code_length"],
            )
        return chosen

    def resolve(**arguments):
        chosen = arguments.get("alarm_interval_assignment_selection")
        if chosen is not None and chosen.selected_hypothesis != "new_model":
            return original_resolve(**arguments)
        entry = pending.get(id(chosen))
        if entry is None or entry["selection"] is not chosen:
            raise ValueError("解決の選択の実体が一致しません")
        _require_equal(
            arguments["proposal_sample_index"],
            entry["arguments"]["alarm_sample_index"],
            "警報の位置が一致しません",
        )
        _require_equal(
            len(arguments["change_interval_training_samples"]),
            chosen.change_interval_sample_count,
            "区間の長さが一致しません",
        )
        expected_concepts = tuple(
            o.observed_concept_id
            for o in entry["arguments"]["pending_sample_observations"][
                -chosen.change_interval_sample_count :
            ]
        )
        _require_equal(
            arguments["change_interval_sample_concept_ids"],
            expected_concepts,
            "区間の真の概念が一致しません",
        )
        if active:
            raise ValueError("診断の解決を入れ子にできません")
        active.append((entry, arguments))
        try:
            return original_resolve(**arguments)
        finally:
            active.pop()
            del pending[id(chosen)]

    def initialize(**arguments):
        initial = original_initialize(**arguments)
        if not active or "starts" in active[-1][0]:
            raise ValueError("初期化と選択の対応が不正です")
        entry, resolve_arguments = active[-1]
        current_id = arguments["current_training_model_id"]
        current = arguments["available_parameter_snapshots_by_model_id"][current_id]
        classifier = (
            resolve_arguments["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        seed = initialization._seed(
            client_id=entry["arguments"]["client_id"],
            alarm_sample_index=entry["arguments"]["alarm_sample_index"],
            count=entry["selection"].change_interval_sample_count,
            search=entry["arguments"]["change_interval_search"],
        )
        reinitialized = initialization.reinitialized_candidate_parameter_snapshot(
            classifier=classifier,
            initialization_seed=seed,
            optimizer_settings=resolve_arguments["parameter_optimizer_settings"],
        )
        c_digest = initialization._digest(
            initialization._concept_specific(reinitialized)
        )
        learner_digest = initialization._digest(
            initialization._concept_specific(entry["learners"][0]["initial"])
        )
        _require_equal(c_digest, learner_digest, "Cと学習者の初期値が一致しません")
        entry.update(
            starts=dict(A=initial, B=current, C=reinitialized),
            initialization_seed=seed,
            c_initial_digest=c_digest,
            learner_initial_digest=learner_digest,
        )
        return initial

    def train(**arguments):
        if not active or "starts" not in active[-1][0]:
            raise ValueError("学習と初期化の対応が不正です")
        entry, resolve_arguments = active[-1]
        select_arguments = entry["arguments"]
        client_id, alarm = (
            select_arguments["client_id"],
            select_arguments["alarm_sample_index"],
        )
        concepts = list(resolve_arguments["change_interval_sample_concept_ids"])
        if any(c is None for c in concepts):
            raise ValueError("診断には区間の真の概念が必要です")
        count = len(concepts)
        interval_observations = select_arguments["pending_sample_observations"][-count:]
        preceding_index = interval_observations[0].sample_index - 1
        previous = traces[client_id][preceding_index] if preceding_index >= 0 else None
        sample_arguments = dict(
            dataset=dataset,
            run_seed=run_seed,
            client_id=client_id,
            alarm_sample_index=alarm,
        )
        states = _random_states()
        before = _state_digest(resolve_arguments)
        candidates = {}
        paused[0] = True
        try:
            static = _generate_samples(
                **sample_arguments,
                purpose="static-evaluation",
                concepts=[concepts[-1]] * 500,
            )
            sequential = _generate_samples(
                **sample_arguments,
                purpose="sequential-evaluation",
                concepts=[concepts[-1]] * 200,
            )
            for start, initial in entry["starts"].items():
                modes = (
                    ("T30", "E100", "E300", "S4-T30", "S4-E300")
                    if start != "B"
                    else ("T30", "E100", "E300")
                )
                for mode in modes:
                    _restore(states)
                    classifier, record = _train_diagnostic(
                        original_train,
                        arguments,
                        initial,
                        mode,
                        sample_arguments,
                        concepts,
                    )
                    record["initial_parameter_digest"] = initialization._digest(initial)
                    record["initial_concept_specific_parameter_digest"] = (
                        initialization._digest(
                            initialization._concept_specific(initial)
                        )
                    )
                    record.update(
                        _evaluate(
                            classifier,
                            static,
                            sequential,
                            arguments["parameter_optimizer_settings"],
                        )
                    )
                    candidates[f"{start}-{mode}"] = record
            _require_equal(
                _state_digest(resolve_arguments),
                before,
                "診断が借用ownerを変更しました",
            )
        finally:
            _restore(states)
            paused[0] = False
        result = original_train(**arguments)
        product_digest = _parameter_digest(
            result.candidate_training_state.candidate_classifier
        )
        _require_equal(
            candidates["A-T30"]["parameter_digest"],
            product_digest,
            "A・T30と製品の候補が一致しません",
        )
        registry = resolve_arguments["held_model_training_state_registry"]
        counts_store = resolve_arguments["model_training_and_assignment_counts_store"]
        held_counts = {
            s.model_id: counts_store.get_model_assigned_sample_concept_counts(
                model_id=s.model_id
            )
            for s in registry.snapshot_ordered_held_model_training_states()
        }
        pooled_counts, held_concepts, unknown_models, already_held = (
            _held_concept_evidence(
                clients, tuple(held_counts), concepts[-1], local_client_id=client_id
            )
        )
        learner_records = [
            {k: v for k, v in learner.items() if k != "initial"}
            | dict(
                concepts=list(reversed(concepts)),
                sample_indices=[
                    o.sample_index for o in reversed(interval_observations)
                ],
            )
            for learner in entry["learners"]
        ]
        diagnostics["alarms"].append(
            dict(
                client_id=client_id,
                alarm_sample_index=alarm,
                change_interval_sample_count=count,
                new_model_learner_name=entry["selection"].new_model_learner_name,
                selection_description_length=entry["selection"].description_length,
                new_model_description_length=entry["new_model_description_length"],
                preceding_current_model_code_length=entry[
                    "preceding_current_model_code_length"
                ],
                switching_code_length=entry["switching_code_length"],
                interval_concepts=concepts,
                preceding_concept=previous,
                alarm_concept=concepts[-1],
                held_model_assigned_concept_counts=held_counts,
                held_model_pooled_assigned_concept_counts=pooled_counts,
                held_model_dominant_concepts=held_concepts,
                held_model_ids_with_unknown_concept=unknown_models,
                already_held_alarm_concept=already_held,
                already_held_concept_definition="unique majority of assignments: global IDs pooled across clients, temporary IDs local; unassigned initial model means concept 0; unresolved presence is null",
                changed_classes=_changed_classes(
                    dataset,
                    previous,
                    concepts[-1],
                    result.candidate_training_state.candidate_classifier.class_count,
                ),
                initialization_seed=entry["initialization_seed"],
                c_initial_digest=entry["c_initial_digest"],
                learner_initial_digest=entry["learner_initial_digest"],
                product_candidate_digest=product_digest,
                evaluation_seeds=dict(
                    static=static[2],
                    sequential=sequential[2],
                    s4=_seed(run_seed, client_id, alarm, "s4-training"),
                ),
                learners=learner_records,
                candidates=candidates,
            )
        )
        return result

    with (
        _model_computation_gate() as paused,
        initialization.patch.object(factory, "assemble_fedsda_run_client", assemble),
        initialization.patch.object(
            streams, "build_client_observed_streams", observe_streams
        ),
        initialization.patch.object(
            processing, "select_alarm_interval_assignment", select
        ),
        initialization.patch.object(response, "resolve_alarm_change_interval", resolve),
        initialization.patch.object(
            resolution, "select_candidate_initial_parameter_snapshot", initialize
        ),
        initialization.patch.object(
            resolution, "train_candidate_on_alarm_change_interval", train
        ),
    ):
        yield
        if pending:
            raise ValueError("未解決の選択が残っています")
